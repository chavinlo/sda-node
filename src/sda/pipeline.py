#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuda import cudart
import gc
import numpy as np
import nvtx
import json
import os
import onnx
import copy
from polygraphy import cuda
import torch
from .utilities import Engine, device_view, save_image
from .utilities import DPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusionPipeline:
    """
    Stable Diffusion pipeline accelerated ysing TensorRT
    """
    def __init__(
        self,
        max_batch_size=16,
        device='cuda',
        verbose=False,
        min_image_shape=256,
        max_image_shape=1024,
        text_max_len=77,
        separator: int = 5,
        steps_min: int = 20,
        steps_max: int = 60
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                PyTorch device to run inference. Default: 'cuda'checkpoints.
            verbose (bool):
                Enable verbose logging.
            min_image_shape (int):
                Minimun image resolution for image generation.
            max_image_shape (int):
                Maximun  image resolution for image generation.
            text_max_len (int):
                Maximun length for prompt for image generation.
            separator (int):
                Separator for scheduler pre-computing.
            steps_min (int):
                Minimun step count for pre-computing.
            steps_max (int):
                Maximun step count for pre-computing.
        """

        self.separator = separator
        self.steps_min = steps_min
        self.steps_max = steps_max

        # assert guidance_scale > 1.0
        # self.guidance_scale = guidance_scale

        self.max_batch_size = max_batch_size
        self.min_batch_size = 1

        # Limit the workspace size for systems with GPU memory larger
        # than 6 GiB to silence OOM warnings from TensorRT optimizer.
        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6*GiB:
            activation_carveout = 4*GiB
            self.max_workspace_size = free_mem - activation_carveout
        else:
            self.max_workspace_size = 0

        # self.output_dir = output_dir
        # self.hf_token = hf_token
        self.device = device
        self.verbose = verbose

        self.min_image_shape = min_image_shape
        self.max_image_shape = max_image_shape
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.text_maxlen = text_max_len
        
        # Create schedulers for 1.x and 2.x

        v1_sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'prediction_type': 'epsilon'}
        v2_sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'prediction_type': 'v_prediction'}

        schedulers = dict()

        _stop_creating = False
        _curr_steps = 0

        print("init: Creating Schedulers...")

        while _stop_creating is False:
            _curr_steps += separator
            if _curr_steps >= self.steps_min and _curr_steps <= steps_max:
                schedulers[_curr_steps] = {
                        1: {
                            "DDIM": DDIMScheduler(device=self.device, **v1_sched_opts),
                            "DPM": DPMScheduler(device=self.device, **v1_sched_opts),
                            "EULER-A": EulerAncestralDiscreteScheduler(device=self.device, **v1_sched_opts),
                            "LMSD": LMSDiscreteScheduler(device=self.device, **v1_sched_opts),
                            "PNDM": PNDMScheduler(device=self.device, **v1_sched_opts)
                        },
                        2: {
                            "DDIM": DDIMScheduler(device=self.device, **v2_sched_opts),
                            "DPM": DPMScheduler(device=self.device, **v2_sched_opts),
                            "EULER-A": EulerAncestralDiscreteScheduler(device=self.device, **v2_sched_opts),
                            "LMSD": LMSDiscreteScheduler(device=self.device, **v2_sched_opts),
                            "PNDM": PNDMScheduler(device=self.device, **v2_sched_opts)
                        },
                    }
                
                for version in schedulers[_curr_steps]:
                    for scheduler_name in schedulers[_curr_steps][version]:
                        scheduler = schedulers[_curr_steps][version][scheduler_name]
                        scheduler.set_timesteps(_curr_steps)
                        scheduler.configure()

            elif _curr_steps > steps_max:
                _stop_creating = True

        self.schedulers = schedulers

        print("init: Finished creating schedulers")

        self.stream = None # loaded in loadResources()
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
        self.models = {} # loaded in loadEngines()
        self.engine = {} # loaded in loadEngines()

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        for engine in self.engine.values():
            del engine

        self.stream.free()
        del self.stream

    def loadEngine(
            self,
            path,
            alias: str = None
    ):
        """
        Loads a compiled model. Given path must also have it's json configuration pair.

        Args:
            path (str):
                Path pointing to the `.plan` file. Must also have a json configuration pair in the same directory under the same name.
            alias (str):
                Alias to give to the model. OPTIONAL. If not given the one in the configuration file will be assigned.
        """
        # path: points to a plan file, the json config file will also be obtained from this path

        # parameters derived from config json: version, max_batch_size,  (Y/N)
        # DO NOT LOAD TOKENIZER HERE

        plan_path = path
        cfg_path = os.path.splitext(plan_path)[0] + '.json'

        if os.path.exists(cfg_path) is False:
            raise OSError(f"loadEngine: Configuration path {str(cfg_path)} does not exists for model path {str(plan_path)}")
        
        cfg = json.load(open(cfg_path, "r"))

        """
        {
            "alias": "STABLE_DIFFUSION-2_1", # IMPORTANT: Syntax to use: "_" for spaces and "-" for type separation
            "type": "unet",
            "version": 2,
            "inpaint": false,
            "max_batch_size": 4,
            "fp16": true
        }
        """

        if alias is None:
            alias = cfg['alias']
            
        _alias = alias
        _type = cfg['type']

        """
        From now on self.models will be a container (dict) that will host the models in the following manner:

        {
            "unet": {
                "STABLE_DIFFUSION-2_1": {
                    "engine": Engine(),
                    "config": dict()
                    },
                "WAIFU_DIFFUSION-1_5": {
                    "engine": Engine(),
                    "config": dict()
                    },
            },
            "vae": {
                "STABLE_DIFFUSION-2_1": {
                    "engine": Engine(),
                    "config": dict()
                    },
            },
            "vaeencoder": {
                "WAIFU_DIFFUSION-1_5": {
                    "engine": Engine(),
                    "config": dict()
                    },
            },
            "textencoder": {
                "STABLE_DIFFUSION-2_1": {
                    "engine": Engine(),
                    "config": dict()
                }
            }
        }

        """
            
        if _type not in ['unet', 'vae', 'vaeencoder', 'textencoder']:
          raise ValueError(f'loadEngine: type {_type} is not in accepted types.')
          
        self.models[_type] = dict()
        self.models[_type][_alias] = dict()
        working_model = self.models[_type][_alias]

        working_model['config'] = cfg

        working_model['engine'] = Engine(plan_path)
        working_model['engine'].load()
        working_model['engine'].activate()

    def loadResource(self):
        self.stream = cuda.Stream()
        # generator is going to be set during generation (bcuz seed)
        # scheduler timesteps and configure are also going to be set during generation
        # events are just for logging... I think
        # buffer allocation is also done during generation because of dynamic resolution

    def _check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch_size and batch_size <= self.max_batch_size
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def _get_shape_dict(self, model_config, batch_size, image_height = None, image_width = None, clip_embedding_dim: int = None):
        #NOTE: clip_embedding_dim is only used for CLIP, so that we only have 1 CLIP allocated, but we can use it for SD 2.x (1024) and SD 1.x (768)
        """
        UNET:
        {
            "alias": "STABLE_DIFFUSION-2_1",
            "type": "unet",
            "version": 2,
            "inpaint": false,
            "max_batch_size": 4,
            "fp16": true
        }
        VAE:
            {
            "alias": "STABLE_DIFFUSION-2_1",
            "type": "vae"
        }
        """

        _alias = model_config['alias']
        _type = model_config['type']

        if _type not in ['unet', 'textencoder', 'vae', 'vaeencoder']:
            print(f"_get_shape_dict: model_type not in accepted models: {_type}")
            ValueError(f"_get_shape_dict: model_type not in accepted models: {_type}")

        if _type in ['unet', 'vae', 'vaeencoder']:
            if image_height is None or image_width is None:
                raise ValueError(f'_get_shape_dict: image_width ({image_width}) or image_height ({image_height}) is None')

            latent_height, latent_width = self._check_dims(batch_size, image_height, image_width)

        if _type == 'unet':
            _version = model_config['version']
            _inpaint = model_config['inpaint']
            _max_batch_size = model_config['max_batch_size']
            _fp16 = model_config['fp16']

            _embedding_dim = 768 if _version == 1 else 1024
            _unet_dim = 9 if _inpaint is True else 4
            return {
                'sample': (2*batch_size, _unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (2*batch_size, self.text_maxlen, _embedding_dim),
                'latent': (2*batch_size, 4, latent_height, latent_width)
            }
        
        if _type == 'textencoder':
            if clip_embedding_dim is None:
                raise ValueError(f"_get_shape_dict: clip_embedding_dim is None, which is not possible. Please configurate this to either 768 for SD 1.x or 1024 for SD 2.x")
            return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, clip_embedding_dim)
            }

        if _type == 'vae' or _type == 'vaeencoder':
            return {
                'latent': (batch_size, 4, latent_height, latent_width),
                'images': (batch_size, 3, image_height, image_width)
            }

    def allocateBuffers(self, model, device, batch_size, image_height, image_width):
        working_config = model['config']
        working_engine = model['engine']

        working_engine.allocate_buffers(
            shape_dict=self._get_shape_dict(
                working_config,
                batch_size, 
                image_height, 
                image_width), 
            device=device)
        
    def clipAllocateBuffers(self, model, device, batch_size, clip_embedding_dim):
        working_config = model['config']
        working_engine = model['engine']
        
        working_engine.allocate_buffers(
            shape_dict=self._get_shape_dict(
                working_config,
                batch_size,
                clip_embedding_dim=clip_embedding_dim), 
                device=device)
        
    def get_model_config(self, model, model_type):
        if not model in self.models[model_type]:
            raise ValueError(f"get_model_config: Model {str(model)} not found in {str(model_type)} sector")
        
        return self.models[model_type][model]['config']
        
    def grab_scheduler(self, steps:int, version: int, scheduler_name: str):
        _final_steps = min(max(round(steps / 5) * 5, 20), 60)
        if steps != _final_steps:
            print(f"grab_scheduler: Warning, step count was rounder from {str(steps)} to {str(_final_steps)}")
        self.scheduler = self.schedulers[steps][version][scheduler_name]

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, generator):
        # What I've changed: added direct generator passing and scheduler
        scheduler = self.scheduler
        latents_dtype = torch.float32 # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    #DONE
    def initialize_timesteps(self, timesteps, strength):
        scheduler = self.scheduler
        scheduler.set_timesteps(timesteps)
        offset = scheduler.steps_offset if hasattr(scheduler, "steps_offset") else 0
        init_timestep = int(timesteps * strength) + offset
        init_timestep = min(init_timestep, timesteps)
        t_start = max(timesteps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:].to(self.device)
        return timesteps, t_start

    #DONE, literally just removing the profiler
    def preprocess_images(self, batch_size, images=()):
        init_images=[]
        for image in images:
            image = image.to(self.device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images .append(image)
        return tuple(init_images)

    def encode_prompt(self, prompt, negative_prompt, textencoder: str, clip_embedding_dim: int):
        if clip_embedding_dim not in [768, 1024]:
            raise ValueError(f"encode_prompt: Invalid clip_embedding_dim, {clip_embedding_dim}")

        #IMPORTANT: allocate buffers
        self.clipAllocateBuffers(
            model=self.models['textencoder'][textencoder],
            device="cuda",
            batch_size=1,
            clip_embedding_dim=clip_embedding_dim
        )

        #text_encoder be the name of the text_encoder to choose
        textencoder_model = self.models['textencoder'][textencoder]['engine']
        # Tokenize prompt
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)

        text_input_ids_inp = device_view(text_input_ids)
        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = textencoder_model.infer({"input_ids": text_input_ids_inp}, self.stream)['text_embeddings'].clone()

        # Tokenize negative prompt
        uncond_input_ids = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        uncond_input_ids_inp = device_view(uncond_input_ids)
        uncond_embeddings = textencoder_model.infer({"input_ids": uncond_input_ids_inp}, self.stream)['text_embeddings']

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings

    def denoise_latent(self, unet:str, latents, text_embeddings, guidance_scale, timesteps=None, step_offset=0, mask=None, masked_image_latents=None):
        #unet is the alias of the unet model to chosse
        #same with scheduler
        #TODO: maybe add every 5 steps schedulers? iirc configuring it takes a while

        unet_entry = self.models['unet'][unet]
        unet_config = unet_entry['config']
        unet_model = unet_entry['engine']

        version = unet_config['version']

        scheduler = self.scheduler

        res_height = latents.shape[2] * 8
        res_width = latents.shape[3] * 8

        self.allocateBuffers(
            model=unet_entry,
            device="cuda",
            batch_size=1,
            image_height=res_height,
            image_width=res_width
        )

        if not isinstance(timesteps, torch.Tensor):
            timesteps = scheduler.timesteps

        #TODO: Are you sure this is the way timesteps is used? what about strngth and that sort of things?
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, step_offset + step_index, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            embeddings_dtype = np.float16
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = unet_model.infer({"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp}, self.stream)['latent']

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

        latents = 1. / 0.18215 * latents
        return latents

    # I2I only
    def encode_image(self, init_image, vaeencoder_name: str):
        vaeencoder_model = self.models['vaeencoder'][vaeencoder_name]['engine']
        init_latents = vaeencoder_model.infer({"images": device_view(init_image)}, self.stream)['latent']
        init_latents = 0.18215 * init_latents
        return init_latents

    def decode_latent(self, latents, vae_name:str):
        vae_model = self.models['vae'][vae_name]['engine']

        #TODO: fix all this latent shape and image resolution shit, it must be organized not scattered on every single function :/

        res_height = latents.shape[2] * 8
        res_width = latents.shape[3] * 8

        self.allocateBuffers(
            model=self.models['vae'][vae_name],
            device="cuda",
            batch_size=1,
            image_height=res_height,
            image_width=res_width
        )
        images = vae_model.infer({"latent": device_view(latents)}, self.stream)['images']
        return images

    def save_image(self, images, pipeline, prompt):
            # Save image
            image_name_prefix = pipeline+'-fp16'+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'
            save_image(images, self.output_dir, image_name_prefix)
