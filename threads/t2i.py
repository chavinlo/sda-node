import torch
from queue import Queue
import base64
from PIL import Image
from .trt.models import CLIP, UNet, VAE
import numpy as np
import json
from polygraphy import cuda
import time
import torch
from transformers import CLIPTokenizer, CLIPTextModel
import tensorrt as trt
from .trt.utilities import Engine, DPMScheduler, LMSDiscreteScheduler, save_image, TRT_LOGGER
import traceback
from diffusers import EulerAncestralDiscreteScheduler, DDPMScheduler
from io import BytesIO
import logging
from .lpw.enc import *
import os

log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.logging.getLogger()

def image_generator(
    config,
    request_queue: Queue,
    image_queue: Queue,
    ):

    sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}

    verbose = True
    max_batch_size = 4
    device = 'cuda'

    models = {
    'clip': CLIP(device=device, verbose=verbose, max_batch_size=max_batch_size, hf_token=""),
    'unet_fp16': UNet(fp16=True, device=device, verbose=verbose, max_batch_size=max_batch_size, hf_token=""),
    'vae': VAE(device=device, verbose=verbose, max_batch_size=max_batch_size, hf_token="")
    }
    engine = {}
    stream = cuda.Stream()

    for model_name, obj in models.items():
        indiv_engine = Engine(model_name, config["model_path"])
        indiv_engine.activate()
        engine[model_name] = indiv_engine

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    schedulers = {
        "DPMS": {},
        "LMSD": {},
        "EULER-A": {'obj': EulerAncestralDiscreteScheduler(**sched_opts)}
    }

    schedulers['DPMS']['config'] = {'type': 'accelerated'}
    schedulers['LMSD']['config'] = {'type': 'accelerated'}
    schedulers['EULER-A']['config'] = {'type': 'diffusers'}

    for i in range(config['schedulers']['max']):
        if i >= config['schedulers']['min'] and i % config['schedulers']['separator'] == 0:
            logger.info(f"Creating schedulers for {i} steps.")
            #DPMS
            temporary_scheduler = DPMScheduler(device=device, **sched_opts)
            temporary_scheduler.set_timesteps(i)
            temporary_scheduler.configure()
            schedulers['DPMS'][i] = temporary_scheduler

            #LMSD 
            temporary_scheduler = LMSDiscreteScheduler(device=device, **sched_opts)
            temporary_scheduler.set_timesteps(i)
            temporary_scheduler.configure()
            schedulers['LMSD'][i] = temporary_scheduler

    def runEngine(model_name, feed_dict):
        indiv_engine = engine[model_name]
        return indiv_engine.infer(feed_dict, stream)

    def imgq(status: str, content):
        response = {
            "status": status,
            "content": content
        }
        image_queue.put(response)

    lpw_pipe = LongPromptWeightingPipeline(
        tokenizer=tokenizer,
        text_encoder=CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    )

    logger.info("Prepartions ready")
    logger.debug(schedulers)

    while True:
        request = request_queue.get()
        data = request
        try:
            start_time = time.time()
            logger.info("got request!", data)
            prompt = data['prompt']
            negative_prompt = data['negprompt']
            image_height = data['height']
            image_width = data['width']
            scheduler = data['scheduler']
            steps = data['steps']
            cfg = data['cfg']
            seed = data['seed']
            mode = data['mode']
            lpw = data['lpw']

            #TODO: a little bit confusing
            logger.debug("scheduler section")
            logger.debug(scheduler)
            if scheduler in schedulers:
                scheduler_type = schedulers[scheduler]['config']['type']
                if scheduler_type == 'accelerated':
                    scheduler = schedulers[scheduler]
                    if steps in scheduler:
                        scheduler = scheduler[steps]
                    else:
                        imgq('fail', f'scheduler w/ conf {str(steps)} not found')
                        continue
                elif scheduler_type == 'diffusers':
                    scheduler = schedulers[scheduler]['obj']
                    scheduler.set_timesteps(steps, device=device)
            else:
                imgq('fail', f'scheduler {scheduler} not found')
                continue

            batch_size = 1
            
            # Spatial dimensions of latent tensor
            latent_height = image_height // 8
            latent_width = image_width // 8

            logger.debug("alloc buffer")
            # Allocate buffers for TensorRT engine bindings
            for model_name, obj in models.items():
                engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=device)

            logger.debug("seed gen")
            # Seeds
            generator = None
            if seed != -1:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            preparation_time = time.time()

            logger.debug("running pipe")
            # Run Stable Diffusion pipeline
            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER) as runtime:
                # latents need to be generated on the target device
                unet_channels = 4 # unet.in_channels
                latents_shape = (batch_size, unet_channels, latent_height, latent_width)
                logger.debug("latents_shape:", latents_shape)
                latents_dtype = torch.float32 # text_embeddings.dtype
                latents = torch.randn(latents_shape, device=device, dtype=latents_dtype, generator=generator)

                # Scale the initial noise by the standard deviation required by the scheduler
                latents = latents * scheduler.init_noise_sigma

                torch.cuda.synchronize()

                if lpw is False:
                    # From Here
                    logger.debug("tokenize")
                    # Tokenize input
                    text_input_ids = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.type(torch.int32).to(device)

                    logger.debug("clip encoder")
                    # CLIP text encoder
                    text_input_ids_inp = cuda.DeviceView(ptr=text_input_ids.data_ptr(), shape=text_input_ids.shape, dtype=np.int32)
                    text_embeddings = runEngine('clip', {"input_ids": text_input_ids_inp})['text_embeddings']

                    # Duplicate text embeddings for each generation per prompt
                    bs_embed, seq_len, _ = text_embeddings.shape
                    text_embeddings = text_embeddings.repeat(1, 1, 1)
                    text_embeddings = text_embeddings.view(bs_embed * 1, seq_len, -1)

                    max_length = text_input_ids.shape[-1]
                    uncond_input_ids = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.type(torch.int32).to(device)
                    uncond_input_ids_inp = cuda.DeviceView(ptr=uncond_input_ids.data_ptr(), shape=uncond_input_ids.shape, dtype=np.int32)
                    uncond_embeddings = runEngine('clip', {"input_ids": uncond_input_ids_inp})['text_embeddings']

                    # Duplicate unconditional embeddings for each generation per prompt
                    seq_len = uncond_embeddings.shape[1]
                    uncond_embeddings = uncond_embeddings.repeat(1, 1, 1)
                    uncond_embeddings = uncond_embeddings.view(batch_size * 1, seq_len, -1)

                    # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                    text_embeddings = text_embeddings.to(dtype=torch.float16) # <- result (?) of the lpw
                    # To here

                    logger.debug("old")
                    
                    logger.debug(text_embeddings)
                    logger.debug(text_embeddings.shape)
                else:
                    text_embeddings = lpw_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=cfg,
                        num_images_per_prompt=1,
                        max_embeddings_multiples=3
                    )
                    text_embeddings = text_embeddings.to(dtype=torch.float16)

                    logger.debug("novo")
                    
                    logger.debug(text_embeddings)
                    logger.debug(text_embeddings.shape)

                clip_time = time.time()

                logger.debug("denoising")
                for step_index, timestep in enumerate(scheduler.timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    # LMSDiscreteScheduler.scale_model_input()
                    if scheduler_type == 'accelerated':
                        latent_model_input = scheduler.scale_model_input(latent_model_input, step_index)
                    elif scheduler_type == 'diffusers':
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

                    # predict the noise residual
                    dtype = np.float16
                    if timestep.dtype != torch.float32:
                        timestep_float = timestep.float()
                    else:
                        timestep_float = timestep
                    sample_inp = cuda.DeviceView(ptr=latent_model_input.data_ptr(), shape=latent_model_input.shape, dtype=np.float32)
                    timestep_inp = cuda.DeviceView(ptr=timestep_float.data_ptr(), shape=timestep_float.shape, dtype=np.float32)
                    embeddings_inp = cuda.DeviceView(ptr=text_embeddings.data_ptr(), shape=text_embeddings.shape, dtype=dtype)
                    noise_pred = runEngine('unet_fp16', {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']

                    # Perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                    #TensorRT schedulers return prev_sample by default, diffusers does not.
                    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
                    if scheduler_type == 'accelerated':
                        latents = scheduler.step(noise_pred, latents, step_index, timestep)
                    elif scheduler_type == 'diffusers':
                        #they also use a different order
                        latents = scheduler.step(
                            model_output=noise_pred, 
                            timestep=timestep, 
                            sample=latents,
                            generator=generator).prev_sample

                denoising_time = time.time()

                logger.debug("finished")
                latents = 1. / 0.18215 * latents
                
                sample_inp = cuda.DeviceView(ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32)
                images = runEngine('vae', {"latent": sample_inp})['images']

                vae_time = time.time()
                
                torch.cuda.synchronize()
                logger.debug("syncronized, converting to img")
                images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
                img = Image.fromarray(images[0])
                serving_time = time.time()
                if mode == 'file':
                    imgq('done', img)
                elif mode == 'json':
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=95)
                    imgq("done", {"time": serving_time - preparation_time, "img": base64.b64encode(buffered.getvalue()).decode('utf-8')})
                benchmark_time = {
                    "PREP": preparation_time - start_time,
                    "CLIP**" if lpw else "CLIP": clip_time - preparation_time,
                    f"UNET x {steps}***" if scheduler_type != "accelerated" else f'UNET x {steps}': denoising_time - clip_time,
                    "VAE*": vae_time - denoising_time,
                    "SERVING": serving_time - vae_time,
                    "TOTALCOM": serving_time - preparation_time,
                    "TOTAL": serving_time - start_time,
                }
                print("Benchs: (Check notes)")
                for i in benchmark_time:
                    print('| {:^14} | {:>9.2f} ms |'.format(i, int(benchmark_time[i]*1000)))
                print(f'w{image_width} x h{image_height}')
                print(f'lpw: {lpw}')
                print('scheduler: {}'.format(data['scheduler']))
                print('accelerated: {}'.format(True if scheduler_type == 'accelerated' else False))

        except Exception as e:
            traceback.print_exc()
            imgq('fail', f'general exception, got {str(e)}')
            continue

