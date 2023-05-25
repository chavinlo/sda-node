import argparse
from cuda import cudart
import tensorrt as trt
import logging
import numpy as np
import nvtx
import time
import torch
import tensorrt as trt
from .utilities import TRT_LOGGER
from .pipeline import StableDiffusionPipeline
from PIL import Image

class DiffusionModePipeline(StableDiffusionPipeline):
    """
    Pipeline for diffusion mode models
    """
    def __init__(
        self,
        *args, **kwargs
    ):
        """
        Initializes the Txt2Img Diffusion pipeline.
        Note the hidden kwargs
        Below are derived kwargs from the StableDiffusionPipeline Class:

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
        super(DiffusionModePipeline, self).__init__(*args, **kwargs)

    def get_pil(self, images):
        images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        return Image.fromarray(images[0])

    def infer(
        self,
        prompt,
        negative_prompt,
        steps,
        models: dict,
        scheduler: int,
        cfg,
        image_height,
        image_width,
        seed=None,
    ):
        """
        Run the diffusion pipeline.

        Args:
            steps (int):
                Number of denoising steps.
            models (dict):
                dictionary containing names of the models to use in format: {"unet": str, "vae": str}
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
        """
        assert len(prompt) == len(negative_prompt)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):

            # Seed Generator
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)
            else:
                generator = None

            # Grab configurations
            _tmp_configs = dict()
            for model_type, model_name in models.items():
                _tmp_configs[model_type] = self.get_model_config(model_name, model_type)

            # Select Scheduler
            self.grab_scheduler(
                steps=steps, 
                version=_tmp_configs['unet']['version'],
                scheduler_name=scheduler
                )

            # Pre-initialize latents
            latents = self.initialize_latents( \
                batch_size=len(prompt), \
                unet_channels=4, \
                latent_height=(image_height // 8), \
                latent_width=(image_width // 8),
                generator=generator
            )

            print(768 if _tmp_configs['unet']['version'] == 1 else 1024)
            print(_tmp_configs['unet']['version'])

            # CLIP text encoder
            text_embeddings = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                textencoder=models['textencoder'],
                clip_embedding_dim=768 if _tmp_configs['unet']['version'] == 1 else 1024)

            # UNet denoiser
            latents = self.denoise_latent(
                unet=models['unet'],
                latents=latents,
                text_embeddings=text_embeddings,
                guidance_scale=cfg,
            )

            # VAE decode latent
            images = self.decode_latent(
                latents=latents,
                vae_name=models['vae']
            )

            return self.get_pil(images)