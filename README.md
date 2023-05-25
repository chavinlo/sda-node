# Stable Diffusion Accelerated (SDA)

Note: Most of the code is borrowed from nvidia's repo

This is a simplified library to use stable diffusion with TensorRT

# Usage

## Import

```python
from sda.diffusion_mode_pipeline import DiffusionModePipeline
```

## Initiate Pipeline

```python
pipe = DiffusionModePipeline(
    max_batch_size=16,
    device="cuda",
    verbose=False,
    min_image_shape=256,
    max_image_shape=1024,
    text_max_len=77
)
```

Remember that the models must match the configuration

## Load Models

```python
pipe.loadEngine(
    path="/shared/models/tensor/unet.plan",
    alias="SD2"
)

pipe.loadEngine(
    path="/shared/models/tensor/clip.plan",
    alias="SD2"
)

pipe.loadEngine(
    path="/shared/models/tensor/vae.plan",
    alias="SD2"
)
```

You can load multiple models on a single GPU

## Prepare and Generate

```python
img = pipe.infer(
    prompt=["A car"],
    negative_prompt=[""],
    steps=50,
    models={"unet": "SD2", "textencoder": "SD2", "vae": "SD2"},
    scheduler="DDIM",
    cfg=7.5,
    image_height=512,
    image_width=512,
    seed=54321
)

img.save("output.png")
```

# Configuration File

The compiler isn't here yet, but you can just run nvidia's repo for now, and then add a json file right next to the models.

For example, for `anime_unet.plan`, `anime_unet.json` should have:

```json
{
	"alias": "STABLE_DIFFUSION-2_1",
	"type": "unet",
	"version": 2,
	"inpaint": false,
	"max_batch_size": 4,
	"fp16": true
}
```

Adjust it to the configuration used during compilation