# SDA: Node
## Stable Diffusion Accelerated
## 25 steps in less than a second!
### Special thanks to Redmond AI for providing compute for testing

This is the Node module for Stable Diffusion Accelerated. Using TensorRT, we can achieve speeds up to 4 times faster than native PyTorch. 

Based on the Demo provided by NVIDIA, we (I) extended it's capabilities, some of them are:

* API for inference
* Weighted prompts
* More Schedulers
* Benchmarking
* More Step Counts

## How it works & where to get models

TensorRT optimizes the SD model by compiling it into a highly optimized version that can be run on NVIDIA GPUs. This adds some limitations such as limited batch size and resolutions (up to 1024px). It optimizes the CLIP, UNET, and VAE.

You can download pre-compiled models from our HuggingFace' TensorRT repository:
https://huggingface.co/tensorrt

# License:
The Software is intended for individual use only. Any use by groups such as companies, large communities, commercial, for-profit entities, etc. must be approved before-hand by the Copyright Owner. This includes but is not limited to: Discord Bots, Software-as-a-Service, etc.