# Benchmarks:
The following benchmarks were ran on A100 instance with the following specs:

```
OS: Ubuntu 20.04.5 LTS focal x86_64 
CPU: AMD EPYC-Milan (30) @ 2.449GHz 
Memory: 11000MiB / 120763MiB
GPU: NVIDIA A100-SXM4-80GB

NVIDIA-SMI: 
Driver Version: 520.61.05
CUDA Version: 11.8

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0

APT:
TensorRT Version: 8.5.2.2

PIP:
tensorrt 8.5.2.2
```

## Bench Format:

```
|      PREP      |      X.00 ms |
|      CLIP      |      X.00 ms |
|  UNET x 25***  |      X.00 ms |
|      VAE*      |      X.00 ms |
|    SERVING     |      X.00 ms |
|    TOTALCOM    |      X.00 ms |
|     TOTAL      |      X.00 ms |
w(mult8) x h(mult8)
lpw: bool
scheduler: str
accelerated: bool
```

PREP: Preparation stage, obtain request, allocate buffer, & create seed
CLIP: Process prompt and create text embeddings
UNET: Unet stage, denoising loop
VAE: VAE stage
SERVING: Serving stage, convert to image, and send to img queue

TOTALCOM: Total Compute (-PREP)
TOTAL: Everything summed up

lpw: wether Long and Extended Prompt is enabled
scheduler: scheduler in use
accelerated: wether scheduler is from tensorrt (accelerated) or from diffusers (non-accelerated and slower)

### NOTES:

*VAE Note: I am certain that the VAE measurement is not true. On the TensorRT demo, it had and average of 14.5ms/img @ 512px

**CLIP Note: When using the Long and Extended Prompt module, normal (pytorch) CLIP inference is used. You can disable this. In the future using accelerated CLIP along with LPW will be possible. (It's just a 2 line change, but a ton of bugs)

***UNET Note: Certain Schedulers (Such as Euler A) are imported from diffusers, where as others are imported from tensorRT which "improves efficiency by precomputing the coefficients for the linear multistep method".

These notes only apply when the "*" is present.

## 512x512 PX

### LPW Disabled
```
|      PREP      |      2.00 ms |
|      CLIP      |      6.00 ms |
|   UNET x 25    |    406.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     21.00 ms |
|    TOTALCOM    |    435.00 ms |
|     TOTAL      |    437.00 ms |
w512 x h512
lpw: False
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      2.00 ms |
|      CLIP      |      5.00 ms |
|   UNET x 25    |    384.00 ms |
|      VAE*      |      3.00 ms |
|    SERVING     |     44.00 ms |
|    TOTALCOM    |    436.00 ms |
|     TOTAL      |    439.00 ms |
w512 x h512
lpw: False
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      3.00 ms |
|      CLIP      |      7.00 ms |
|  UNET x 25***  |    412.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     21.00 ms |
|    TOTALCOM    |    441.00 ms |
|     TOTAL      |    444.00 ms |
w512 x h512
lpw: False
scheduler: EULER-A
accelerated: False
```

### LPW Enabled

```
|      PREP      |      2.00 ms |
|     CLIP**     |     24.00 ms |
|   UNET x 25    |    417.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     21.00 ms |
|    TOTALCOM    |    463.00 ms |
|     TOTAL      |    466.00 ms |
w512 x h512
lpw: True
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      2.00 ms |
|     CLIP**     |     26.00 ms |
|   UNET x 25    |    386.00 ms |
|      VAE*      |      3.00 ms |
|    SERVING     |     42.00 ms |
|    TOTALCOM    |    459.00 ms |
|     TOTAL      |    462.00 ms |
w512 x h512
lpw: True
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      3.00 ms |
|     CLIP**     |     38.00 ms |
|  UNET x 25***  |    424.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     20.00 ms |
|    TOTALCOM    |    484.00 ms |
|     TOTAL      |    487.00 ms |
w512 x h512
lpw: True
scheduler: EULER-A
accelerated: False
```

## 768x768 PX

### LPW Disabled

```
|      PREP      |      2.00 ms |
|      CLIP      |      7.00 ms |
|   UNET x 25    |    942.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     57.00 ms |
|    TOTALCOM    |   1008.00 ms |
|     TOTAL      |   1011.00 ms |
w768 x h768
lpw: False
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      1.00 ms |
|      CLIP      |      4.00 ms |
|   UNET x 25    |    892.00 ms |
|      VAE*      |      6.00 ms |
|    SERVING     |    112.00 ms |
|    TOTALCOM    |   1015.00 ms |
|     TOTAL      |   1017.00 ms |
w768 x h768
lpw: False
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      2.00 ms |
|      CLIP      |      5.00 ms |
|  UNET x 25***  |    975.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     58.00 ms |
|    TOTALCOM    |   1040.00 ms |
|     TOTAL      |   1043.00 ms |
w768 x h768
lpw: False
scheduler: EULER-A
accelerated: False
```

### LPW Enabled

```
|      PREP      |      2.00 ms |
|     CLIP**     |     23.00 ms |
|   UNET x 25    |    968.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     57.00 ms |
|    TOTALCOM    |   1050.00 ms |
|     TOTAL      |   1053.00 ms |
w768 x h768
lpw: True
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      3.00 ms |
|     CLIP**     |     22.00 ms |
|   UNET x 25    |    873.00 ms |
|      VAE*      |      6.00 ms |
|    SERVING     |    111.00 ms |
|    TOTALCOM    |   1013.00 ms |
|     TOTAL      |   1016.00 ms |
w768 x h768
lpw: True
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      3.00 ms |
|     CLIP**     |     23.00 ms |
|  UNET x 25***  |    937.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     58.00 ms |
|    TOTALCOM    |   1020.00 ms |
|     TOTAL      |   1024.00 ms |
w768 x h768
lpw: True
scheduler: EULER-A
accelerated: False
```

## 1024x1024 PX

### LPW Disabled

```
|      PREP      |      7.00 ms |
|      CLIP      |      6.00 ms |
|   UNET x 25    |   2052.00 ms |
|      VAE*      |      2.00 ms |
|    SERVING     |     97.00 ms |
|    TOTALCOM    |   2159.00 ms |
|     TOTAL      |   2166.00 ms |
w1024 x h1024
lpw: False
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      3.00 ms |
|      CLIP      |      6.00 ms |
|   UNET x 25    |   1919.00 ms |
|      VAE*      |     10.00 ms |
|    SERVING     |    216.00 ms |
|    TOTALCOM    |   2152.00 ms |
|     TOTAL      |   2156.00 ms |
w1024 x h1024
lpw: False
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      3.00 ms |
|      CLIP      |      7.00 ms |
|  UNET x 25***  |   2044.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     95.00 ms |
|    TOTALCOM    |   2148.00 ms |
|     TOTAL      |   2152.00 ms |
w1024 x h1024
lpw: False
scheduler: EULER-A
accelerated: False
```

### LPW Enabled

```
|      PREP      |      3.00 ms |
|     CLIP**     |     21.00 ms |
|   UNET x 25    |   2034.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     95.00 ms |
|    TOTALCOM    |   2152.00 ms |
|     TOTAL      |   2155.00 ms |
w1024 x h1024
lpw: True
scheduler: DPMS
accelerated: True
```

```
|      PREP      |      3.00 ms |
|     CLIP**     |     23.00 ms |
|   UNET x 25    |   1903.00 ms |
|      VAE*      |     10.00 ms |
|    SERVING     |    218.00 ms |
|    TOTALCOM    |   2155.00 ms |
|     TOTAL      |   2158.00 ms |
w1024 x h1024
lpw: True
scheduler: LMSD
accelerated: True
```

```
|      PREP      |      3.00 ms |
|     CLIP**     |     29.00 ms |
|  UNET x 25***  |   2032.00 ms |
|      VAE*      |      0.00 ms |
|    SERVING     |     95.00 ms |
|    TOTALCOM    |   2158.00 ms |
|     TOTAL      |   2162.00 ms |
w1024 x h1024
lpw: True
scheduler: EULER-A
accelerated: False
```