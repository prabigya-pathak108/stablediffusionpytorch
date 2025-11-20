# Stable Diffusion PyTorch

A PyTorch implementation of Stable Diffusion with multiple sampling algorithms.

## Features

- Multiple sampling algorithms: DDPM, K-Euler, K-LMS
- Text-to-image and image-to-image generation
- Classifier-free guidance (CFG)
- Configurable dimensions and inference steps

## Setup

1. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model data:
```bash
wget https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar
tar -xf data.v20221029.tar
```

## Usage

```python
from stable_diffusion.pipeline import StableDiffusionPipeline

# Initialize pipeline
pipeline = StableDiffusionPipeline(device="cuda")

# Generate image from text
images = pipeline.generate(
    prompt="a beautiful landscape with mountains",
    sampler="k_lms",
    cfg_scale=7.5,
    n_inference_steps=50
)

# Save image
images[0].save("output.png")
```

## Samplers

- `ddpm`: Original DDPM sampler (stochastic)
- `k_euler`: K-Euler sampler (deterministic) 
- `k_lms`: K-LMS sampler (improved quality)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- transformers
- numpy
- Pillow