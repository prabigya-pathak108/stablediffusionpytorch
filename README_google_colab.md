# Stable Diffusion PyTorch - Google Colab Guide

This guide shows how to run Stable Diffusion PyTorch in Google Colab with proper setup for Drive integration and local runtime.

## Setup in Google Colab

### 1. Mount Google Drive and Clone Repository

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to Drive and clone repository
import os
os.chdir('/content/drive/MyDrive')

# Clone the repository (if not already cloned)
!git clone https://github.com/prabigya-pathak108/stablediffusionpytorch.git
os.chdir('stablediffusionpytorch')
```

### 2. Create Virtual Environment (Local Runtime)

```python
# Check if virtual environment already exists
import os
venv_path = '/content/drive/MyDrive/stablediffusionpytorch/venv'

if not os.path.exists(venv_path):
    print("Creating virtual environment...")
    !python -m venv {venv_path}
    print("Virtual environment created!")
else:
    print("Virtual environment already exists!")

# Activate virtual environment
import sys
sys.path.insert(0, f'{venv_path}/lib/python3.10/site-packages')
```

### 3. Install Dependencies

```python
# Install requirements in virtual environment
venv_pip = f'{venv_path}/bin/pip'
!{venv_pip} install --upgrade pip
!{venv_pip} install torch torchvision transformers numpy pillow
!{venv_pip} install -r requirements.txt
```

### 4. Download Model Data

```python
# Download and extract model data
import os
data_path = '/content/drive/MyDrive/stablediffusionpytorch/data'

if not os.path.exists(f'{data_path}/tokenizer'):
    print("Downloading model data...")
    !wget https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar
    !tar -xf data.v20221029.tar
    !rm data.v20221029.tar
    print("Model data downloaded and extracted!")
else:
    print("Model data already exists!")
```

### 5. Setup GPU Runtime

```python
# Check GPU availability
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## Usage in Colab

### Basic Text-to-Image Generation

```python
import sys
sys.path.append('/content/drive/MyDrive/stablediffusionpytorch')

from stable_diffusion.pipeline import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

# Initialize pipeline
pipeline = StableDiffusionPipeline(device=device)

# Generate image
prompt = "a beautiful landscape with mountains and lake, digital art"
images = pipeline.generate(
    prompt=prompt,
    sampler="k_lms",
    cfg_scale=7.5,
    n_inference_steps=20,  # Reduced for faster generation in Colab
    width=512,
    height=512
)

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(images[0])
plt.axis('off')
plt.title(f'Generated: "{prompt}"')
plt.show()

# Save to Drive
images[0].save('/content/drive/MyDrive/stablediffusionpytorch/output.png')
```

### Image-to-Image Generation

```python
# Load input image (upload or from Drive)
from google.colab import files
uploaded = files.upload()

# Or load from Drive
# input_image = Image.open('/content/drive/MyDrive/your_image.jpg')

input_image = Image.open(list(uploaded.keys())[0])
input_image = input_image.resize((512, 512))

# Generate image-to-image
new_images = pipeline.img2img(
    prompt="same image but in anime style",
    image=input_image,
    strength=0.7,
    sampler="k_euler",
    cfg_scale=7.5,
    n_inference_steps=20
)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(input_image)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(new_images[0])
axes[1].set_title('Generated')
axes[1].axis('off')
plt.show()
```

## Tips for Colab Usage

### Memory Management
```python
# Clear GPU memory when needed
import gc
import torch

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Use after generation
clear_memory()
```

### Batch Processing
```python
# Generate multiple images efficiently
prompts = [
    "a cat wearing sunglasses",
    "a futuristic city at sunset",
    "a magical forest with glowing trees"
]

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
    
    images = pipeline.generate(
        prompt=prompt,
        sampler="k_lms",
        n_inference_steps=15
    )
    
    # Save to Drive
    images[0].save(f'/content/drive/MyDrive/stablediffusionpytorch/output_{i+1}.png')
    
    # Clear memory between generations
    clear_memory()
    
print("All images generated!")
```

## Notes

- **Runtime**: Use GPU runtime for faster generation
- **Memory**: Reduce `n_inference_steps` (15-25) for lower memory usage
- **Persistence**: Files saved to Drive will persist between sessions
- **Virtual Environment**: Created once and reused across sessions
- **Model Data**: Downloaded once and cached in Drive

## Troubleshooting

- If memory issues occur, restart runtime and reduce image dimensions
- For slow generation, ensure GPU runtime is enabled
- If import errors occur, check virtual environment activation
- For Drive mounting issues, clear browser cache and re-authenticate