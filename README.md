 # Text_to_Image_Generation


## Stable Diffusion XL with LoRA Weights

This repository contains a setup for generating images using **Stable Diffusion XL** with **LoRA** (Low-Rank Adaptation) weights. We use a fine-tuned LoRA checkpoint from the [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) model and the [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model as the base.

### Installation

To run the code, you'll need to install the necessary libraries, including `diffusers`, `transformers`, `accelerate`, `mediapy`, and `peft`:

```bash
!pip install --quiet --upgrade diffusers transformers accelerate mediapy peft
```

### Model Setup

In this example, we are using the Stable Diffusion XL base model and loading fine-tuned **LoRA weights** from a model checkpoint stored on Hugging Face Hub.

```python
import mediapy as media
import random
import sys
import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
```

### Step-by-Step Explanation:

1. **Set the number of inference steps**: You can choose either 8 or 12 steps. The performance and quality of generated images may vary based on this value.
   
   ```python
   num_inference_steps = 12
   ```

2. **Base Model**: We use the base model from Stability AI, `stabilityai/stable-diffusion-xl-base-1.0`.
   
   ```python
   base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
   ```

3. **LoRA Checkpoint**: We download the LoRA checkpoint from the Hugging Face Hub. The checkpoint name is dynamically created based on the number of inference steps chosen.
   
   ```python
   repo_name = "ByteDance/Hyper-SD"
   ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-CFG-lora.safetensors"
   ```

4. **Device Configuration**: The code runs on a CUDA-enabled GPU. If you don't have access to a GPU, modify the device to `"cpu"`.
   
   ```python
   device = "cuda"
   ```

5. **Load the Pretrained Model**: We load the `stabilityai/stable-diffusion-xl-base-1.0` model from Hugging Face using the `DiffusionPipeline`. We load it with `fp16` precision to save memory and improve performance on GPUs.

   ```python
   pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
   ```

6. **Download and Load LoRA Weights**: The LoRA weights are downloaded from the repository and fused into the model. This allows us to generate images using the fine-tuned Hyper SD XL model.
   
   ```python
   pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
   pipe.fuse_lora()
   ```

7. **Set the Scheduler**: We replace the default scheduler with the TCDScheduler, which may provide better results for this specific setup.
   
   ```python
   pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
   ```

## Image Generation

After setting up the model, you can use it to generate images from prompts by calling `pipe()` with your desired parameters (e.g., prompt, guidance scale).

---
## Example Prompts
### Input: 
"Create a vibrant and detailed display of traditional Indian sweets on a decorative plate. Include popular sweets like golden jalebis, creamy rasgullas, colorful laddus, soft barfis, and syrupy gulab jamuns. The scene is festive, with bright colors, intricate patterns, and flowers, capturing the rich textures and vibrant hues of the sweets. The background should reflect a traditional Indian setting, with elements like decorative fabrics, brass utensils, and marigold garlands."
## Output of the Prompt

![download](https://github.com/user-attachments/assets/9a9458df-6bd8-45a3-b939-c52baddf3b84)

