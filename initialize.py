import os
import torch
from dotenv import load_dotenv
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image

# Load environment variables from .env file
load_dotenv()

CACHE_DIR = "./cache"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

print("Initializing upscale controlnet...")
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", 
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN
)

print("Initializing FLUX-dev...")
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN
)

print("Initialized models")
