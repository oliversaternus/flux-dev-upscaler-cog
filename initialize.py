import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

CACHE_DIR = "./cache"

print("Initializing upscale controlnet...")
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", 
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR
)

print("Initializing FLUX-dev...")
self.pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR
)

print("Initialized models")
