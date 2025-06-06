import os
import time
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image

CACHE_DIR = "./cache"

class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory"""

        start = time.time()
        
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
        
        self.pipe.to("cuda")

        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        input_image: Path = Input(
            description="Input image to upscale",
            default=None,
        ),
        upscale_factor: int = Input(
            description="Upscale factor", ge=1, le=4, default=2
        ),
        controlnet_conditioning_scale: float = Input(
            description="Higher values will closer to original image, but overall lower quality", ge=0.1, le=1.5, default=0.6
        ),
        num_inference_steps: int = Input(
            description="Higher values increase computation time", ge=1, le=50, default=28
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        predict_start = time.time()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(input_image)
        w, h = image.size
        control_image = image.resize((w * upscale_factor, h * upscale_factor))
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        image = self.pipe(
            prompt="", 
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps, 
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0],
            generator=generator,
        ).images[0]

        output_path = f"/tmp/out-0.png"
        image.save(output_path)

        print(f"prediction took: {time.time() - predict_start:.2f}s")
        return output_path
