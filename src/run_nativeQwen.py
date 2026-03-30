import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from diffusers.utils import load_image
import sys
from diffusers import QwenImageEditPipeline
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from example_data.param_prompts import prompts

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
print(seeds)

guidance_scale = 3.5

# Editing configuration
idx = 3
start_edit_turn = 0
final_edit_turn = 10

# previous_turn_image_as_ref: the output from the last editing turn (used as input for the next turn)
ref_image_path = f"output/img{idx}_Qwen-native_{start_edit_turn}.png" if start_edit_turn > 0 else f"../example_data/img_{idx}.png"
previous_turn_image_as_ref = load_image(ref_image_path).convert("RGB").resize((1024, 1024))

for edit_turn in range(start_edit_turn + 1, final_edit_turn + 1):
    current_prompt = prompts[min(idx, len(prompts)) - 1][edit_turn - 1]
    print(f"\n===== Executing Turn {edit_turn}: '{current_prompt}' =====")

    edited_image = pipe(
        num_inference_steps=28,
        prompt=current_prompt,
        image=previous_turn_image_as_ref,
        height=1024,
        width=1024,
        generator=torch.Generator(device="cuda").manual_seed(seeds[edit_turn - 1]),
        guidance_scale=guidance_scale,
    ).images[0]

    output_filename = os.path.join(output_dir, f"img{idx}_Qwen-native_{edit_turn}.png")
    edited_image.save(output_filename)
    print(f"Image saved to '{output_filename}'")

    previous_turn_image_as_ref = edited_image
