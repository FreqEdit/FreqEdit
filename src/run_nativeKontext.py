import torch
from diffusers.utils import load_image
import os
import sys
from diffusers import FluxKontextPipeline
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from example_data.param_prompts import prompts

device = "cuda:0"

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to(device)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
print(seeds)

# Editing configuration
idx = 3
start_edit_turn = 0
final_edit_turn = 10

# previous_turn_image_as_ref: the output from the last editing turn (used as input for the next turn)
ref_image_path = f"output/img{idx}_Kontext-native_{start_edit_turn}.png" if start_edit_turn > 0 else f"../example_data/img_{idx}.png"
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
        generator=torch.Generator(device=device).manual_seed(seeds[edit_turn - 1]),
    ).images[0]

    output_filename = os.path.join(output_dir, f"img{idx}_Kontext-native_{edit_turn}.png")
    edited_image.save(output_filename)
    print(f"Image saved to '{output_filename}'")

    previous_turn_image_as_ref = edited_image
