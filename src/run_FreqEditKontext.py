import torch
from diffusers.utils import load_image
import os
import sys
from pipelines.FreqEditKontext_pipeline import FreqEditKontextPipeline
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from example_data.param_prompts import prompts

device = "cuda:0"

pipe = FreqEditKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to(device)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
print(seeds)

# FreqEdit parameters
Inject_time_scale = 0.3
base_alpha = 1.6
gamma = 2.0
levels = 2
wavelet_mode = "db4"
use_adaptive_weight = True
backward_steps = 4

# Quality Guidance parameters (Kontext only)
QG_time_scale = 0.7
QG_lambda = 0.3

guidance_scale = 3.5

# Editing configuration
idx = 3
start_edit_turn = 0
final_edit_turn = 10

# source_image: the original image, kept unchanged across all turns (used for Quality Guidance)
source_image_path = f"../example_data/img_{idx}.png"
source_image = load_image(source_image_path).convert("RGB").resize((1024, 1024))

# previous_turn_image_as_ref: the output from the last editing turn (used as input for the next turn)
ref_image_path = f"output/img{idx}_Kontext-multi-turns_{start_edit_turn}.png" if start_edit_turn > 0 else f"../example_data/img_{idx}.png"
previous_turn_image_as_ref = load_image(ref_image_path).convert("RGB").resize((1024, 1024))

for edit_turn in range(start_edit_turn + 1, final_edit_turn + 1):
    current_prompt = prompts[min(idx, len(prompts)) - 1][edit_turn - 1]
    print(f"\n===== Executing Turn {edit_turn}: '{current_prompt}' =====")

    edited_image = pipe(
        num_inference_steps=28,
        prompt=current_prompt,
        image=previous_turn_image_as_ref,
        source_image=source_image,
        height=1024,
        width=1024,
        generator=torch.Generator(device=device).manual_seed(seeds[edit_turn - 1]),
        QG_lambda=QG_lambda,
        Inject_time_scale=Inject_time_scale,
        QG_time_scale=QG_time_scale,
        base_alpha=base_alpha,
        levels=levels,
        prompt_preserve="A high quality picture.",
        guidance_scale=guidance_scale,
        wavelet_mode=wavelet_mode,
        backward_steps=backward_steps,
        gamma=gamma,
        use_adaptive_weight=use_adaptive_weight,
    ).images[0]

    output_filename = os.path.join(output_dir, f"img{idx}_Kontext-multi-turns_{edit_turn}.png")
    edited_image.save(output_filename)
    print(f"Image saved to '{output_filename}'")

    previous_turn_image_as_ref = edited_image
