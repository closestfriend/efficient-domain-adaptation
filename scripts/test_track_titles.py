#!/usr/bin/env python3
"""
Compare Brie (fine-tuned) vs baseline Qwen for generating ambient drone track titles.
Tests 20 prompts, each requesting 5 track titles for a post-music ambient electronic drone artist.
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# 20 prompts for track title generation
PROMPTS = [
    "Give me 5 track titles for a post-music ambient electronic drone piece exploring themes of emptiness and silence.",
    "Suggest 5 track names for an ambient drone album about forgotten architectural spaces.",
    "Create 5 titles for minimal electronic drone tracks inspired by deep ocean trenches.",
    "Propose 5 track names for a contemplative drone album about the passage of geological time.",
    "Generate 5 titles for ambient pieces focused on the sound of decay and entropy.",
    "List 5 track titles for drone compositions exploring liminal spaces and transitions.",
    "Suggest 5 names for ambient electronic tracks about light diffusion through fog.",
    "Create 5 track titles for minimalist drone works inspired by Arctic isolation.",
    "Provide 5 titles for post-ambient pieces about the silence between sounds.",
    "Generate 5 track names for electronic drone compositions exploring memory and forgetting.",
    "Suggest 5 titles for ambient works about the texture of concrete and industrial surfaces.",
    "Create 5 track names for drone pieces inspired by nocturnal urban environments.",
    "List 5 titles for minimal electronic compositions about temporal dislocation.",
    "Propose 5 track names for ambient drone works exploring emptied ritual spaces.",
    "Generate 5 titles for post-music pieces about the weight of atmosphere.",
    "Suggest 5 track names for drone compositions inspired by abandoned infrastructure.",
    "Create 5 titles for ambient electronic works about the border between sound and silence.",
    "List 5 track names for minimal drone pieces exploring subsonic frequencies.",
    "Propose 5 titles for post-ambient compositions about geological strata and deep time.",
    "Generate 5 track names for electronic drone works focused on the phenomenology of listening."
]


def generate_with_model(model, tokenizer, prompt: str, device: str = "mps") -> Dict:
    """Generate response from a model with timing."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    dt = time.time() - t0

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        'output': response.strip(),
        'latency_s': round(dt, 3),
    }


def main():
    device = "mps"  # Change to "cuda" for NVIDIA or "cpu" for CPU
    output_file = "exports/track_titles_comparison.jsonl"

    print("="*80)
    print("Track Title Generation Comparison: Baseline vs Brie")
    print("="*80)

    # Load baseline model
    print("\n[1/2] Loading baseline Qwen 2.5 0.5B Instruct...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        device_map=device,
        torch_dtype=torch.float16,
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print("Baseline model loaded.")

    # Generate baseline responses
    print("\nGenerating baseline responses...")
    baseline_results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] Baseline: ", end="", flush=True)
        result = generate_with_model(baseline_model, baseline_tokenizer, prompt, device)
        baseline_results.append(result)
        print(f"{result['latency_s']}s")

    # Free baseline model memory
    del baseline_model
    del baseline_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load fine-tuned Brie model
    print("\n[2/2] Loading fine-tuned Brie model...")
    brie_model = AutoPeftModelForCausalLM.from_pretrained(
        "runs/brie-v1-0.5b/",
        device_map=device,
        torch_dtype=torch.float16,
    )
    brie_tokenizer = AutoTokenizer.from_pretrained("runs/brie-v1-0.5b/")
    print("Brie model loaded.")

    # Generate Brie responses
    print("\nGenerating Brie responses...")
    brie_results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] Brie: ", end="", flush=True)
        result = generate_with_model(brie_model, brie_tokenizer, prompt, device)
        brie_results.append(result)
        print(f"{result['latency_s']}s")

    # Save results to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_file}...")
    with output_path.open('w', encoding='utf-8') as f:
        for i, prompt in enumerate(PROMPTS):
            record = {
                'prompt_num': i + 1,
                'prompt': prompt,
                'baseline_output': baseline_results[i]['output'],
                'baseline_latency_s': baseline_results[i]['latency_s'],
                'brie_output': brie_results[i]['output'],
                'brie_latency_s': brie_results[i]['latency_s'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    baseline_avg = sum(r['latency_s'] for r in baseline_results) / len(baseline_results)
    brie_avg = sum(r['latency_s'] for r in brie_results) / len(brie_results)

    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Baseline avg latency: {baseline_avg:.3f}s")
    print(f"Brie avg latency: {brie_avg:.3f}s")
    print(f"\nResults saved to: {output_file}")

    # Print first comparison example
    print("\n" + "="*80)
    print("EXAMPLE COMPARISON (Prompt 1)")
    print("="*80)
    print(f"Prompt: {PROMPTS[0]}")
    print("\n" + "-"*80)
    print("[BASELINE]")
    print(baseline_results[0]['output'])
    print(f"(latency: {baseline_results[0]['latency_s']}s)")
    print("\n" + "-"*80)
    print("[BRIE]")
    print(brie_results[0]['output'])
    print(f"(latency: {brie_results[0]['latency_s']}s)")
    print("="*80)

    print(f"\nTo view all comparisons, check: {output_file}")


if __name__ == '__main__':
    main()
