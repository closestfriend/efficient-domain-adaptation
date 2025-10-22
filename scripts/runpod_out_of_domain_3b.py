#!/usr/bin/env python3
"""Test Brie 3B vs baseline on OUT-OF-DOMAIN tasks on RunPod (CUDA)"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
from datetime import datetime

# CUDA device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model paths for 3B
baseline_id = "Qwen/Qwen2.5-3B-Instruct"
brie_path = "runs/brie-v2-3b"

print(f"Loading baseline {baseline_id}...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    baseline_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_id)

print(f"Loading Brie v2 3B from {brie_path}...")
brie_model = AutoPeftModelForCausalLM.from_pretrained(
    brie_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
brie_tokenizer = AutoTokenizer.from_pretrained(brie_path)

def generate_response(model, tokenizer, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> tuple[str, float]:
    """Generate response and return (text, latency_seconds)"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start_time

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip(), latency

# Out-of-domain test prompts (NOT in training data)
CODING_PROMPTS = [
    "Write a Python function to find the longest palindromic substring in a string.",
    "Explain the difference between == and === in JavaScript.",
    "How do you reverse a linked list in-place?",
]

MATH_PROMPTS = [
    "Explain the Pythagorean theorem with an example.",
    "What is the difference between mean, median, and mode?",
    "How do you calculate compound interest?",
]

PRACTICAL_PROMPTS = [
    "Give me a simple recipe for chocolate chip cookies.",
    "How do I change a flat tire?",
    "What are 5 tips for better sleep hygiene?",
]

CREATIVE_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Describe a futuristic city in 3 sentences.",
    "Create a haiku about coffee.",
]

FACTUAL_PROMPTS = [
    "What are the main differences between capitalism and socialism?",
    "Explain how photosynthesis works.",
    "What caused World War I?",
]

ALL_PROMPTS = CODING_PROMPTS + MATH_PROMPTS + PRACTICAL_PROMPTS + CREATIVE_PROMPTS + FACTUAL_PROMPTS

print(f"\nRunning {len(ALL_PROMPTS)} OUT-OF-DOMAIN test prompts on Brie 3B...\n")
print("=" * 80)

results = []

for i, prompt in enumerate(ALL_PROMPTS, 1):
    print(f"\n[{i}/{len(ALL_PROMPTS)}] Prompt: {prompt}\n")

    # Generate baseline response
    baseline_output, baseline_latency = generate_response(baseline_model, baseline_tokenizer, prompt)
    print(f"BASELINE ({baseline_latency:.2f}s):\n{baseline_output[:200]}...\n")
    print("-" * 80)

    # Generate Brie v2 response
    brie_output, brie_latency = generate_response(brie_model, brie_tokenizer, prompt)
    print(f"BRIE V2 3B ({brie_latency:.2f}s):\n{brie_output[:200]}...\n")
    print("=" * 80)

    # Determine category
    if i <= 3:
        category = "coding"
    elif i <= 6:
        category = "math"
    elif i <= 9:
        category = "practical"
    elif i <= 12:
        category = "creative"
    else:
        category = "factual"

    # Save result
    results.append({
        "prompt_num": i,
        "prompt": prompt,
        "category": category,
        "baseline_output": baseline_output,
        "baseline_latency_s": round(baseline_latency, 3),
        "brie_output": brie_output,
        "brie_latency_s": round(brie_latency, 3),
    })

# Save to JSONL with timestamp
run_id = f"3b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_file = f"exports/out_of_domain_{run_id}.jsonl"
with open(output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"\n\nResults saved to {output_file}")
print("\nSummary:")
print(f"  Model size: 3B")
print(f"  Total prompts: {len(results)}")
avg_baseline_latency = sum(r["baseline_latency_s"] for r in results) / len(results)
avg_brie_latency = sum(r["brie_latency_s"] for r in results) / len(results)
avg_baseline_length = sum(len(r["baseline_output"]) for r in results) / len(results)
avg_brie_length = sum(len(r["brie_output"]) for r in results) / len(results)

print(f"  Avg baseline latency: {avg_baseline_latency:.2f}s")
print(f"  Avg Brie latency: {avg_brie_latency:.2f}s")
print(f"  Avg baseline length: {avg_baseline_length:.0f} chars")
print(f"  Avg Brie length: {avg_brie_length:.0f} chars")
print(f"  Length difference: {((avg_brie_length / avg_baseline_length) - 1) * 100:+.1f}%")

print(f"\n\nNext steps:")
print(f"1. Download this file locally: {output_file}")
print(f"2. Run judge: python judge_existing_outputs.py {output_file} --model sonnet")
print(f"3. Compare results with 0.5B out-of-domain performance")
