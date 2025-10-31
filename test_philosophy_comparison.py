#!/usr/bin/env python3
"""Compare Brie v2 vs baseline on philosophy & creative brainstorming prompts"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
import sys
import argparse
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description="Compare Brie vs baseline on philosophy prompts")
parser.add_argument(
    "--model-size",
    type=str,
    default="3b",
    choices=["0.5b", "3b", "7b", "qwen3-0.6b", "llama-3b"],
    help="Model size to use (default: 3b)"
)
args = parser.parse_args()

# Map size to model paths
BASELINE_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}
BRIE_MAP = {
    "0.5b": "runs/brie-v2-0.5b/checkpoint-290",
    "3b": "runs/brie-v2-3b",
    "7b": "runs/brie-v2-7b",
    "qwen3-0.6b": "runs/brie-v3-qwen3-0.6b",
    "llama-3b": "runs/brie-llama-3b",
}

baseline_id = BASELINE_MAP[args.model_size]
brie_path = BRIE_MAP[args.model_size]

print(f"Loading baseline {baseline_id}...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    baseline_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_id)

print(f"Loading Brie v2 ({args.model_size.upper()})...")
brie_model = AutoPeftModelForCausalLM.from_pretrained(
    brie_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
brie_tokenizer = AutoTokenizer.from_pretrained(brie_path)

def generate_response(model, tokenizer, prompt: str, system_prompt: str = "You are a helpful AI assistant.", model_name: str = "") -> tuple[str, float]:
    """Generate response and return (text, latency_seconds)"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Disable thinking for Qwen3 models to ensure fair comparison
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True
    }
    if "Qwen3" in model_name or "qwen3" in model_name:
        template_kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(
        messages,
        **template_kwargs
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

# Test prompts from Brie's training domains
PHILOSOPHY_PROMPTS = [
    "Explain Heidegger's concept of Dasein in simple terms.",
    "What is the difference between Continental and Analytic philosophy?",
    "How does Derrida's concept of différance challenge traditional metaphysics?",
    "Explain Sartre's 'existence precedes essence' with an example.",
    "What is phenomenology and why is it important?",
]

BRAINSTORMING_PROMPTS = [
    "Give me 5 creative article ideas about the philosophy of artificial intelligence.",
    "Brainstorm 5 ways to explain quantum mechanics to a 10-year-old.",
    "Suggest 5 thought experiments about personal identity and consciousness.",
    "Generate 5 creative angles for writing about the ethics of AI alignment.",
    "Propose 5 unconventional perspectives on the meaning of 'understanding' in AI systems.",
]

CREATIVE_PROMPTS = [
    "Write a short meditation on the experience of contemplating emptiness.",
    "Describe the feeling of being present in a moment of silence.",
    "How would you explain 'being' to someone who has never thought about philosophy?",
]

ALL_PROMPTS = PHILOSOPHY_PROMPTS + BRAINSTORMING_PROMPTS + CREATIVE_PROMPTS

print(f"\nRunning {len(ALL_PROMPTS)} test prompts...\n")
print("=" * 80)

results = []

for i, prompt in enumerate(ALL_PROMPTS, 1):
    print(f"\n[{i}/{len(ALL_PROMPTS)}] Prompt: {prompt}\n")

    # Generate baseline response
    baseline_output, baseline_latency = generate_response(baseline_model, baseline_tokenizer, prompt, model_name=baseline_id)
    print(f"BASELINE ({baseline_latency:.2f}s):\n{baseline_output}\n")
    print("-" * 80)

    # Generate Brie v2 response
    brie_output, brie_latency = generate_response(brie_model, brie_tokenizer, prompt, model_name=brie_path)
    print(f"BRIE V2 ({brie_latency:.2f}s):\n{brie_output}\n")
    print("=" * 80)

    # Save result
    results.append({
        "prompt_num": i,
        "prompt": prompt,
        "baseline_output": baseline_output,
        "baseline_latency_s": round(baseline_latency, 3),
        "brie_output": brie_output,
        "brie_latency_s": round(brie_latency, 3),
    })

# Save to JSONL with model size and timestamp
run_id = f"{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_file = f"exports/philosophy_comparison_{run_id}.jsonl"
with open(output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"\n\nResults saved to {output_file}")
print("\nSummary:")
print(f"  Model size: {args.model_size.upper()}")
print(f"  Total prompts: {len(results)}")
avg_baseline_latency = sum(r["baseline_latency_s"] for r in results) / len(results)
avg_brie_latency = sum(r["brie_latency_s"] for r in results) / len(results)
print(f"  Avg baseline latency: {avg_baseline_latency:.2f}s")
print(f"  Avg Brie latency: {avg_brie_latency:.2f}s")
