#!/usr/bin/env python3
"""LLM-as-Judge: Blind evaluation of Brie v2 vs Baseline using Claude as judge"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
import anthropic
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Compare Brie vs baseline using Claude as judge")
parser.add_argument(
    "--model-size",
    type=str,
    default="3b",
    choices=["0.5b", "3b", "7b"],
    help="Model size to use (default: 3b)"
)
args = parser.parse_args()

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "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"

# Map size to model paths
BASELINE_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}
BRIE_MAP = {
    "0.5b": "runs/brie-v2-0.5b",
    "3b": "runs/brie-v2-3b",
    "7b": "runs/brie-v2-7b",
}

baseline_id = BASELINE_MAP[args.model_size]
brie_path = BRIE_MAP[args.model_size]

print(f"Loading baseline {baseline_id}...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    baseline_id,
    device_map="mps",
    torch_dtype=torch.float16,
)
baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_id)

print(f"Loading Brie v2 ({args.model_size.upper()})...")
brie_model = AutoPeftModelForCausalLM.from_pretrained(
    brie_path,
    device_map="mps",
    torch_dtype=torch.float16,
)
brie_tokenizer = AutoTokenizer.from_pretrained(brie_path)

print("Initializing Claude API client...")
claude_client = anthropic.Anthropic()

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate response from a local model"""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def judge_with_claude(prompt: str, response_a: str, response_b: str) -> dict:
    """Have Claude judge compare two responses blindly"""

    # Randomly assign A/B to avoid position bias
    order = random.choice(["AB", "BA"])
    if order == "AB":
        first, second = response_a, response_b
        first_label, second_label = "Response A", "Response B"
    else:
        first, second = response_b, response_a
        first_label, second_label = "Response B", "Response A"

    judge_prompt = f"""You are an expert evaluator of creative writing and philosophical prose. Compare these two responses to the same prompt.

Original Prompt: "{prompt}"

{first_label}:
{first}

{second_label}:
{second}

Evaluate both responses on these criteria (rate 1-5 for each, where 5 is excellent and 1 is poor):
1. Creativity & Originality
2. Coherence & Structure
3. Depth & Insight
4. Engagement & Interest
5. Writing Quality

Provide your evaluation in this EXACT format:
Response A - Creativity: X, Coherence: X, Depth: X, Engagement: X, Quality: X
Response B - Creativity: X, Coherence: X, Depth: X, Engagement: X, Quality: X
Winner: [A or B or Tie]
Reasoning: [2-3 sentences explaining your choice]

Be critical and honest. Consider whether responses are truly insightful or just verbose."""

    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {"role": "user", "content": judge_prompt}
            ]
        )

        judgment = message.content[0].text

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        judgment = f"ERROR: {str(e)}"

    return {
        "judgment": judgment.strip(),
        "order": order,
    }

# Creative writing prompts (focused on Brie's training domain)
CREATIVE_PROMPTS = [
    "Write a short philosophical meditation on the nature of time.",
    "Describe a moment of sudden understanding or insight.",
    "Write about the experience of being alone in nature.",
    "Create a thought experiment about consciousness and identity.",
    "Describe the feeling of reading a book that fundamentally changes your perspective on life.",
]

results = []

print("\n" + "="*80)
print("GENERATING CREATIVE WRITING SAMPLES")
print("="*80)

for i, prompt in enumerate(CREATIVE_PROMPTS, 1):
    print(f"\n[{i}/{len(CREATIVE_PROMPTS)}] Prompt: {prompt}\n")

    # Generate from both models
    print("Generating baseline response...")
    baseline_response = generate_response(baseline_model, baseline_tokenizer, prompt)

    print("Generating Brie v2 response...")
    brie_response = generate_response(brie_model, brie_tokenizer, prompt)

    print("Getting Claude's judgment...")
    judgment_data = judge_with_claude(prompt, baseline_response, brie_response)

    # Determine which model won
    judgment_text = judgment_data["judgment"].lower()
    order = judgment_data["order"]

    # Parse winner
    winner = "unknown"
    if "winner: a" in judgment_text or "winner:a" in judgment_text:
        winner_label = "A"
    elif "winner: b" in judgment_text or "winner:b" in judgment_text:
        winner_label = "B"
    elif "tie" in judgment_text:
        winner_label = "Tie"
    else:
        winner_label = "unknown"

    # Map back to actual model based on order
    if winner_label == "A":
        winner = "baseline" if order == "AB" else "brie"
    elif winner_label == "B":
        winner = "brie" if order == "AB" else "baseline"
    elif winner_label == "Tie":
        winner = "tie"

    result = {
        "prompt_num": i,
        "prompt": prompt,
        "baseline_response": baseline_response,
        "baseline_length": len(baseline_response),
        "brie_response": brie_response,
        "brie_length": len(brie_response),
        "judgment": judgment_data["judgment"],
        "presentation_order": order,
        "winner": winner,
    }

    results.append(result)

    print(f"\n{'─'*80}")
    print(f"BASELINE ({len(baseline_response)} chars):")
    print(baseline_response)
    print(f"\n{'─'*80}")
    print(f"BRIE V2 ({len(brie_response)} chars):")
    print(brie_response)
    print(f"\n{'─'*80}")
    print("CLAUDE'S JUDGMENT:")
    print(judgment_data["judgment"])
    print(f"\nWINNER: {winner.upper()}")
    print("="*80)

# Save results with model size in filename
from datetime import datetime
run_id = f"{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_file = f"exports/claude_judge_{run_id}.jsonl"
with open(output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

# Calculate statistics
baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
brie_wins = sum(1 for r in results if r["winner"] == "brie")
ties = sum(1 for r in results if r["winner"] == "tie")
unknown = sum(1 for r in results if r["winner"] == "unknown")

print(f"\n\n{'='*80}")
print(f"FINAL RESULTS - CLAUDE AS JUDGE ({args.model_size.upper()})")
print("="*80)
print(f"Model size: {args.model_size.upper()}")
print(f"Total prompts: {len(results)}")
print(f"Baseline wins: {baseline_wins} ({baseline_wins/len(results)*100:.1f}%)")
print(f"Brie v2 wins: {brie_wins} ({brie_wins/len(results)*100:.1f}%)")
print(f"Ties: {ties} ({ties/len(results)*100:.1f}%)")
print(f"Unknown: {unknown}")
print(f"\nResults saved to {output_file}")
print("\n" + "="*80)
