#!/usr/bin/env python3
"""Judge existing comparison outputs using Claude as judge"""
import json
import random
import anthropic
import os
import sys
from datetime import datetime

# Load environment variables from .env.local if it exists
if os.path.exists('.env.local'):
    with open('.env.local', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key.strip()] = value

# Parse arguments
if len(sys.argv) < 2:
    print("Usage: python judge_existing_outputs.py <input_jsonl_file> [--model sonnet|sonnet37|opus|haiku]")
    sys.exit(1)

input_file = sys.argv[1]
judge_model = "claude-3-7-sonnet-20250219"  # Default to 3.7 Sonnet

if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    model_choice = sys.argv[idx + 1]
    if model_choice == "opus":
        judge_model = "claude-opus-4-20250514"
    elif model_choice == "haiku":
        judge_model = "claude-3-5-haiku-20241022"
    elif model_choice == "sonnet":
        judge_model = "claude-3-5-sonnet-20241022"
    elif model_choice == "sonnet37":
        judge_model = "claude-3-7-sonnet-20250219"

print(f"Loading responses from {input_file}")
print(f"Using judge model: {judge_model}\n")

# Load existing responses
with open(input_file, 'r') as f:
    comparisons = [json.loads(line) for line in f]

print(f"Found {len(comparisons)} comparisons to judge\n")

# Set API key (get from environment or use hardcoded)
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    # Check if key is in test_llm_as_judge_claude.py
    try:
        with open("test_llm_as_judge_claude.py", "r") as f:
            for line in f:
                if "ANTHROPIC_API_KEY" in line and "sk-ant" in line:
                    api_key = line.split('"')[1]
                    break
    except:
        pass

if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found!")
    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    sys.exit(1)

os.environ["ANTHROPIC_API_KEY"] = api_key

# Initialize Claude
client = anthropic.Anthropic()

def judge_with_claude(prompt: str, baseline_response: str, brie_response: str) -> dict:
    """Have Claude judge compare two responses blindly"""

    # Randomly assign A/B to avoid position bias
    order = random.choice(["AB", "BA"])
    if order == "AB":
        first, second = baseline_response, brie_response
        first_label, second_label = "Response A", "Response B"
    else:
        first, second = brie_response, baseline_response
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
        message = client.messages.create(
            model=judge_model,
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

# Process each comparison
results = []
for i, comp in enumerate(comparisons, 1):
    prompt = comp["prompt"]

    # Handle different field names in input files
    if "baseline_output" in comp:
        baseline_output = comp["baseline_output"]
        brie_output = comp["brie_output"]
    elif "baseline_response" in comp:
        baseline_output = comp["baseline_response"]
        brie_output = comp["brie_response"]
    else:
        print(f"Error: Unknown field names in input file")
        sys.exit(1)

    print(f"[{i}/{len(comparisons)}] Judging: {prompt[:60]}...")

    judgment_data = judge_with_claude(prompt, baseline_output, brie_output)

    # Determine winner
    judgment_text = judgment_data["judgment"].lower()
    order = judgment_data["order"]

    # Parse winner from judgment
    winner = "unknown"
    if "winner: a" in judgment_text or "winner:a" in judgment_text:
        winner_label = "A"
    elif "winner: b" in judgment_text or "winner:b" in judgment_text:
        winner_label = "B"
    elif "tie" in judgment_text:
        winner_label = "Tie"
    else:
        winner_label = "unknown"

    # Map labels to models (labels are always fixed regardless of presentation order)
    # Response A is ALWAYS baseline, Response B is ALWAYS brie
    if winner_label == "A":
        winner = "baseline"
    elif winner_label == "B":
        winner = "brie"
    elif winner_label == "Tie":
        winner = "tie"

    result = {
        **comp,  # Include all original fields
        "judgment": judgment_data["judgment"],
        "presentation_order": order,
        "winner": winner,
    }

    results.append(result)
    print(f"  Winner: {winner.upper()}\n")

# Save results
base_name = input_file.replace("exports/", "").replace(".jsonl", "")
output_file = f"exports/{base_name}_judged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

with open(output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

# Calculate statistics
baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
brie_wins = sum(1 for r in results if r["winner"] == "brie")
ties = sum(1 for r in results if r["winner"] == "tie")
unknown = sum(1 for r in results if r["winner"] == "unknown")

print("\n" + "="*80)
print(f"FINAL RESULTS - {judge_model}")
print("="*80)
print(f"Total prompts: {len(results)}")
print(f"Baseline wins: {baseline_wins} ({baseline_wins/len(results)*100:.1f}%)")
print(f"Brie v2 wins: {brie_wins} ({brie_wins/len(results)*100:.1f}%)")
print(f"Ties: {ties} ({ties/len(results)*100:.1f}%)")
print(f"Unknown: {unknown}")
print(f"\nResults saved to {output_file}")
print("="*80)
