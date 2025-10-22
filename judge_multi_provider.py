#!/usr/bin/env python3
"""Judge existing comparison outputs using OpenAI or Google Gemini as judge"""
import json
import random
import os
import sys
from datetime import datetime
import argparse

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
parser = argparse.ArgumentParser(description="Judge existing outputs with OpenAI or Gemini")
parser.add_argument("input_file", help="Input JSONL file with model outputs")
parser.add_argument(
    "--judge",
    type=str,
    required=True,
    choices=["openai", "gemini", "both"],
    help="Which judge to use: openai, gemini, or both"
)
parser.add_argument(
    "--model",
    type=str,
    help="Specific model to use (default: gpt-4o for OpenAI, gemini-2.0-flash-exp for Gemini)"
)
args = parser.parse_args()

# Default models
OPENAI_DEFAULT = "gpt-4o"
GEMINI_DEFAULT = "gemini-2.5-flash-lite"

# Determine which judges to use
judges_to_run = []
if args.judge in ["openai", "both"]:
    openai_model = args.model if args.judge == "openai" and args.model else OPENAI_DEFAULT
    judges_to_run.append(("openai", openai_model))
if args.judge in ["gemini", "both"]:
    gemini_model = args.model if args.judge == "gemini" and args.model else GEMINI_DEFAULT
    judges_to_run.append(("gemini", gemini_model))

print(f"Loading responses from {args.input_file}")
print(f"Judge configuration: {', '.join([f'{j[0]}={j[1]}' for j in judges_to_run])}\n")

# Load existing responses
with open(args.input_file, 'r') as f:
    comparisons = [json.loads(line) for line in f]

print(f"Found {len(comparisons)} comparisons to judge\n")

# Initialize API clients based on which judges we're using
clients = {}

if any(j[0] == "openai" for j in judges_to_run):
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        clients["openai"] = openai.OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")
    except ImportError:
        print("ERROR: openai package not installed!")
        print("Install with: pip install openai")
        sys.exit(1)

if any(j[0] == "gemini" for j in judges_to_run):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found!")
            print("Set it with: export GEMINI_API_KEY='your-key-here'")
            sys.exit(1)
        genai.configure(api_key=api_key)
        clients["gemini"] = genai
        print("✓ Gemini client initialized")
    except ImportError:
        print("ERROR: google-generativeai package not installed!")
        print("Install with: pip install google-generativeai")
        sys.exit(1)

print()

def create_judge_prompt(prompt: str, first: str, second: str, first_label: str, second_label: str) -> str:
    """Create the judge prompt (same format as Claude judge)"""
    return f"""You are an expert evaluator of creative writing and philosophical prose. Compare these two responses to the same prompt.

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

def judge_with_openai(prompt: str, baseline_response: str, brie_response: str, model: str) -> dict:
    """Have OpenAI GPT judge compare two responses blindly"""

    # Randomly assign A/B to avoid position bias
    order = random.choice(["AB", "BA"])
    if order == "AB":
        first, second = baseline_response, brie_response
        first_label, second_label = "Response A", "Response B"
    else:
        first, second = brie_response, baseline_response
        first_label, second_label = "Response B", "Response A"

    judge_prompt = create_judge_prompt(prompt, first, second, first_label, second_label)

    try:
        response = clients["openai"].chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        judgment = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        judgment = f"ERROR: {str(e)}"

    return {
        "judgment": judgment.strip(),
        "order": order,
    }

def judge_with_gemini(prompt: str, baseline_response: str, brie_response: str, model: str) -> dict:
    """Have Google Gemini judge compare two responses blindly"""

    # Randomly assign A/B to avoid position bias
    order = random.choice(["AB", "BA"])
    if order == "AB":
        first, second = baseline_response, brie_response
        first_label, second_label = "Response A", "Response B"
    else:
        first, second = brie_response, baseline_response
        first_label, second_label = "Response B", "Response A"

    judge_prompt = create_judge_prompt(prompt, first, second, first_label, second_label)

    try:
        gemini_model = clients["gemini"].GenerativeModel(model)
        response = gemini_model.generate_content(
            judge_prompt,
            generation_config=clients["gemini"].types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )
        judgment = response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        judgment = f"ERROR: {str(e)}"

    return {
        "judgment": judgment.strip(),
        "order": order,
    }

def parse_winner(judgment_text: str) -> str:
    """Parse winner from judgment text"""
    judgment_lower = judgment_text.lower()

    if "winner: a" in judgment_lower or "winner:a" in judgment_lower:
        return "baseline"
    elif "winner: b" in judgment_lower or "winner:b" in judgment_lower:
        return "brie"
    elif "tie" in judgment_lower:
        return "tie"
    else:
        return "unknown"

# Process each comparison with each judge
for judge_type, judge_model in judges_to_run:
    print(f"\n{'='*80}")
    print(f"RUNNING JUDGE: {judge_type.upper()} ({judge_model})")
    print(f"{'='*80}\n")

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

        # Call appropriate judge
        if judge_type == "openai":
            judgment_data = judge_with_openai(prompt, baseline_output, brie_output, judge_model)
        elif judge_type == "gemini":
            judgment_data = judge_with_gemini(prompt, baseline_output, brie_output, judge_model)

        # Parse winner
        winner = parse_winner(judgment_data["judgment"])

        result = {
            **comp,  # Include all original fields
            "judgment": judgment_data["judgment"],
            "presentation_order": judgment_data["order"],
            "winner": winner,
            "judge_model": judge_model,
            "judge_type": judge_type,
        }

        results.append(result)
        print(f"  Winner: {winner.upper()}\n")

    # Save results for this judge
    base_name = os.path.basename(args.input_file).replace("exports/", "").replace(".jsonl", "")
    judge_name = judge_model.replace("-", "_").replace(".", "_")
    output_file = f"exports/{base_name}_judged_{judge_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Calculate statistics
    baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
    brie_wins = sum(1 for r in results if r["winner"] == "brie")
    ties = sum(1 for r in results if r["winner"] == "tie")
    unknown = sum(1 for r in results if r["winner"] == "unknown")

    print(f"\n{'='*80}")
    print(f"RESULTS - {judge_type.upper()} ({judge_model})")
    print(f"{'='*80}")
    print(f"Total prompts: {len(results)}")
    print(f"Baseline wins: {baseline_wins} ({baseline_wins/len(results)*100:.1f}%)")
    print(f"Brie wins: {brie_wins} ({brie_wins/len(results)*100:.1f}%)")
    print(f"Ties: {ties} ({ties/len(results)*100:.1f}%)")
    print(f"Unknown: {unknown}")
    print(f"\nResults saved to {output_file}")
    print(f"{'='*80}\n")

print("\n" + "="*80)
print("ALL JUDGING COMPLETE")
print("="*80)
