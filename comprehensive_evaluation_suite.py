#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Brie v2
Runs multiple test configurations with Claude Sonnet + Opus as judges
"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
import anthropic
import os
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import time

# Parse arguments
parser = argparse.ArgumentParser(description="Comprehensive evaluation suite for Brie v2")
parser.add_argument(
    "--model-size",
    type=str,
    default="3b",
    choices=["0.5b", "3b", "7b"],
    help="Model size to use (default: 3b)"
)
args = parser.parse_args()

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

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "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"

print("="*80)
print(f"COMPREHENSIVE BRIE V2 EVALUATION SUITE ({args.model_size.upper()})")
print("="*80)

print(f"\nLoading baseline {baseline_id}...")
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

# =============================================================================
# PROMPT SETS
# =============================================================================

ORIGINAL_PROMPTS = [
    "Write a short philosophical meditation on the nature of time.",
    "Describe a moment of sudden understanding or insight.",
    "Write about the experience of being alone in nature.",
    "Create a thought experiment about consciousness and identity.",
    "Describe the feeling of reading a book that fundamentally changes your perspective on life.",
]

PHILOSOPHY_PROMPTS = [
    "Explain the concept of 'being-in-the-world' from phenomenology.",
    "What is the relationship between language and reality in continental philosophy?",
    "Describe the paradox of free will in a deterministic universe.",
    "How does existentialism address the question of meaning in life?",
    "What is the difference between ontology and epistemology?",
]

BRAINSTORMING_PROMPTS = [
    "Generate 5 creative approaches to teaching philosophy to beginners.",
    "Suggest 5 innovative ways to explore the concept of identity in art.",
    "Propose 5 unconventional perspectives on the ethics of technology.",
    "Brainstorm 5 thought-provoking questions about human consciousness.",
    "Create 5 creative angles for writing about the nature of reality.",
]

CONTEMPLATIVE_PROMPTS = [
    "Write a meditation on the experience of uncertainty.",
    "Describe the feeling of recognizing a pattern you've never seen before.",
    "Reflect on the relationship between silence and understanding.",
    "Explore the sensation of being between two states of mind.",
    "Contemplate the boundary between self and other.",
]

EXPANDED_CREATIVE_PROMPTS = [
    "Describe the moment when a familiar place suddenly feels strange.",
    "Write about the experience of learning something that contradicts everything you believed.",
    "Explore the feeling of being misunderstood despite clear communication.",
    "Reflect on the paradox of seeking answers while knowing questions are more valuable.",
    "Describe the experience of multiple conflicting truths existing simultaneously.",
]

# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.75,
    max_tokens: int = 512
) -> Tuple[str, float]:
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

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start_time

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip(), latency

def judge_with_claude(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str = "claude-3-5-sonnet-20241022"
) -> dict:
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
            model=judge_model,
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        judgment = message.content[0].text
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        judgment = f"ERROR: {str(e)}"

    return {
        "judgment": judgment.strip(),
        "order": order,
        "judge_model": judge_model,
    }

def parse_winner(judgment_text: str, order: str) -> str:
    """Parse winner from judgment text"""
    judgment_lower = judgment_text.lower()

    if "winner: a" in judgment_lower or "winner:a" in judgment_lower:
        winner_label = "A"
    elif "winner: b" in judgment_lower or "winner:b" in judgment_lower:
        winner_label = "B"
    elif "tie" in judgment_lower:
        return "tie"
    else:
        return "unknown"

    # Map back to actual model based on order
    if winner_label == "A":
        return "baseline" if order == "AB" else "brie"
    elif winner_label == "B":
        return "brie" if order == "AB" else "baseline"

    return "unknown"

# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

TEST_CONFIGS = {
    "reproducibility_run2": {
        "prompts": ORIGINAL_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "Reproducibility test - Run 2 of original prompts"
    },
    "reproducibility_run3": {
        "prompts": ORIGINAL_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "Reproducibility test - Run 3 of original prompts"
    },
    "philosophy_domain": {
        "prompts": PHILOSOPHY_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022", "claude-opus-4-20250514"],
        "description": "Philosophy-specific prompts with both judges"
    },
    "brainstorming_domain": {
        "prompts": BRAINSTORMING_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022", "claude-opus-4-20250514"],
        "description": "Brainstorming prompts with both judges"
    },
    "contemplative_domain": {
        "prompts": CONTEMPLATIVE_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022", "claude-opus-4-20250514"],
        "description": "Contemplative/meditative prompts with both judges"
    },
    "expanded_creative": {
        "prompts": EXPANDED_CREATIVE_PROMPTS,
        "temperature": 0.75,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "New creative prompts"
    },
    "temp_low": {
        "prompts": ORIGINAL_PROMPTS[:3],  # First 3 only
        "temperature": 0.5,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "Low temperature test (0.5)"
    },
    "temp_high": {
        "prompts": ORIGINAL_PROMPTS[:3],  # First 3 only
        "temperature": 1.0,
        "max_tokens": 512,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "High temperature test (1.0)"
    },
    "tokens_short": {
        "prompts": ORIGINAL_PROMPTS[:3],  # First 3 only
        "temperature": 0.75,
        "max_tokens": 256,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "Short response test (256 tokens)"
    },
    "tokens_long": {
        "prompts": ORIGINAL_PROMPTS[:3],  # First 3 only
        "temperature": 0.75,
        "max_tokens": 1024,
        "judges": ["claude-3-5-sonnet-20241022"],
        "description": "Long response test (1024 tokens)"
    },
}

# =============================================================================
# RUN TESTS
# =============================================================================

def run_test_config(config_name: str, config: dict) -> List[dict]:
    """Run a single test configuration"""
    print(f"\n{'='*80}")
    print(f"TEST: {config_name}")
    print(f"Description: {config['description']}")
    print(f"Prompts: {len(config['prompts'])}")
    print(f"Temperature: {config['temperature']}, Max Tokens: {config['max_tokens']}")
    print(f"Judges: {', '.join([j.split('-')[1] for j in config['judges']])}")
    print(f"{'='*80}\n")

    results = []

    for i, prompt in enumerate(config['prompts'], 1):
        print(f"[{i}/{len(config['prompts'])}] Prompt: {prompt[:60]}...")

        # Generate from both models
        print("  Generating baseline...")
        baseline_response, baseline_latency = generate_response(
            baseline_model, baseline_tokenizer, prompt,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

        print("  Generating Brie v2...")
        brie_response, brie_latency = generate_response(
            brie_model, brie_tokenizer, prompt,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

        # Judge with each judge model
        for judge_model in config['judges']:
            judge_name = judge_model.split('-')[1]  # Extract "sonnet" or "opus"
            print(f"  Getting {judge_name.upper()} judgment...")

            judgment_data = judge_with_claude(prompt, baseline_response, brie_response, judge_model)
            winner = parse_winner(judgment_data["judgment"], judgment_data["order"])

            result = {
                "config_name": config_name,
                "prompt_num": i,
                "prompt": prompt,
                "temperature": config['temperature'],
                "max_tokens": config['max_tokens'],
                "baseline_response": baseline_response,
                "baseline_length": len(baseline_response),
                "baseline_latency": round(baseline_latency, 3),
                "brie_response": brie_response,
                "brie_length": len(brie_response),
                "brie_latency": round(brie_latency, 3),
                "judge_model": judge_model,
                "judgment": judgment_data["judgment"],
                "presentation_order": judgment_data["order"],
                "winner": winner,
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)
            print(f"    Winner: {winner.upper()}")

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    all_results = []

    print("\nStarting comprehensive evaluation...")
    print(f"Total test configurations: {len(TEST_CONFIGS)}")

    start_time = time.time()

    for config_name, config in TEST_CONFIGS.items():
        try:
            config_results = run_test_config(config_name, config)
            all_results.extend(config_results)

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = f"exports/comprehensive_eval_{args.model_size}_intermediate_{timestamp}.jsonl"
            with open(intermediate_file, "w") as f:
                for result in all_results:
                    f.write(json.dumps(result) + "\n")
            print(f"\n  Intermediate results saved to {intermediate_file}")

        except Exception as e:
            print(f"\nERROR in {config_name}: {e}")
            continue

    # Save final results
    final_file = f"exports/comprehensive_eval_{args.model_size}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(final_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    # Calculate statistics
    total_time = time.time() - start_time

    print(f"\n\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION COMPLETE ({args.model_size.upper()})")
    print(f"{'='*80}")
    print(f"Model size: {args.model_size.upper()}")
    print(f"\nTotal comparisons: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Overall statistics
    baseline_wins = sum(1 for r in all_results if r["winner"] == "baseline")
    brie_wins = sum(1 for r in all_results if r["winner"] == "brie")
    ties = sum(1 for r in all_results if r["winner"] == "tie")
    unknown = sum(1 for r in all_results if r["winner"] == "unknown")

    print(f"\nOVERALL RESULTS:")
    print(f"  Baseline wins: {baseline_wins} ({baseline_wins/len(all_results)*100:.1f}%)")
    print(f"  Brie v2 wins: {brie_wins} ({brie_wins/len(all_results)*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/len(all_results)*100:.1f}%)")
    print(f"  Unknown: {unknown}")

    # By judge
    print(f"\nBY JUDGE:")
    for judge_model in set(r["judge_model"] for r in all_results):
        judge_results = [r for r in all_results if r["judge_model"] == judge_model]
        judge_brie_wins = sum(1 for r in judge_results if r["winner"] == "brie")
        judge_name = judge_model.split('-')[1]
        print(f"  {judge_name.upper()}: Brie wins {judge_brie_wins}/{len(judge_results)} ({judge_brie_wins/len(judge_results)*100:.1f}%)")

    print(f"\nResults saved to {final_file}")
    print("="*80)
