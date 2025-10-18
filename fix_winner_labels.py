#!/usr/bin/env python3
"""
Fix winner labels in evaluation results due to parse_winner bug

BUG: The original parse_winner function incorrectly inverted results
when order == "BA". Response A is always baseline, Response B is always brie,
regardless of presentation order.

This script re-parses all judgments with the corrected logic.
"""
import json
import sys
from datetime import datetime

def parse_winner_CORRECTED(judgment_text: str, order: str) -> str:
    """Corrected parse_winner function

    Response A is ALWAYS baseline, Response B is ALWAYS brie.
    The 'order' parameter only controls presentation order, not label mapping.
    """
    judgment_lower = judgment_text.lower()

    if "winner: a" in judgment_lower or "winner:a" in judgment_lower:
        winner_label = "A"
    elif "winner: b" in judgment_lower or "winner:b" in judgment_lower:
        winner_label = "B"
    elif "tie" in judgment_lower:
        return "tie"
    else:
        return "unknown"

    # Response A is always baseline, Response B is always brie
    if winner_label == "A":
        return "baseline"
    elif winner_label == "B":
        return "brie"

    return "unknown"

def parse_winner_BUGGY(judgment_text: str, order: str) -> str:
    """Original buggy parse_winner function (for comparison)"""
    judgment_lower = judgment_text.lower()

    if "winner: a" in judgment_lower or "winner:a" in judgment_lower:
        winner_label = "A"
    elif "winner: b" in judgment_lower or "winner:b" in judgment_lower:
        winner_label = "B"
    elif "tie" in judgment_lower:
        return "tie"
    else:
        return "unknown"

    # BUGGY: Incorrectly inverts when order == "BA"
    if winner_label == "A":
        return "baseline" if order == "AB" else "brie"
    elif winner_label == "B":
        return "brie" if order == "AB" else "baseline"

    return "unknown"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_winner_labels.py <input_jsonl_file>")
        print("Example: python3 fix_winner_labels.py exports/comprehensive_eval_3b_final_20251018_175044.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".jsonl", "_CORRECTED.jsonl")

    print("=" * 80)
    print("FIXING WINNER LABELS DUE TO parse_winner BUG")
    print("=" * 80)
    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Load results
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Re-parse all winners
    changes = 0
    for result in results:
        judgment = result.get("judgment", "")
        order = result.get("presentation_order", "AB")

        old_winner = result.get("winner", "unknown")
        new_winner = parse_winner_CORRECTED(judgment, order)

        if old_winner != new_winner:
            changes += 1
            result["winner_ORIGINAL_BUGGY"] = old_winner

        result["winner"] = new_winner

    print(f"Changed {changes} results ({changes/len(results)*100:.1f}%)")
    print()

    # Save corrected results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Calculate new statistics
    baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
    brie_wins = sum(1 for r in results if r["winner"] == "brie")
    ties = sum(1 for r in results if r["winner"] == "tie")
    unknown = sum(1 for r in results if r["winner"] == "unknown")

    print("=" * 80)
    print("CORRECTED RESULTS")
    print("=" * 80)
    print(f"Total comparisons: {len(results)}")
    print(f"  Baseline wins: {baseline_wins} ({baseline_wins/len(results)*100:.1f}%)")
    print(f"  Brie v2 wins: {brie_wins} ({brie_wins/len(results)*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/len(results)*100:.1f}%)")
    print(f"  Unknown: {unknown}")

    # By judge
    print(f"\nBY JUDGE:")
    for judge_model in set(r["judge_model"] for r in results):
        judge_results = [r for r in results if r["judge_model"] == judge_model]
        judge_brie_wins = sum(1 for r in judge_results if r["winner"] == "brie")
        judge_name = judge_model.split('-')[1] if '-' in judge_model else judge_model
        print(f"  {judge_name.upper()}: Brie wins {judge_brie_wins}/{len(judge_results)} ({judge_brie_wins/len(judge_results)*100:.1f}%)")

    print(f"\nCorrected results saved to: {output_file}")
    print("=" * 80)
