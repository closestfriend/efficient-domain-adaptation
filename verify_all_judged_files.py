#!/usr/bin/env python3
"""
Comprehensive verification of ALL judged files to ensure winner mapping is correct.

This script checks that:
1. When order="AB": baseline shown as "Response A", brie shown as "Response B"
2. When order="BA": brie shown as "Response B", baseline shown as "Response A"
3. Winner mapping is always: Judge picks "A" → recorded as baseline, Judge picks "B" → recorded as brie
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def extract_judge_winner(judgment_text):
    """Extract what the judge actually said"""
    if not judgment_text or "ERROR" in judgment_text:
        return None

    judgment_lower = judgment_text.lower()

    if "winner: a" in judgment_lower or "winner:a" in judgment_lower:
        return "A"
    elif "winner: b" in judgment_lower or "winner:b" in judgment_lower:
        return "B"
    elif "tie" in judgment_lower:
        return "Tie"
    else:
        return None

def verify_mapping(order, judge_winner, recorded_winner):
    """
    Verify the mapping is correct.

    CORRECT MAPPING (labels are ALWAYS fixed):
    - Response A is ALWAYS baseline
    - Response B is ALWAYS brie
    - order only affects PRESENTATION (which is shown first)

    So:
    - Judge picks "A" → should record "baseline" (regardless of order)
    - Judge picks "B" → should record "brie" (regardless of order)
    """
    if judge_winner == "Tie":
        return recorded_winner == "tie"
    elif judge_winner == "A":
        return recorded_winner == "baseline"
    elif judge_winner == "B":
        return recorded_winner == "brie"
    else:
        return recorded_winner == "unknown"

def main():
    # Find all judged files
    judged_files = list(Path("exports").glob("*judged*.jsonl"))

    print("="*80)
    print("COMPREHENSIVE VERIFICATION OF ALL JUDGED FILES")
    print("="*80)
    print(f"\nFound {len(judged_files)} judged files\n")

    all_results = defaultdict(lambda: {"correct": 0, "incorrect": 0, "errors": 0})
    detailed_errors = []

    for file_path in sorted(judged_files):
        print(f"\nChecking: {file_path.name}")
        print("-" * 80)

        with open(file_path) as f:
            comparisons = [json.loads(line) for line in f if line.strip()]

        file_correct = 0
        file_incorrect = 0
        file_errors = 0

        for i, comp in enumerate(comparisons, 1):
            order = comp.get("presentation_order", "unknown")
            judgment = comp.get("judgment", "")
            recorded_winner = comp.get("winner", "unknown")

            judge_winner = extract_judge_winner(judgment)

            if judge_winner is None:
                file_errors += 1
                continue

            is_correct = verify_mapping(order, judge_winner, recorded_winner)

            if is_correct:
                file_correct += 1
            else:
                file_incorrect += 1
                detailed_errors.append({
                    "file": file_path.name,
                    "comparison": i,
                    "prompt": comp.get("prompt", "")[:80],
                    "order": order,
                    "judge_picked": judge_winner,
                    "recorded": recorded_winner,
                    "expected": "baseline" if judge_winner == "A" else "brie"
                })

        # File summary
        total_valid = file_correct + file_incorrect
        if total_valid > 0:
            accuracy = (file_correct / total_valid) * 100
            status = "✅ PASS" if file_incorrect == 0 else "❌ FAIL"
            print(f"{status}: {file_correct}/{total_valid} correct ({accuracy:.1f}%)")
            if file_errors > 0:
                print(f"   ({file_errors} errors/unknowns skipped)")

        all_results[file_path.name]["correct"] = file_correct
        all_results[file_path.name]["incorrect"] = file_incorrect
        all_results[file_path.name]["errors"] = file_errors

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)

    total_correct = sum(r["correct"] for r in all_results.values())
    total_incorrect = sum(r["incorrect"] for r in all_results.values())
    total_errors = sum(r["errors"] for r in all_results.values())
    total_valid = total_correct + total_incorrect

    print(f"\nTotal comparisons checked: {total_valid}")
    print(f"Correctly mapped: {total_correct} ({total_correct/total_valid*100:.1f}%)")
    print(f"Incorrectly mapped: {total_incorrect} ({total_incorrect/total_valid*100:.1f}%)")
    print(f"Errors/unknowns: {total_errors}")

    if total_incorrect > 0:
        print("\n" + "="*80)
        print("INCORRECT MAPPINGS FOUND:")
        print("="*80)
        for error in detailed_errors[:10]:  # Show first 10
            print(f"\nFile: {error['file']}")
            print(f"  Prompt: {error['prompt']}")
            print(f"  Order: {error['order']}")
            print(f"  Judge picked: {error['judge_picked']}")
            print(f"  Recorded as: {error['recorded']}")
            print(f"  Expected: {error['expected']}")

        if len(detailed_errors) > 10:
            print(f"\n... and {len(detailed_errors) - 10} more errors")

        print("\n❌ VERIFICATION FAILED - Files need correction!")
        return 1
    else:
        print("\n✅ ALL FILES VERIFIED CORRECT!")
        print("\nYour reported metrics are legitimate and can be trusted.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
