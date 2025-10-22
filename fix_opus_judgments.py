#!/usr/bin/env python3
"""Fix the winner parsing bug in Opus 4 judgment files"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python fix_opus_judgments.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file.replace(".jsonl", "_FIXED.jsonl")

print(f"Fixing {input_file}")
print(f"Output will be saved to {output_file}\n")

# Load the data
with open(input_file, 'r') as f:
    results = [json.loads(line) for line in f]

# Count original winners
orig_brie = sum(1 for r in results if r.get('winner') == 'brie')
orig_baseline = sum(1 for r in results if r.get('winner') == 'baseline')
orig_tie = sum(1 for r in results if r.get('winner') == 'tie')

print(f"ORIGINAL (buggy) results:")
print(f"  Brie: {orig_brie}/{len(results)} ({orig_brie/len(results)*100:.1f}%)")
print(f"  Baseline: {orig_baseline}/{len(results)} ({orig_baseline/len(results)*100:.1f}%)")
print(f"  Ties: {orig_tie}/{len(results)} ({orig_tie/len(results)*100:.1f}%)\n")

# Fix each result
fixed_results = []
corrections = 0

for result in results:
    judgment_text = result['judgment'].lower()
    order = result['presentation_order']

    # Parse winner from judgment
    if "winner: a" in judgment_text or "winner:a" in judgment_text:
        winner_label = "A"
    elif "winner: b" in judgment_text or "winner:b" in judgment_text:
        winner_label = "B"
    elif "tie" in judgment_text:
        winner_label = "Tie"
    else:
        winner_label = "unknown"

    # CORRECT mapping based on presentation order
    if order == "AB":
        # Response A = baseline, Response B = brie
        if winner_label == "A":
            winner = "baseline"
        elif winner_label == "B":
            winner = "brie"
        elif winner_label == "Tie":
            winner = "tie"
        else:
            winner = "unknown"
    elif order == "BA":
        # Response A = brie, Response B = baseline (REVERSED!)
        if winner_label == "A":
            winner = "brie"
        elif winner_label == "B":
            winner = "baseline"
        elif winner_label == "Tie":
            winner = "tie"
        else:
            winner = "unknown"
    else:
        winner = "unknown"

    # Track if we corrected it
    if result.get('winner') != winner:
        corrections += 1

    # Update result with correct winner
    fixed_result = {**result, 'winner': winner}
    fixed_results.append(fixed_result)

# Save fixed results
with open(output_file, 'w') as f:
    for result in fixed_results:
        f.write(json.dumps(result) + '\n')

# Count fixed winners
fixed_brie = sum(1 for r in fixed_results if r.get('winner') == 'brie')
fixed_baseline = sum(1 for r in fixed_results if r.get('winner') == 'baseline')
fixed_tie = sum(1 for r in fixed_results if r.get('winner') == 'tie')

print(f"FIXED results:")
print(f"  Brie: {fixed_brie}/{len(fixed_results)} ({fixed_brie/len(fixed_results)*100:.1f}%)")
print(f"  Baseline: {fixed_baseline}/{len(fixed_results)} ({fixed_baseline/len(fixed_results)*100:.1f}%)")
print(f"  Ties: {fixed_tie}/{len(fixed_results)} ({fixed_tie/len(fixed_results)*100:.1f}%)\n")

print(f"Corrections made: {corrections}/{len(results)}")
print(f"Saved to {output_file}")
