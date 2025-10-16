#!/usr/bin/env python3
"""Analyze what made Brie win vs lose"""
import json
import sys
import argparse
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description="Analyze Brie wins and losses from evaluation results")
parser.add_argument(
    "results_file",
    type=str,
    help="Path to comprehensive eval results JSONL file"
)
args = parser.parse_args()

# Load results
try:
    with open(args.results_file) as f:
        results = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: Could not find file {args.results_file}")
    sys.exit(1)

print("="*80)
print("BRIE WIN/LOSS ANALYSIS")
print("="*80)
print(f"Analyzing: {args.results_file}")
print("="*80)

# Separate wins and losses
brie_wins = [r for r in results if r["winner"] == "brie"]
baseline_wins = [r for r in results if r["winner"] == "baseline"]
ties = [r for r in results if r["winner"] == "tie"]

print(f"\nTotal: {len(results)} comparisons")
print(f"  Brie wins: {len(brie_wins)} ({len(brie_wins)/len(results)*100:.1f}%)")
print(f"  Baseline wins: {len(baseline_wins)} ({len(baseline_wins)/len(results)*100:.1f}%)")
print(f"  Ties: {len(ties)} ({len(ties)/len(results)*100:.1f}%)")

# =============================================================================
# PATTERN 1: Response Length
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 1: RESPONSE LENGTH")
print("="*80)

brie_wins_avg_len = sum(r["brie_length"] for r in brie_wins) / len(brie_wins) if brie_wins else 0
baseline_wins_brie_len = sum(r["brie_length"] for r in baseline_wins) / len(baseline_wins) if baseline_wins else 0

print(f"\nWhen Brie WINS:")
print(f"  Avg Brie length: {brie_wins_avg_len:.0f} chars")
print(f"  Avg Baseline length: {sum(r['baseline_length'] for r in brie_wins) / len(brie_wins):.0f} chars")
print(f"  Brie is {((brie_wins_avg_len / (sum(r['baseline_length'] for r in brie_wins) / len(brie_wins))) - 1) * 100:+.1f}% longer/shorter")

print(f"\nWhen Brie LOSES:")
print(f"  Avg Brie length: {baseline_wins_brie_len:.0f} chars")
print(f"  Avg Baseline length: {sum(r['baseline_length'] for r in baseline_wins) / len(baseline_wins):.0f} chars")
print(f"  Brie is {((baseline_wins_brie_len / (sum(r['baseline_length'] for r in baseline_wins) / len(baseline_wins))) - 1) * 100:+.1f}% longer/shorter")

# =============================================================================
# PATTERN 2: By Test Configuration
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 2: BY TEST CONFIGURATION")
print("="*80)

config_stats = defaultdict(lambda: {"brie": 0, "baseline": 0, "tie": 0, "unknown": 0, "total": 0})
for r in results:
    config_stats[r["config_name"]]["total"] += 1
    winner = r["winner"]
    if winner in ["brie", "baseline", "tie", "unknown"]:
        config_stats[r["config_name"]][winner] += 1

for config, stats in sorted(config_stats.items()):
    brie_rate = stats["brie"] / stats["total"] * 100
    print(f"\n{config}:")
    print(f"  Brie wins: {stats['brie']}/{stats['total']} ({brie_rate:.1f}%)")
    print(f"  Baseline wins: {stats['baseline']}/{stats['total']}")
    print(f"  Ties: {stats['tie']}/{stats['total']}")

# =============================================================================
# PATTERN 3: By Temperature
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 3: BY TEMPERATURE")
print("="*80)

temp_stats = defaultdict(lambda: {"brie": 0, "baseline": 0, "tie": 0, "unknown": 0, "total": 0})
for r in results:
    temp = r["temperature"]
    temp_stats[temp]["total"] += 1
    winner = r["winner"]
    if winner in ["brie", "baseline", "tie", "unknown"]:
        temp_stats[temp][winner] += 1

for temp in sorted(temp_stats.keys()):
    stats = temp_stats[temp]
    brie_rate = stats["brie"] / stats["total"] * 100
    print(f"\nTemp {temp}:")
    print(f"  Brie wins: {stats['brie']}/{stats['total']} ({brie_rate:.1f}%)")
    print(f"  Baseline wins: {stats['baseline']}/{stats['total']}")

# =============================================================================
# PATTERN 4: By Max Tokens
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 4: BY MAX TOKENS")
print("="*80)

token_stats = defaultdict(lambda: {"brie": 0, "baseline": 0, "tie": 0, "unknown": 0, "total": 0})
for r in results:
    tokens = r["max_tokens"]
    token_stats[tokens]["total"] += 1
    winner = r["winner"]
    if winner in ["brie", "baseline", "tie", "unknown"]:
        token_stats[tokens][winner] += 1

for tokens in sorted(token_stats.keys()):
    stats = token_stats[tokens]
    brie_rate = stats["brie"] / stats["total"] * 100
    print(f"\nMax tokens {tokens}:")
    print(f"  Brie wins: {stats['brie']}/{stats['total']} ({brie_rate:.1f}%)")
    print(f"  Baseline wins: {stats['baseline']}/{stats['total']}")

# =============================================================================
# PATTERN 5: Judge Agreement
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 5: JUDGE DISAGREEMENT")
print("="*80)

# Group by config + prompt
prompt_groups = defaultdict(list)
for r in results:
    key = (r["config_name"], r["prompt_num"])
    prompt_groups[key].append(r)

disagreements = []
for key, group in prompt_groups.items():
    if len(group) == 2:  # Both judges evaluated
        winners = [r["winner"] for r in group]
        if winners[0] != winners[1]:
            disagreements.append((key, group))

print(f"\nTotal prompts evaluated by multiple judges: {len([g for g in prompt_groups.values() if len(g) == 2])}")
print(f"Disagreements: {len(disagreements)} ({len(disagreements) / len([g for g in prompt_groups.values() if len(g) == 2]) * 100:.1f}%)")

print(f"\nExamples of judge disagreements:")
for i, ((config, prompt_num), group) in enumerate(disagreements[:5], 1):
    print(f"\n{i}. {config} - Prompt #{prompt_num}")
    print(f"   '{group[0]['prompt'][:60]}...'")
    for r in group:
        judge_name = "Sonnet" if "sonnet" in r["judge_model"] else "Opus"
        print(f"   {judge_name}: {r['winner'].upper()}")

# =============================================================================
# PATTERN 6: Strongest Wins and Worst Losses
# =============================================================================
print(f"\n{'='*80}")
print("PATTERN 6: STRONGEST BRIE WINS")
print("="*80)

# Sort by length difference (where Brie is much longer)
brie_wins_sorted = sorted(brie_wins, key=lambda r: r["brie_length"] - r["baseline_length"], reverse=True)

print("\nTop 3 Brie wins (by length advantage):")
for i, r in enumerate(brie_wins_sorted[:3], 1):
    print(f"\n{i}. {r['config_name']} - Prompt #{r['prompt_num']}")
    print(f"   '{r['prompt'][:70]}'")
    print(f"   Brie: {r['brie_length']} chars, Baseline: {r['baseline_length']} chars")
    print(f"   Judge reasoning (first 150 chars):")
    reasoning_start = r["judgment"].find("Reasoning:")
    if reasoning_start != -1:
        reasoning = r["judgment"][reasoning_start+10:reasoning_start+160].strip()
        print(f"   {reasoning}...")

print(f"\n{'='*80}")
print("PATTERN 7: WORST BRIE LOSSES")
print("="*80)

# Sort losses by how much judge favored baseline
print("\nTop 3 Baseline wins (strong victories):")
for i, r in enumerate(baseline_wins[:3], 1):
    print(f"\n{i}. {r['config_name']} - Prompt #{r['prompt_num']}")
    print(f"   '{r['prompt'][:70]}'")
    print(f"   Brie: {r['brie_length']} chars, Baseline: {r['baseline_length']} chars")
    print(f"   Judge reasoning (first 150 chars):")
    reasoning_start = r["judgment"].find("Reasoning:")
    if reasoning_start != -1:
        reasoning = r["judgment"][reasoning_start+10:reasoning_start+160].strip()
        print(f"   {reasoning}...")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print("="*80)

print(f"\n1. LENGTH PATTERN:")
if brie_wins_avg_len > baseline_wins_brie_len:
    print(f"   Brie tends to win when it generates LONGER responses")
    print(f"   (Wins avg: {brie_wins_avg_len:.0f} chars, Losses avg: {baseline_wins_brie_len:.0f} chars)")
else:
    print(f"   Brie tends to win when it generates SHORTER responses")

print(f"\n2. BEST CONFIGURATIONS:")
best_config = max(config_stats.items(), key=lambda x: x[1]["brie"] / x[1]["total"])
print(f"   {best_config[0]}: {best_config[1]['brie']}/{best_config[1]['total']} wins")

worst_config = min(config_stats.items(), key=lambda x: x[1]["brie"] / x[1]["total"])
print(f"\n3. WORST CONFIGURATIONS:")
print(f"   {worst_config[0]}: {worst_config[1]['brie']}/{worst_config[1]['total']} wins")

print(f"\n4. JUDGE AGREEMENT:")
if disagreements:
    print(f"   Judges disagree {len(disagreements) / len([g for g in prompt_groups.values() if len(g) == 2]) * 100:.1f}% of the time")
    print(f"   This suggests responses are CLOSE in quality")

print(f"\n{'='*80}")
