#!/usr/bin/env python3
"""Analyze multiple runs of philosophy comparison tests"""
import json
import glob
from statistics import mean, stdev

# Load all run files
run_files = sorted(glob.glob("exports/philosophy_comparison_run*.jsonl"))
print(f"Found {len(run_files)} run files\n")

all_runs = []
for run_file in run_files:
    with open(run_file) as f:
        run_data = [json.loads(line) for line in f]
        all_runs.append(run_data)

# Aggregate stats per prompt
prompt_stats = {}
for run_idx, run in enumerate(all_runs, 1):
    for item in run:
        prompt_num = item["prompt_num"]
        if prompt_num not in prompt_stats:
            prompt_stats[prompt_num] = {
                "prompt": item["prompt"],
                "baseline_latencies": [],
                "brie_latencies": [],
                "baseline_lengths": [],
                "brie_lengths": [],
            }

        prompt_stats[prompt_num]["baseline_latencies"].append(item["baseline_latency_s"])
        prompt_stats[prompt_num]["brie_latencies"].append(item["brie_latency_s"])
        prompt_stats[prompt_num]["baseline_lengths"].append(len(item["baseline_output"]))
        prompt_stats[prompt_num]["brie_lengths"].append(len(item["brie_output"]))

# Calculate overall statistics
print("="*80)
print("AGGREGATE STATISTICS ACROSS ALL RUNS (n=4)")
print("="*80)

all_baseline_latencies = []
all_brie_latencies = []
all_baseline_lengths = []
all_brie_lengths = []

for prompt_num in sorted(prompt_stats.keys()):
    stats = prompt_stats[prompt_num]
    all_baseline_latencies.extend(stats["baseline_latencies"])
    all_brie_latencies.extend(stats["brie_latencies"])
    all_baseline_lengths.extend(stats["baseline_lengths"])
    all_brie_lengths.extend(stats["brie_lengths"])

print(f"\nTotal samples: {len(all_baseline_latencies)} per model")
print(f"Total prompts: {len(prompt_stats)}")
print(f"Runs: {len(all_runs)}")

print("\n" + "-"*80)
print("LATENCY (seconds)")
print("-"*80)
print(f"Baseline - Mean: {mean(all_baseline_latencies):.2f}s, StdDev: {stdev(all_baseline_latencies):.2f}s")
print(f"Brie v2  - Mean: {mean(all_brie_latencies):.2f}s, StdDev: {stdev(all_brie_latencies):.2f}s")
print(f"Difference: Brie v2 is {mean(all_brie_latencies) - mean(all_baseline_latencies):.2f}s slower on average")

print("\n" + "-"*80)
print("RESPONSE LENGTH (characters)")
print("-"*80)
print(f"Baseline - Mean: {mean(all_baseline_lengths):.0f} chars, StdDev: {stdev(all_baseline_lengths):.0f}")
print(f"Brie v2  - Mean: {mean(all_brie_lengths):.0f} chars, StdDev: {stdev(all_brie_lengths):.0f}")
print(f"Difference: Brie v2 is {((mean(all_brie_lengths) / mean(all_baseline_lengths)) - 1) * 100:.1f}% longer")

print("\n" + "-"*80)
print("PER-PROMPT BREAKDOWN")
print("-"*80)

for prompt_num in sorted(prompt_stats.keys()):
    stats = prompt_stats[prompt_num]
    print(f"\n[{prompt_num}] {stats['prompt'][:60]}...")

    baseline_lat_mean = mean(stats["baseline_latencies"])
    brie_lat_mean = mean(stats["brie_latencies"])
    baseline_len_mean = mean(stats["baseline_lengths"])
    brie_len_mean = mean(stats["brie_lengths"])

    print(f"  Latency - Baseline: {baseline_lat_mean:.2f}s, Brie: {brie_lat_mean:.2f}s (Δ{brie_lat_mean - baseline_lat_mean:+.2f}s)")
    print(f"  Length  - Baseline: {baseline_len_mean:.0f} chars, Brie: {brie_len_mean:.0f} chars ({((brie_len_mean/baseline_len_mean-1)*100):+.1f}%)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Brie v2 consistently produces:")
print(f"  • {((mean(all_brie_lengths) / mean(all_baseline_lengths)) - 1) * 100:.1f}% longer responses")
print(f"  • Takes {mean(all_brie_latencies) - mean(all_baseline_latencies):.2f}s more per response")
print(f"  • More detailed, philosophical style (qualitative observation)")
