#!/usr/bin/env python3
"""Compare Claude, GPT-4o, and Gemini judges on the same evaluations"""
import json
import glob
from collections import defaultdict

print("=" * 120)
print("CROSS-JUDGE COMPARISON: Claude (Sonnet & Opus 4) vs GPT-4o vs Gemini 2.5 Flash Lite")
print("=" * 120)

# Define the evaluation files to analyze
import os

eval_files = {
    "Comprehensive 0.5B": {
        "sonnet": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED.jsonl",
        "opus": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED_judged_20251021_203643.jsonl",
        "gpt4o": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED_judged_gpt_4o_20251021_181849.jsonl",
        "gemini": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED_judged_gemini_2_5_flash_lite_20251021_182548.jsonl",
    },
    "Comprehensive 3B": {
        "sonnet": "exports/comprehensive_eval_3b_final_20251018_175044_CORRECTED.jsonl",
        "opus": "exports/comprehensive_eval_3b_final_20251018_175044_CORRECTED_judged_20251021_203743.jsonl",
        "gpt4o": "exports/comprehensive_eval_3b_final_20251018_175044_CORRECTED_judged_gpt_4o_20251021_182842.jsonl",
        "gemini": "exports/comprehensive_eval_3b_final_20251018_175044_CORRECTED_judged_gemini_2_5_flash_lite_20251021_182943.jsonl",
    },
}

for eval_name, files in eval_files.items():
    print(f"\n{'─' * 120}")
    print(f"{eval_name}")
    print(f"{'─' * 120}\n")

    # Load all judgments
    judges_data = {}

    # Load Claude Sonnet judgments
    with open(files["sonnet"], 'r') as f:
        sonnet_results = [json.loads(line) for line in f]

    # Filter to only Sonnet judgments (file may contain both)
    sonnet_results = [r for r in sonnet_results if 'sonnet' in r.get('judge_model', '').lower()]

    # Load Claude Opus 4 judgments if available
    if files["opus"]:
        with open(files["opus"], 'r') as f:
            opus_results = [json.loads(line) for line in f]
    else:
        opus_results = []

    # Claude 3.5 Sonnet
    sonnet_total = len(sonnet_results)
    sonnet_brie = sum(1 for r in sonnet_results if r.get('winner') == 'brie')
    sonnet_baseline = sum(1 for r in sonnet_results if r.get('winner') == 'baseline')
    sonnet_tie = sum(1 for r in sonnet_results if r.get('winner') == 'tie')

    judges_data['Claude 3.5 Sonnet'] = {
        'total': sonnet_total,
        'brie': sonnet_brie,
        'baseline': sonnet_baseline,
        'tie': sonnet_tie,
        'brie_rate': sonnet_brie / sonnet_total * 100 if sonnet_total > 0 else 0,
    }

    # Claude Opus 4
    opus_total = len(opus_results)
    opus_brie = sum(1 for r in opus_results if r.get('winner') == 'brie')
    opus_baseline = sum(1 for r in opus_results if r.get('winner') == 'baseline')
    opus_tie = sum(1 for r in opus_results if r.get('winner') == 'tie')

    judges_data['Claude Opus 4'] = {
        'total': opus_total,
        'brie': opus_brie,
        'baseline': opus_baseline,
        'tie': opus_tie,
        'brie_rate': opus_brie / opus_total * 100 if opus_total > 0 else 0,
    }

    # Load GPT-4o judgments
    with open(files["gpt4o"], 'r') as f:
        gpt_results = [json.loads(line) for line in f]

    gpt_total = len(gpt_results)
    gpt_brie = sum(1 for r in gpt_results if r.get('winner') == 'brie')
    gpt_baseline = sum(1 for r in gpt_results if r.get('winner') == 'baseline')
    gpt_tie = sum(1 for r in gpt_results if r.get('winner') == 'tie')

    judges_data['GPT-4o'] = {
        'total': gpt_total,
        'brie': gpt_brie,
        'baseline': gpt_baseline,
        'tie': gpt_tie,
        'brie_rate': gpt_brie / gpt_total * 100 if gpt_total > 0 else 0,
    }

    # Load Gemini judgments
    with open(files["gemini"], 'r') as f:
        gemini_results = [json.loads(line) for line in f]

    gemini_total = len(gemini_results)
    gemini_brie = sum(1 for r in gemini_results if r.get('winner') == 'brie')
    gemini_baseline = sum(1 for r in gemini_results if r.get('winner') == 'baseline')
    gemini_tie = sum(1 for r in gemini_results if r.get('winner') == 'tie')

    judges_data['Gemini 2.5 Flash Lite'] = {
        'total': gemini_total,
        'brie': gemini_brie,
        'baseline': gemini_baseline,
        'tie': gemini_tie,
        'brie_rate': gemini_brie / gemini_total * 100 if gemini_total > 0 else 0,
    }

    # Print comparison table
    print(f"{'Judge':<25} {'Total':<8} {'Brie Wins':<15} {'Baseline Wins':<18} {'Ties':<10} {'Brie Win %':<12}")
    print(f"{'-'*25} {'-'*8} {'-'*15} {'-'*18} {'-'*10} {'-'*12}")

    for judge_name in ['Claude 3.5 Sonnet', 'Claude Opus 4', 'GPT-4o', 'Gemini 2.5 Flash Lite']:
        data = judges_data[judge_name]
        if data['total'] == 0:
            print(f"{judge_name:<25} {'N/A':<8} {'N/A':<15} {'N/A':<18} {'N/A':<10} {'N/A':<12}")
        else:
            print(f"{judge_name:<25} {data['total']:<8} "
                  f"{data['brie']:<6} ({data['brie']/data['total']*100:5.1f}%)  "
                  f"{data['baseline']:<6} ({data['baseline']/data['total']*100:5.1f}%)   "
                  f"{data['tie']:<6} ({data['tie']/data['total']*100:4.1f}%)  "
                  f"{data['brie_rate']:5.1f}%")

    # Calculate agreement
    print(f"\n  Agreement Analysis:")

    # Prompt-by-prompt comparison (using GPT and Gemini since they have same prompts)
    # Note: Claude Sonnet/Opus have different prompt counts
    agreements = defaultdict(int)
    disagreements = []

    for i in range(min(len(gpt_results), len(gemini_results))):
        gpt_winner = gpt_results[i].get('winner')
        gemini_winner = gemini_results[i].get('winner')

        if gpt_winner == gemini_winner:
            agreements['agree'] += 1
        else:
            agreements['disagree'] += 1
            disagreements.append({
                'prompt_num': i + 1,
                'prompt': gpt_results[i].get('prompt', '')[:60] + "...",
                'gpt4o': gpt_winner,
                'gemini': gemini_winner,
            })

    total_comparisons = min(len(gpt_results), len(gemini_results))

    print(f"    GPT-4o ↔ Gemini agreement: {agreements['agree']:3d}/{total_comparisons} ({agreements['agree']/total_comparisons*100:5.1f}%)")

    if disagreements and len(disagreements) <= 10:
        print(f"\n  Cases where GPT-4o and Gemini disagreed ({len(disagreements)} total):")
        for d in disagreements:
            print(f"    #{d['prompt_num']}: {d['prompt']}")
            print(f"      GPT-4o: {d['gpt4o']:8s}  |  Gemini: {d['gemini']:8s}")

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print("\nBrie Win Rates by Judge and Model Size:")
print(f"\n{'Model Size':<20} {'Claude Sonnet':<15} {'Claude Opus 4':<15} {'GPT-4o':<15} {'Gemini 2.5':<15}")
print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

for eval_name, files in eval_files.items():
    # Load data
    with open(files["sonnet"], 'r') as f:
        sonnet_data = [json.loads(line) for line in f]
    sonnet_results = [r for r in sonnet_data if 'sonnet' in r.get('judge_model', '').lower()]

    if files["opus"]:
        with open(files["opus"], 'r') as f:
            opus_results = [json.loads(line) for line in f]
    else:
        opus_results = []

    with open(files["gpt4o"], 'r') as f:
        gpt_results = [json.loads(line) for line in f]
    with open(files["gemini"], 'r') as f:
        gemini_results = [json.loads(line) for line in f]

    sonnet_rate = sum(1 for r in sonnet_results if r.get('winner') == 'brie') / len(sonnet_results) * 100 if sonnet_results else 0
    opus_rate = sum(1 for r in opus_results if r.get('winner') == 'brie') / len(opus_results) * 100 if opus_results else 0
    gpt_rate = sum(1 for r in gpt_results if r.get('winner') == 'brie') / len(gpt_results) * 100
    gemini_rate = sum(1 for r in gemini_results if r.get('winner') == 'brie') / len(gemini_results) * 100

    print(f"{eval_name:<20} {sonnet_rate:5.1f}%{' '*9} {opus_rate:5.1f}%{' '*9} {gpt_rate:5.1f}%{' '*9} {gemini_rate:5.1f}%")

print("\n" + "=" * 120)
