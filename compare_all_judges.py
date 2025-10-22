#!/usr/bin/env python3
"""Compare Claude, GPT-4o, and Gemini judges on the same evaluations"""
import json
import glob
from collections import defaultdict

print("=" * 120)
print("CROSS-JUDGE COMPARISON: Claude 3.5 Sonnet vs GPT-4o vs Gemini 2.5 Flash Lite")
print("=" * 120)

# Define the evaluation files to analyze
eval_files = {
    "Comprehensive 0.5B": {
        "original": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED.jsonl",
        "gpt4o": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED_judged_gpt_4o_20251021_181849.jsonl",
        "gemini": "exports/comprehensive_eval_0.5b_final_20251016_220256_CORRECTED_judged_gemini_2_5_flash_lite_20251021_182548.jsonl",
    },
    "Comprehensive 3B": {
        "original": "exports/comprehensive_eval_3b_final_20251018_175044_CORRECTED.jsonl",
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

    # Load Claude judgments from original file
    with open(files["original"], 'r') as f:
        claude_results = [json.loads(line) for line in f]

    claude_total = len(claude_results)
    claude_brie = sum(1 for r in claude_results if r.get('winner') == 'brie')
    claude_baseline = sum(1 for r in claude_results if r.get('winner') == 'baseline')
    claude_tie = sum(1 for r in claude_results if r.get('winner') == 'tie')

    judges_data['Claude 3.5 Sonnet'] = {
        'total': claude_total,
        'brie': claude_brie,
        'baseline': claude_baseline,
        'tie': claude_tie,
        'brie_rate': claude_brie / claude_total * 100 if claude_total > 0 else 0,
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

    for judge_name in ['Claude 3.5 Sonnet', 'GPT-4o', 'Gemini 2.5 Flash Lite']:
        data = judges_data[judge_name]
        print(f"{judge_name:<25} {data['total']:<8} "
              f"{data['brie']:<6} ({data['brie']/data['total']*100:5.1f}%)  "
              f"{data['baseline']:<6} ({data['baseline']/data['total']*100:5.1f}%)   "
              f"{data['tie']:<6} ({data['tie']/data['total']*100:4.1f}%)  "
              f"{data['brie_rate']:5.1f}%")

    # Calculate agreement
    print(f"\n  Agreement Analysis:")

    # Prompt-by-prompt comparison
    agreements = defaultdict(int)
    disagreements = []

    for i in range(min(len(claude_results), len(gpt_results), len(gemini_results))):
        claude_winner = claude_results[i].get('winner')
        gpt_winner = gpt_results[i].get('winner')
        gemini_winner = gemini_results[i].get('winner')

        winners = [claude_winner, gpt_winner, gemini_winner]

        if claude_winner == gpt_winner == gemini_winner:
            agreements['all_three'] += 1
        elif claude_winner == gpt_winner or claude_winner == gemini_winner or gpt_winner == gemini_winner:
            agreements['two_agree'] += 1
        else:
            agreements['all_disagree'] += 1
            disagreements.append({
                'prompt_num': i + 1,
                'prompt': claude_results[i].get('prompt', '')[:60] + "...",
                'claude': claude_winner,
                'gpt4o': gpt_winner,
                'gemini': gemini_winner,
            })

    total_comparisons = min(len(claude_results), len(gpt_results), len(gemini_results))

    print(f"    All 3 judges agree:     {agreements['all_three']:3d}/{total_comparisons} ({agreements['all_three']/total_comparisons*100:5.1f}%)")
    print(f"    2 judges agree:         {agreements['two_agree']:3d}/{total_comparisons} ({agreements['two_agree']/total_comparisons*100:5.1f}%)")
    print(f"    All disagree:           {agreements['all_disagree']:3d}/{total_comparisons} ({agreements['all_disagree']/total_comparisons*100:5.1f}%)")

    # Pairwise agreements
    claude_gpt_agree = sum(1 for i in range(total_comparisons)
                           if claude_results[i].get('winner') == gpt_results[i].get('winner'))
    claude_gemini_agree = sum(1 for i in range(total_comparisons)
                              if claude_results[i].get('winner') == gemini_results[i].get('winner'))
    gpt_gemini_agree = sum(1 for i in range(total_comparisons)
                           if gpt_results[i].get('winner') == gemini_results[i].get('winner'))

    print(f"\n  Pairwise Agreement:")
    print(f"    Claude ↔ GPT-4o:        {claude_gpt_agree:3d}/{total_comparisons} ({claude_gpt_agree/total_comparisons*100:5.1f}%)")
    print(f"    Claude ↔ Gemini:        {claude_gemini_agree:3d}/{total_comparisons} ({claude_gemini_agree/total_comparisons*100:5.1f}%)")
    print(f"    GPT-4o ↔ Gemini:        {gpt_gemini_agree:3d}/{total_comparisons} ({gpt_gemini_agree/total_comparisons*100:5.1f}%)")

    if disagreements:
        print(f"\n  Cases where all 3 judges disagreed ({len(disagreements)} total):")
        for d in disagreements[:5]:  # Show first 5
            print(f"    #{d['prompt_num']}: {d['prompt']}")
            print(f"      Claude: {d['claude']:8s}  |  GPT-4o: {d['gpt4o']:8s}  |  Gemini: {d['gemini']:8s}")

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print("\nBrie Win Rates by Judge and Model Size:")
print(f"\n{'Model Size':<20} {'Claude 3.5 Sonnet':<20} {'GPT-4o':<20} {'Gemini 2.5 Flash Lite':<25}")
print(f"{'-'*20} {'-'*20} {'-'*20} {'-'*25}")

for eval_name, files in eval_files.items():
    # Load data
    with open(files["original"], 'r') as f:
        claude_results = [json.loads(line) for line in f]
    with open(files["gpt4o"], 'r') as f:
        gpt_results = [json.loads(line) for line in f]
    with open(files["gemini"], 'r') as f:
        gemini_results = [json.loads(line) for line in f]

    claude_rate = sum(1 for r in claude_results if r.get('winner') == 'brie') / len(claude_results) * 100
    gpt_rate = sum(1 for r in gpt_results if r.get('winner') == 'brie') / len(gpt_results) * 100
    gemini_rate = sum(1 for r in gemini_results if r.get('winner') == 'brie') / len(gemini_results) * 100

    print(f"{eval_name:<20} {claude_rate:5.1f}%{' '*14} {gpt_rate:5.1f}%{' '*14} {gemini_rate:5.1f}%")

print("\n" + "=" * 120)
