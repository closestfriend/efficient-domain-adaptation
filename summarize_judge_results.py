#!/usr/bin/env python3
"""Summarize and compare results from different judge models"""
import json
import glob
from collections import defaultdict

# Find all judged result files
judged_files = glob.glob("exports/*_judged_*.jsonl")

# Group by input file
results_by_input = defaultdict(list)

for file in judged_files:
    # Load results
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    if not results:
        continue

    # Get metadata
    judge_model = results[0].get('judge_model', 'unknown')
    judge_type = results[0].get('judge_type', 'unknown')

    # Determine input file from filename
    base_name = file.replace('exports/', '').replace('.jsonl', '')
    if '_judged_' in base_name:
        input_file = base_name.split('_judged_')[0]
    else:
        continue

    # Calculate statistics
    total = len(results)
    baseline_wins = sum(1 for r in results if r.get('winner') == 'baseline')
    brie_wins = sum(1 for r in results if r.get('winner') == 'brie')
    ties = sum(1 for r in results if r.get('winner') == 'tie')
    unknown = sum(1 for r in results if r.get('winner') == 'unknown')

    results_by_input[input_file].append({
        'judge_type': judge_type,
        'judge_model': judge_model,
        'file': file,
        'total': total,
        'baseline_wins': baseline_wins,
        'brie_wins': brie_wins,
        'ties': ties,
        'unknown': unknown,
        'brie_win_rate': brie_wins / total * 100 if total > 0 else 0,
    })

# Print summary
print("=" * 100)
print("MULTI-JUDGE EVALUATION SUMMARY")
print("=" * 100)

for input_file in sorted(results_by_input.keys()):
    print(f"\n{'─' * 100}")
    print(f"Input File: {input_file}")
    print(f"{'─' * 100}")

    judges = results_by_input[input_file]

    # Sort by judge type
    judges.sort(key=lambda x: (x['judge_type'], x['judge_model']))

    for judge_data in judges:
        judge_label = f"{judge_data['judge_type'].upper()}"
        if 'gpt' in judge_data['judge_model'].lower():
            model_name = "GPT-4o"
        elif 'gemini' in judge_data['judge_model'].lower():
            if '2.5' in judge_data['judge_model']:
                model_name = "Gemini 2.5 Flash Lite"
            elif '2.0' in judge_data['judge_model']:
                model_name = "Gemini 2.0 Flash"
            else:
                model_name = "Gemini 1.5 Pro"
        elif 'claude' in judge_data['judge_model'].lower():
            if '3-5-sonnet' in judge_data['judge_model']:
                model_name = "Claude 3.5 Sonnet"
            elif '3-7-sonnet' in judge_data['judge_model']:
                model_name = "Claude 3.7 Sonnet"
            elif 'opus' in judge_data['judge_model']:
                model_name = "Claude Opus 4"
            else:
                model_name = judge_data['judge_model']
        else:
            model_name = judge_data['judge_model']

        print(f"\n  Judge: {model_name}")
        print(f"    Total comparisons: {judge_data['total']}")
        print(f"    Brie wins:         {judge_data['brie_wins']:3d} ({judge_data['brie_win_rate']:5.1f}%)")
        print(f"    Baseline wins:     {judge_data['baseline_wins']:3d} ({judge_data['baseline_wins']/judge_data['total']*100:5.1f}%)")
        print(f"    Ties:              {judge_data['ties']:3d} ({judge_data['ties']/judge_data['total']*100:5.1f}%)")
        if judge_data['unknown'] > 0:
            print(f"    Unknown:           {judge_data['unknown']:3d}")

print("\n" + "=" * 100)
print("CROSS-JUDGE COMPARISON")
print("=" * 100)

# Group by eval type
eval_types = {}
for input_file in results_by_input.keys():
    if 'comprehensive_eval_0.5b' in input_file:
        eval_type = "Comprehensive 0.5B"
    elif 'comprehensive_eval_3b' in input_file:
        eval_type = "Comprehensive 3B"
    elif 'philosophy_comparison' in input_file:
        eval_type = "Philosophy Comparison 0.5B"
    else:
        eval_type = input_file

    if eval_type not in eval_types:
        eval_types[eval_type] = []
    eval_types[eval_type].extend(results_by_input[input_file])

for eval_type in sorted(eval_types.keys()):
    print(f"\n{eval_type}:")
    judges = eval_types[eval_type]
    judges.sort(key=lambda x: -x['brie_win_rate'])

    for judge_data in judges:
        if 'gpt' in judge_data['judge_model'].lower():
            model_name = "GPT-4o"
        elif 'gemini' in judge_data['judge_model'].lower():
            if '2.5' in judge_data['judge_model']:
                model_name = "Gemini 2.5 Flash Lite"
            elif '2.0' in judge_data['judge_model']:
                model_name = "Gemini 2.0 Flash"
            else:
                model_name = "Gemini 1.5 Pro"
        elif 'claude' in judge_data['judge_model'].lower():
            if '3-5-sonnet' in judge_data['judge_model']:
                model_name = "Claude 3.5 Sonnet"
            elif '3-7-sonnet' in judge_data['judge_model']:
                model_name = "Claude 3.7 Sonnet"
            elif 'opus' in judge_data['judge_model']:
                model_name = "Claude Opus 4"
            else:
                model_name = judge_data['judge_model']
        else:
            model_name = judge_data['judge_model']

        print(f"  {model_name:25s} → Brie wins: {judge_data['brie_win_rate']:5.1f}% ({judge_data['brie_wins']}/{judge_data['total']})")

print("\n" + "=" * 100)
