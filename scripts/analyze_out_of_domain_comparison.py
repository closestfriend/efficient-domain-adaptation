#!/usr/bin/env python3
"""Compare out-of-domain performance between 0.5B and 3B Brie models"""
import json
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print("Usage: python analyze_out_of_domain_comparison.py <0.5b_judged_file> <3b_judged_file>")
    print("\nExample:")
    print("  python scripts/analyze_out_of_domain_comparison.py \\")
    print("    exports/out_of_domain_0.5b_*_judged_*.jsonl \\")
    print("    exports/out_of_domain_3b_*_judged_*.jsonl")
    sys.exit(1)

file_0_5b = sys.argv[1]
file_3b = sys.argv[2]

print("=" * 100)
print("OUT-OF-DOMAIN PERFORMANCE: 0.5B vs 3B Comparison")
print("=" * 100)

# Load results
with open(file_0_5b, 'r') as f:
    results_0_5b = [json.loads(line) for line in f]

with open(file_3b, 'r') as f:
    results_3b = [json.loads(line) for line in f]

print(f"\n0.5B file: {file_0_5b}")
print(f"3B file:   {file_3b}\n")

def analyze_results(results, model_name):
    """Analyze win rates by category"""
    by_category = defaultdict(lambda: {"brie": 0, "baseline": 0, "tie": 0, "total": 0})

    for r in results:
        category = r.get("category", "unknown")
        winner = r.get("winner", "unknown")

        by_category[category]["total"] += 1
        if winner == "brie":
            by_category[category]["brie"] += 1
        elif winner == "baseline":
            by_category[category]["baseline"] += 1
        elif winner == "tie":
            by_category[category]["tie"] += 1

    # Calculate overall
    total_brie = sum(c["brie"] for c in by_category.values())
    total_baseline = sum(c["baseline"] for c in by_category.values())
    total_tie = sum(c["tie"] for c in by_category.values())
    total_all = sum(c["total"] for c in by_category.values())

    print(f"\n{'─' * 100}")
    print(f"{model_name} Out-of-Domain Results")
    print(f"{'─' * 100}\n")

    print(f"{'Category':<15} {'Total':<8} {'Brie Wins':<15} {'Baseline Wins':<18} {'Ties':<10} {'Brie Win %':<12}")
    print(f"{'-'*15} {'-'*8} {'-'*15} {'-'*18} {'-'*10} {'-'*12}")

    for category in sorted(by_category.keys()):
        stats = by_category[category]
        total = stats["total"]
        brie = stats["brie"]
        baseline = stats["baseline"]
        tie = stats["tie"]
        win_rate = (brie / total * 100) if total > 0 else 0

        print(f"{category:<15} {total:<8} "
              f"{brie:<6} ({brie/total*100:5.1f}%)  "
              f"{baseline:<6} ({baseline/total*100:5.1f}%)   "
              f"{tie:<6} ({tie/total*100:4.1f}%)  "
              f"{win_rate:5.1f}%")

    print(f"{'-'*15} {'-'*8} {'-'*15} {'-'*18} {'-'*10} {'-'*12}")
    overall_win_rate = (total_brie / total_all * 100) if total_all > 0 else 0
    print(f"{'OVERALL':<15} {total_all:<8} "
          f"{total_brie:<6} ({total_brie/total_all*100:5.1f}%)  "
          f"{total_baseline:<6} ({total_baseline/total_all*100:5.1f}%)   "
          f"{total_tie:<6} ({total_tie/total_all*100:4.1f}%)  "
          f"{overall_win_rate:5.1f}%")

    return by_category, overall_win_rate

# Analyze both models
stats_0_5b, overall_0_5b = analyze_results(results_0_5b, "Brie 0.5B")
stats_3b, overall_3b = analyze_results(results_3b, "Brie 3B")

# Comparison
print(f"\n{'=' * 100}")
print("COMPARISON: 3B Improvement over 0.5B")
print(f"{'=' * 100}\n")

print(f"{'Category':<15} {'0.5B Win %':<15} {'3B Win %':<15} {'Improvement':<15}")
print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")

for category in sorted(set(stats_0_5b.keys()) | set(stats_3b.keys())):
    stats_05 = stats_0_5b.get(category, {"brie": 0, "total": 1})
    stats_3 = stats_3b.get(category, {"brie": 0, "total": 1})

    rate_05 = (stats_05["brie"] / stats_05["total"] * 100) if stats_05["total"] > 0 else 0
    rate_3 = (stats_3["brie"] / stats_3["total"] * 100) if stats_3["total"] > 0 else 0
    improvement = rate_3 - rate_05

    print(f"{category:<15} {rate_05:5.1f}%{' '*9} {rate_3:5.1f}%{' '*9} {improvement:+5.1f}%")

print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
overall_improvement = overall_3b - overall_0_5b
print(f"{'OVERALL':<15} {overall_0_5b:5.1f}%{' '*9} {overall_3b:5.1f}%{' '*9} {overall_improvement:+5.1f}%")

print(f"\n{'=' * 100}")
print("KEY FINDINGS")
print(f"{'=' * 100}\n")

if overall_improvement > 0:
    print(f"✅ 3B model shows {overall_improvement:.1f}% improvement in out-of-domain tasks")
else:
    print(f"⚠️  3B model shows {abs(overall_improvement):.1f}% decrease in out-of-domain tasks")

# Find biggest improvements
improvements = []
for category in sorted(set(stats_0_5b.keys()) | set(stats_3b.keys())):
    stats_05 = stats_0_5b.get(category, {"brie": 0, "total": 1})
    stats_3 = stats_3b.get(category, {"brie": 0, "total": 1})

    rate_05 = (stats_05["brie"] / stats_05["total"] * 100) if stats_05["total"] > 0 else 0
    rate_3 = (stats_3["brie"] / stats_3["total"] * 100) if stats_3["total"] > 0 else 0
    improvement = rate_3 - rate_05

    improvements.append((category, rate_05, rate_3, improvement))

improvements.sort(key=lambda x: x[3], reverse=True)

print(f"\nBiggest improvements:")
for category, rate_05, rate_3, imp in improvements[:3]:
    print(f"  • {category}: {rate_05:.1f}% → {rate_3:.1f}% ({imp:+.1f}%)")

if len(improvements) > 3:
    print(f"\nBiggest regressions:")
    for category, rate_05, rate_3, imp in improvements[-3:]:
        print(f"  • {category}: {rate_05:.1f}% → {rate_3:.1f}% ({imp:+.1f}%)")

print(f"\n{'=' * 100}")
