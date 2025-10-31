# Review Protocol for Public-Facing Work

## Purpose

This document establishes a review process for any work before it enters the public sphere (HuggingFace, GitHub READMEs, blog posts, portfolio pieces, etc.). The goal is to catch reputation-damaging mistakes while preserving the authentic quality of the work.

## Review Principles

### 1. Measured Language
- **Avoid:** "groundbreaking", "novel", "significant", "critical discovery"
- **Use:** "found", "observed", "tested", "compared"
- Let data speak for itself without editorializing

### 2. Accurate Claims
- Only state what the data directly shows
- "Observation: X showed Y" not "Finding: X proves Y"
- "In testing, A performed better than B" not "A is superior to B"
- No extrapolation beyond evaluation scope

### 3. Honest Limitations
- Acknowledge constraints explicitly
- Don't hide trade-offs
- "Optimized for X, less optimal for Y" > "Great at X"

### 4. Appropriate Framing
- "Controlled comparison" > "Research study"
- "Tested systematically" > "Comprehensive research"
- "Portfolio project" > "Production system"
- Match scope to actual work done

### 5. No Overselling
- Real numbers without spin: "75.4% win rate" not "Strong 75% win rate"
- Factual: "Trained on 1,213 examples" not "Extensively trained"
- Neutral: "Tested across 3 architectures" not "Rigorous multi-architecture study"

## Red Flags to Catch

### Language
- [ ] Claims of novelty without literature review
- [ ] Words ending in "-est" (best, fastest, strongest)
- [ ] Unqualified universals ("always", "never", "all")
- [ ] Dramatic language ("revolutionary", "game-changing")
- [ ] False precision ("significantly" without statistical test)

### Technical
- [ ] Cherry-picked results
- [ ] Evaluation methodology not clearly described
- [ ] Baseline comparisons missing
- [ ] Sample sizes too small for claims made
- [ ] Confounding factors not acknowledged

### Scope Creep
- [ ] Claims beyond what was tested
- [ ] Generalizations from narrow evaluation
- [ ] "This shows X" when only tested Y
- [ ] Future work presented as current capability

## Review Checklist

Before any public release:

- [ ] Every claim has supporting data in the repository
- [ ] All numbers are accurate and verifiable
- [ ] Limitations section is honest and complete
- [ ] Language is measured and factual
- [ ] No comparisons to other work without citation
- [ ] Methodology is transparent and replicable
- [ ] Title and abstract match actual scope
- [ ] Code/data availability clearly stated

## What Good Looks Like

**Instead of:**
> "Our novel fine-tuning approach achieves state-of-the-art results, significantly outperforming baseline with a groundbreaking 91% win rate across comprehensive benchmarks."

**Write:**
> "Fine-tuned on 1,213 examples. In blind A/B testing (n=57), judges preferred this model in 91.2% of comparisons. Tested on philosophy and creative writing tasks. See EVALUATION.md for methodology."

## Reviewer Role

When reviewing work, the reviewer should:

1. **Read as a skeptical peer** - Would this hold up to questions?
2. **Check every number** - Can I find this in the data?
3. **Question every adjective** - Is "significant" earned? Is "comprehensive" accurate?
4. **Imagine worst-case readers** - Could this be misinterpreted? Could someone call out exaggeration?
5. **Protect long-term reputation** - Does this build or erode trust?

## Examples from This Project

### Good Edits Made

**Before:** "Critical discovery: top_p significantly handicaps creative models"
**After:** "Note: In testing, top_p=0.95 constrained creative outputs"

**Before:** "Multi-architecture personality transfer study"
**After:** "Controlled comparison testing across architectures"

**Before:** "Key Finding: Personality successfully transfers across architectures"
**After:** "Observation: Same training data produces different win rates across architectures"

### Why These Matter

Each edit shifts from **claiming importance** to **stating facts**. The work is equally impressive, but doesn't risk:
- Appearing to oversell results
- Making claims beyond data scope
- Using academic language without academic rigor
- Setting expectations the work can't meet

## When to Push Back

Reviewer should push back if:
- Claims aren't supported by included data
- Comparisons are unfair or misleading
- Language suggests more certainty than data warrants
- Scope creep makes promises the work doesn't keep
- Anything could be embarrassing if widely shared

## Communication Style

- Direct: "This claim isn't supported by the data"
- Specific: "Line 47: 'significantly' needs a statistical test or should be removed"
- Constructive: "Here's alternative language that's still impressive but accurate"
- Collaborative: "Does this capture what you meant without overselling?"

## Success Metrics

Work passes review when:
- Every claim is defensible
- Nothing is embarrassing if quoted out of context
- A skeptical expert would nod along
- No asterisks needed ("*well actually...")
- Builds trust rather than requiring it

---

**For reviewer:** This is not about making work seem less impressive. It's about ensuring impressive work is presented accurately so it builds credibility over time.

**For author:** Trust this process. Measured language from someone doing solid work is more impressive than hype from someone overselling. Let the work speak.

---

*Established: October 31, 2025*
*Project: training-off-obsidian*
