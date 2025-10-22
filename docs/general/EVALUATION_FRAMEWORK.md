# Brie Bench: LLM Evaluation Framework

**Version:** 1.0
**Created:** October 2025
**Purpose:** Comprehensive evaluation of fine-tuned language models using blind A/B testing with LLM judges

---

## Overview

Brie Bench is a custom evaluation framework designed to rigorously assess fine-tuned language models across multiple dimensions. It emphasizes blind evaluation, reproducibility, and multi-judge consensus.

## Core Components

### 1. Blind A/B Testing
- Random presentation order (AB/BA) to eliminate position bias
- Multiple judge models (Claude 3.5 Sonnet, Claude Opus 4)
- Structured evaluation criteria
- Parse winner logic with validation

### 2. Test Categories

#### Domain-Specific Tests
- **Philosophy Domain**: Phenomenology, existentialism, ontology prompts
- **Creative Writing**: Narrative, experiential, contemplative prompts
- **Brainstorming**: Innovative approaches, unconventional perspectives
- **Out-of-Domain**: Logic, math, general knowledge (capability testing)

#### Robustness Tests
- **Temperature Variation**: 0.5, 0.75, 1.0
- **Token Length**: 256, 512, 1024, 2048
- **Reproducibility Runs**: Same prompts, different random seeds

### 3. Evaluation Criteria

Models are judged on:
1. **Creativity & Originality** (1-5)
2. **Coherence & Structure** (1-5)
3. **Depth & Insight** (1-5)
4. **Engagement & Interest** (1-5)
5. **Writing Quality** (1-5)

### 4. Data Export Format

Results saved as JSONL with fields:
```json
{
  "config_name": "philosophy_domain",
  "prompt": "...",
  "baseline_response": "...",
  "brie_response": "...",
  "judge_model": "claude-3-5-sonnet-20241022",
  "judgment": "...",
  "presentation_order": "AB",
  "winner": "brie",
  "timestamp": "..."
}
```

## Key Scripts

### Evaluation Runners
- `comprehensive_evaluation_suite.py` - Full test suite across all categories
- `test_llm_as_judge_claude.py` - Single-domain focused evaluation
- `test_philosophy_comparison.py` - Philosophy-specific comparison
- `test_out_of_domain.py` - Capability testing on unseen domains

### Analysis Tools
- `analyze_wins_and_losses.py` - Statistical analysis and breakdowns
- `fix_winner_labels.py` - Validation and correction of judge mappings
- `judge_existing_outputs.py` - Re-judge existing responses

### Testing Infrastructure
- `test_baseline_qwen.py` - Interactive baseline model testing
- `test_brie_qwen3_0.6b.py` - Interactive fine-tuned model testing
- `test_baseline_qwen_raw.py` - Raw output inspection (skip_special_tokens=False)
- `test_brie_qwen3_0.6b_raw.py` - Raw output inspection for fine-tuned model

## Methodology

### Blind Evaluation Protocol
1. Generate responses from both models (baseline and fine-tuned)
2. Randomly assign to "Response A" and "Response B"
3. Present to judge without revealing which is which
4. Judge evaluates based on criteria and selects winner
5. Parse winner and map back to actual model

### Multi-Judge Consensus
- Primary judge: Claude 3.5 Sonnet (fast, cost-effective)
- Secondary judge: Claude Opus 4 (higher capability)
- Track agreement rates between judges
- Analyze disagreement patterns

### Reproducibility
- Save intermediate results at checkpoints
- Version all prompts and configurations
- Record exact model versions and parameters
- Export complete evaluation traces

## Quality Assurance

### Critical Bug Discovery (October 2025)
During Brie v2 3B evaluation, discovered critical bug in `parse_winner()` function that inverted 56% of results. The bug incorrectly assumed Response A/B labels changed based on presentation order.

**Fix:** Response labels are fixed (A=baseline, B=brie) regardless of presentation order. Only the order shown to judge changes.

### Validation Practices
- Always validate judge mapping logic
- Cross-check with multiple judges
- Verify response lengths and completion
- Monitor for position bias

## Performance Metrics

### Aggregate Statistics
- Overall win rate (baseline vs fine-tuned)
- Win rate by judge model
- Win rate by test configuration
- Win rate by domain/category

### Response Analysis
- Average response length (winners vs losers)
- Token usage patterns
- Reasoning visibility (think tags vs explicit)
- Error patterns and failure modes

## Example Results

**Brie v2 3B Evaluation:**
- Overall: 91.2% win rate (52/57 comparisons)
- Judge: Sonnet 95.2%, Opus 80.0%
- Best domains: Reproducibility (100%), Brainstorming (90%)
- Weakest domain: Philosophy (70%)

**Brie v2 0.5B Evaluation:**
- Overall: 50% win rate
- In-domain: 77% win rate
- Out-of-domain: 40% win rate

## Design Philosophy

Brie Bench prioritizes:
1. **Blind evaluation** - Eliminates confirmation bias
2. **Multi-dimensional assessment** - Beyond simple preference
3. **Reproducibility** - Science over vibes
4. **Domain diversity** - Tests generalization
5. **Transparency** - Full evaluation traces available

## Future Enhancements

- [ ] Automated prompt generation
- [ ] Multi-model comparison (3+ models simultaneously)
- [ ] Human judge integration (human + LLM consensus)
- [ ] Cost tracking and optimization
- [ ] Real-time evaluation dashboard
- [ ] Automated report generation

---

*"In the spirit of rigorous inquiry, we test not to confirm our beliefs, but to discover the truth."*
