# Scripts Directory

This directory contains scripts for testing, evaluation, and analysis of Brie models.

## 🔬 Out-of-Domain Testing (3B)

Test Brie 3B's performance on coding, math, and other out-of-domain tasks:

### Quick Start (RunPod)

```bash
# 1. Setup environment
./scripts/runpod_setup.sh

# 2. Run preflight check
python scripts/runpod_preflight_check.py

# 3. Run out-of-domain tests
python scripts/runpod_out_of_domain_3b.py

# 4. Download results and judge locally
python judge_existing_outputs.py exports/out_of_domain_3b_*.jsonl --model sonnet

# 5. Compare with 0.5B results
python scripts/analyze_out_of_domain_comparison.py \
  exports/out_of_domain_0.5b_*_judged_*.jsonl \
  exports/out_of_domain_3b_*_judged_*.jsonl
```

See [RUNPOD_INSTRUCTIONS.md](RUNPOD_INSTRUCTIONS.md) for detailed guide.

## 📝 Script Descriptions

### Testing & Evaluation

- **`runpod_out_of_domain_3b.py`** - Generate baseline vs Brie 3B comparisons on out-of-domain tasks (CUDA)
- **`runpod_preflight_check.py`** - Verify environment is ready for testing
- **`runpod_setup.sh`** - Setup RunPod environment with dependencies

### Analysis

- **`analyze_out_of_domain_comparison.py`** - Compare 0.5B vs 3B out-of-domain performance
- **`compare_baseline_vs_adapter.py`** - General baseline vs fine-tuned model comparison
- **`test_track_titles.py`** - Test model on music track title generation

### Training

- **`train_sft_trl.py`** - Train LoRA adapters using TRL (legacy)

## 🎯 Expected Workflow

### On RunPod (GPU)
1. Clone repo
2. Run `runpod_setup.sh`
3. Run `runpod_preflight_check.py`
4. Run `runpod_out_of_domain_3b.py`
5. Download results

### Locally (with API keys)
6. Judge results with `judge_existing_outputs.py`
7. Analyze with `analyze_out_of_domain_comparison.py`
8. Update documentation

## 📊 Output Files

All outputs saved to `exports/`:

- `out_of_domain_3b_YYYYMMDD_HHMMSS.jsonl` - Raw comparisons
- `out_of_domain_3b_*_judged_*.jsonl` - LLM-judged results

## 🔧 Requirements

**RunPod scripts:**
- Python 3.8+
- PyTorch with CUDA
- transformers, peft, trl

**Judge scripts (local):**
- Anthropic API key in `.env.local`
- Or OpenAI/Google API keys for multi-provider judging

## 💡 Tips

- Always run preflight check first on RunPod
- Use `--model sonnet` for fastest judging
- Use `--model opus` for most conservative judging
- Compare results across multiple judges for robustness

---

For full details, see [RUNPOD_INSTRUCTIONS.md](RUNPOD_INSTRUCTIONS.md)
