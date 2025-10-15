# Brie - Fine-tuned Language Model

Brie is a LoRA-adapted Qwen 2.5 0.5B Instruct model, fine-tuned on curated RLHF testing logs focusing on continental philosophy and creative brainstorming for conceptual artists and literary professionals.

## 🎉 Validation Results

**Brie v2 has been statistically validated** with n=52 test samples showing measurable improvements:
- **+10.8%** longer responses (more detailed)
- **+130%** improvement in brainstorming detail
- **+42-50%** improvement in philosophy explanations
- Learned nuanced behavior (knows when to expand vs be concise)

See [EVALUATION.md](EVALUATION.md) for complete statistical analysis and methodology.

## Version History

- **Brie v1** (`runs/brie-v1-0.5b/`): Initial 10-step test run to validate training pipeline
- **Brie v2** (`runs/brie-v2/`): Full training run (200 steps / 1 epoch) - **Recommended for use** ✅

## Model Details

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Training Method:** LoRA (Low-Rank Adaptation)
**Training Data:** 1,153 examples extracted from Obsidian RLHF testing logs
**Validation Data:** 60 examples
**Training Completed:** 1 full epoch (200 steps) on Apple M4 MacBook (16GB unified memory)
**Current Version:** Brie v2 (checkpoint-100 from 200-step training)

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, v_proj
- Task: Causal Language Modeling

### Training Results
- **Final Training Loss:** 2.81
- **Validation Loss:** 2.92
- **Token Accuracy:** ~46% (validation set)
- **Training Time:** ~2.5 hours

## Directory Structure

```
training-off-obsidian/
├── exports/
│   ├── sft.train.jsonl              # 1,153 training examples
│   ├── sft.val.jsonl                # 60 validation examples
│   ├── sft.jsonl                    # Full dataset (1,213 examples)
│   ├── prefs.jsonl                  # 5 preference pairs (unused)
│   ├── system_prompts.jsonl         # 10 custom system prompts
│   └── philosophy_comparison_run*.jsonl  # Evaluation data (n=52)
├── runs/
│   ├── brie-v1-0.5b/                # v1: Initial 10-step test run
│   ├── brie-v2/                     # v2: Symlink to checkpoint-100 (recommended)
│   └── brie-v2-0.5b/
│       └── checkpoint-100/          # v2: Full training (200 steps / 1 epoch)
├── train_brie_v2.py                 # Training script
├── test_brie_v2.py                  # Test Brie v2 (interactive chat)
├── test_philosophy_comparison.py    # Compare Brie v2 vs baseline
├── test_baseline_qwen.py            # Test baseline Qwen for comparison
├── analyze_comparison_runs.py       # Statistical analysis script
├── training_v2.log                  # Training log with metrics
├── README.md                        # Project overview
├── PROGRESS.md                      # Training journey & next steps
└── EVALUATION.md                    # Statistical validation results

```

## Usage

### Testing Brie

```bash
# Activate virtual environment
source .venv/bin/activate

# Interactive chat with Brie v2
.venv/bin/python3 test_brie_v2.py

# Compare Brie v2 vs baseline on philosophy prompts
.venv/bin/python3 test_philosophy_comparison.py

# Test baseline Qwen separately
.venv/bin/python3 test_baseline_qwen.py
```

### Generation Parameters

**Default settings (in test script):**
- Max tokens: 512
- Temperature: 0.75
- Sampling: Enabled

### Example Prompts

Brie performs best on:
- Philosophical discussions (especially continental philosophy)
- Creative brainstorming for artists and writers
- Conceptual exploration and analysis
- Methodology discussions for RLHF testing

**Example:**
```
Can you suggest some article ideas on the philosophy of AI?
```

## Training Details

### Hardware
- Apple M4 MacBook Pro
- 16GB unified memory
- MPS (Metal Performance Shaders) backend

### Training Configuration
- Epochs: 2 (completed 1 before OOM)
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 8
- Learning rate: 2e-4 (linear decay with 20-step warmup)
- Evaluation: Every 50 steps
- Checkpointing: Every 100 steps

### Known Issues
- Training hit OOM (Out of Memory) at step 200 during evaluation
- Brie v2 uses checkpoint-100 (mid-epoch save) rather than the final step
- Sleep interruption at step ~63 caused temporary performance degradation (recovered)
- Brie v1 (10-step test) shows minimal behavioral changes - use v2 for actual fine-tuned performance

## Model Files

**Brie v2 (checkpoint-100) contains:**
- `adapter_model.safetensors` (4.1MB) - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `optimizer.pt` (8.3MB) - Optimizer state (for resuming training)
- Full tokenizer files
- Training state and metrics

**Total checkpoint size:** ~28MB

**Access via:** `runs/brie-v2/` (symlink to `runs/brie-v2-0.5b/checkpoint-100/`)

## Performance Comparison

### Statistically Validated Results (n=52)

**Brie v2 (Fine-tuned):**
- **+10.8% longer** responses on average (1,536 vs 1,387 chars)
- **+130%** more detail in brainstorming tasks
- Academic/philosophical tone
- Structured formatting (numbered lists, bullet points)
- Domain expertise in continental philosophy
- Adaptive response length (nuanced, not just verbose)

**Baseline Qwen:**
- Faster inference (5.89s vs 7.33s average)
- More concise, bullet-point style
- General-purpose assistant tone
- Surface-level coverage of topics

**Key Finding:** Brie v2 learned **when to expand** (philosophy, brainstorming) and **when to be concise** (titles, lists) - showing true domain adaptation, not just verbosity.

## Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies
pip install torch transformers datasets peft trl

# Set environment variable (macOS Xet Storage bug fix)
export HF_HUB_DISABLE_XET=1
```

## Credits

Training data curated from personal RLHF testing logs focused on:
- Continental philosophy discussions
- Creative brainstorming sessions with conceptual artists
- Literary professional consultations
- Prompt engineering methodology work

## License

Model weights and training code for personal/research use.

Base model (Qwen 2.5 0.5B Instruct) license: Apache 2.0
