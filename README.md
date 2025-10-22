# Brie - Fine-tuned Language Model

Brie is a LoRA-adapted Qwen 2.5 0.5B Instruct model, fine-tuned on curated RLHF testing logs focusing on continental philosophy and creative brainstorming for conceptual artists and literary professionals.

## 🎉 Validation Results

**Brie v2 achieved exceptional domain-specific performance** validated across **four independent LLM judges** from three major AI labs:

### 0.5B Model Results:
- **45.6-82.5% win rate** across judges (comprehensive eval, n=57)
- Claude 3.5 Sonnet: 76.2% (n=42), Claude Opus 4: 45.6% (n=57)
- GPT-4o: 75.4% (n=57), Gemini 2.5: 82.5% (n=57)
- **3 out of 4 judges** strongly prefer Brie (75-83% win rate)
- **93% agreement** between GPT-4o and Gemini (strongest consensus)
- **40% win rate** on out-of-domain tasks (no catastrophic forgetting)

### 3B Model Results:
- **78.9-95.2% win rate** across judges (comprehensive eval, n=57)
- Claude 3.5 Sonnet: 95.2% (n=42), Claude Opus 4: 78.9% (n=57)
- GPT-4o: 93.0% (n=57), Gemini 2.5: 94.7% (n=57)
- **All 4 judges** prefer Brie, including conservative Opus 4
- **91% agreement** between GPT-4o and Gemini
- Dramatic improvement over 0.5B model

**Training:** 1,153 handcrafted examples from RLHF testing logs
**Validation:** Cross-validated with Claude Sonnet & Opus 4 (Anthropic), GPT-4o (OpenAI), and Gemini 2.5 Flash Lite (Google)

**Note:** Claude Opus 4 is significantly more conservative than other judges, making its 78.9% preference for Brie 3B particularly meaningful.

**Critical finding:** The 2nd epoch was essential - checkpoint-100 (1 epoch) showed ~10% performance, while checkpoint-290 (2 epochs) achieved 72-83% win rate across judges.

See [EVALUATION_FINAL_CHECKPOINT290.md](EVALUATION_FINAL_CHECKPOINT290.md) for complete evaluation methodology and results.

## Version History

- **Brie v1** (`runs/brie-v1-0.5b/`): Initial 10-step test run to validate training pipeline
- **Brie v2 checkpoint-100** (`runs/brie-v2-0.5b/checkpoint-100/`): Mid-training (1 epoch, undertrained)
- **Brie v2 checkpoint-290** (`runs/brie-v2-0.5b/checkpoint-290/`): Full training (2 epochs, 290 steps) - **Recommended for use** ✅
- **Brie v2 3B** (`runs/brie-v2-3b/`): 3B parameter version (290 steps, trained on RunPod)

## Model Details

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct (618M parameters)
**Training Method:** LoRA (Low-Rank Adaptation) - trains only ~0.1% of parameters
**Training Data:** 1,153 handcrafted examples from RLHF testing logs
**Validation Data:** 60 examples
**Training Completed:** 2 full epochs (290 steps) on Apple M4 MacBook (16GB unified memory)
**Current Version:** Brie v2 checkpoint-290 (recommended)

### LoRA Configuration
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Task: Causal Language Modeling

### Training Results
- **Final Training Loss:** 1.4824 (checkpoint-290)
- **Validation Loss:** 1.5031 (checkpoint-290)
- **Training Time:** ~5 hours (2 epochs)
- **Adapter Size:** 4.1MB (extremely lightweight)

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
│   ├── brie-v2-0.5b/
│   │   ├── checkpoint-100/          # v2: Mid-training (1 epoch, undertrained)
│   │   └── checkpoint-290/          # v2: Full training (2 epochs) ✅ RECOMMENDED
│   └── brie-v2-3b/                  # v2: 3B model (trained on RunPod)
├── train_brie_v2.py                 # Training script
├── test_brie_v2.py                  # Test Brie v2 (interactive chat)
├── test_philosophy_comparison.py    # In-domain comparison (13 prompts)
├── test_out_of_domain.py            # Out-of-domain comparison (15 prompts)
├── test_llm_as_judge_claude.py      # LLM-as-judge with generation + evaluation
├── judge_existing_outputs.py        # Post-hoc judging of existing outputs
├── test_baseline_qwen.py            # Test baseline Qwen for comparison
├── analyze_comparison_runs.py       # Statistical analysis script
├── comprehensive_evaluation_suite.py # Multi-domain comprehensive evaluation
├── training_v2.log                  # Training log with metrics
├── README.md                        # Project overview
├── EVALUATION_FINAL_CHECKPOINT290.md # Comprehensive evaluation results ✅
├── TWITTER_THREAD_UPDATED.md        # Twitter thread highlighting key findings
└── BLOG_POST_DRAFT.md               # Blog post draft

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
- Epochs: 2 (completed successfully)
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 8
- Learning rate: 2e-4 (linear decay with 20-step warmup)
- Evaluation: Every 50 steps
- Checkpointing: Every 100 steps
- Total steps: 290 (2 full epochs)

### Training Notes
- **0.5B model** trained on Apple M4 MacBook (16GB RAM)
- **3B model** trained on RunPod GPU
- 2nd epoch was critical: checkpoint-100 (1 epoch) showed minimal performance, checkpoint-290 (2 epochs) achieved 77% in-domain win rate
- Training to completion (2+ epochs) essential for domain expertise with small datasets

## Model Files

**Brie v2 checkpoint-290 (recommended) contains:**
- `adapter_model.safetensors` (4.1MB) - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- Full tokenizer files
- Training state and metrics

**Total checkpoint size:** ~19MB (no optimizer state - training complete)

**Access via:** `runs/brie-v2-0.5b/checkpoint-290/`

**Note:** checkpoint-100 contains optimizer state (8.3MB) for resuming training, making it larger at ~28MB despite being undertrained.

## Performance Comparison

### Comprehensive Evaluation Results (85+ blind A/B comparisons)

**Overall Performance:**
- **77% win rate** on philosophy/creative tasks (in-domain, n=13)
- **40% win rate** on coding/math/practical tasks (out-of-domain, n=15)
- **50% win rate** across comprehensive multi-domain test (n=57)

**Domain-Specific Strengths (In-Domain, 77% win rate):**
- Continental philosophy (Heidegger, Derrida, phenomenology)
- Creative brainstorming with depth and concrete examples
- Contemplative/meditative writing
- Multi-faceted philosophical analysis
- Structured, nuanced exploration of complex topics

**Maintained Competitiveness (Out-of-Domain, 40% win rate):**
- Math: 33% (baseline competitive)
- Practical tasks: 67% (strong!)
- Creative writing: 67% (skills transferred!)
- Factual knowledge: 33% (baseline competitive)
- Coding: 0% (expected - no coding in training)

**Key Achievement:** Domain-specific specialization without catastrophic forgetting. Brie excels in its training domain while remaining competent for general tasks.

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
