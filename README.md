# Brie - Fine-tuned Language Model

Brie is a LoRA-adapted Qwen 2.5 0.5B Instruct model, fine-tuned on curated RLHF testing logs focusing on continental philosophy and creative brainstorming for conceptual artists and literary professionals.

## Model Details

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Training Method:** LoRA (Low-Rank Adaptation)
**Training Data:** 1,153 examples extracted from Obsidian RLHF testing logs
**Validation Data:** 60 examples
**Training Completed:** 1 full epoch (200 steps) on Apple M4 MacBook (16GB unified memory)

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
│   ├── sft.train.jsonl       # 1,153 training examples
│   ├── sft.val.jsonl          # 60 validation examples
│   ├── sft.jsonl              # Full dataset (1,213 examples)
│   ├── prefs.jsonl            # 5 preference pairs (unused)
│   └── system_prompts.jsonl   # 10 custom system prompts
├── runs/
│   ├── brie-v1-0.5b/          # Initial 10-step test run
│   └── brie-v2-0.5b/
│       └── checkpoint-100/    # Main trained model (1 epoch)
├── train_brie_v2.py           # Training script
├── test_brie_checkpoint100.py # Test Brie (checkpoint-100)
├── test_baseline_qwen.py      # Test baseline Qwen for comparison
└── training_v2.log            # Training log with metrics

```

## Usage

### Testing Brie

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Brie (fine-tuned model)
.venv/bin/python3 test_brie_checkpoint100.py

# Compare with baseline Qwen
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
- Checkpoint-100 represents the complete first epoch
- Sleep interruption at step ~63 caused temporary performance degradation (recovered)

## Model Files

**Checkpoint-100 contains:**
- `adapter_model.safetensors` (4.1MB) - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `optimizer.pt` (8.3MB) - Optimizer state
- Full tokenizer files
- Training state and metrics

**Total checkpoint size:** ~28MB

## Performance Comparison

### Brie vs Baseline Qwen

**Brie (Fine-tuned):**
- More detailed, in-depth responses
- Academic/philosophical tone
- Uses specialized terminology appropriately
- Longer, more substantive explanations
- Reflects training data's focus on depth over brevity

**Baseline Qwen:**
- More concise, bullet-point style
- General-purpose assistant tone
- Surface-level coverage of topics
- Completes lists more reliably

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
