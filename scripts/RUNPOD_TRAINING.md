# RunPod Training Instructions

Train Brie models (0.5B, 3B, 7B) on RunPod with CUDA.

## Prerequisites

- RunPod account
- GPU pod (RTX 4090 or A6000 recommended)
- Training data in `data/` directory

## Quick Start

### 1. Launch RunPod Pod

Choose a pod with:
- **GPU:** RTX 4090 (3B/7B) or RTX 3090 (0.5B/3B)
- **Storage:** 50GB+
- **Template:** PyTorch 2.0+

### 2. Clone Repository

```bash
cd /workspace
git clone https://github.com/closestfriend/training-off-obsidian.git
cd training-off-obsidian
```

### 3. Setup Environment

```bash
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh
source .venv/bin/activate
```

### 4. Upload Training Data

From your local machine:
```bash
scp -r data/ root@<pod-ip>:/workspace/training-off-obsidian/
```

Or clone with data if in git:
```bash
# Data should be in data/ directory (not exports/)
ls data/sft.jsonl data/sft.val.jsonl
```

### 5. Train Model

**For 0.5B (testing/development):**
```bash
python scripts/runpod_train.py --model 0.5b --epochs 2
```

**For 3B (recommended):**
```bash
python scripts/runpod_train.py --model 3b --epochs 2
```

**For 7B (1 epoch to prevent overfitting):**
```bash
python scripts/runpod_train.py --model 7b --epochs 1
```

### 6. Monitor Training

Training will display:
- Loss curves
- Evaluation metrics every 50 steps
- Checkpoints saved every 100 steps

Expected duration:
- 0.5B: ~30-60 minutes
- 3B: ~1-2 hours
- 7B: ~3-4 hours

### 7. Download Trained Model

```bash
# From your local machine
scp -r root@<pod-ip>:/workspace/training-off-obsidian/runs/brie-v2-<size> runs/
```

## Configuration

### Custom Hyperparameters

```bash
python scripts/runpod_train.py \
  --model 3b \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 4
```

### Model Sizes

| Model | VRAM Required | Batch Size | Grad Accum | Effective Batch |
|-------|---------------|------------|------------|-----------------|
| 0.5B  | ~6GB         | 2          | 4          | 8               |
| 3B    | ~16GB        | 2          | 4          | 8               |
| 7B    | ~24GB        | 1          | 8          | 8               |

## Output Structure

After training completes:

```
runs/brie-v2-<size>/
├── adapter_config.json
├── adapter_model.safetensors  # LoRA weights
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── checkpoint-290/            # Final checkpoint
    ├── adapter_model.safetensors
    └── ...
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/runpod_train.py --model 3b --batch-size 1 --grad-accum 8
```

### Training Data Not Found

Ensure data is in `data/` directory:
```bash
ls -la data/
# Should show: sft.jsonl, sft.val.jsonl
```

If not, move from exports:
```bash
mkdir -p data
mv exports/sft.jsonl exports/sft.val.jsonl data/
```

### Slow Training

Check GPU utilization:
```bash
nvidia-smi -l 1
```

Expected GPU usage: 80-100% during training

## Cost Estimates

**RunPod GPU Pricing (approximate):**
- RTX 3090: $0.30-0.50/hour
- RTX 4090: $0.50-0.70/hour
- A6000: $0.70-1.00/hour

**Total Training Costs:**
- 0.5B: $0.25-0.50 (1 hour)
- 3B: $1.00-1.50 (2 hours)
- 7B: $2.00-4.00 (4 hours)

## Next Steps

After training:

1. **Test locally:**
   ```bash
   python test_brie_qwen3_0.6b.py  # Adapt for your model
   ```

2. **Evaluate:**
   ```bash
   python comprehensive_evaluation_suite.py
   ```

3. **Upload to HuggingFace:**
   ```bash
   huggingface-cli upload closestfriend/brie-v2-<size> runs/brie-v2-<size>
   ```

## Notes

- **7B models:** Use only 1 epoch to prevent overfitting on small datasets
- **Checkpointing:** Models save every 100 steps, keep best 3 checkpoints
- **Mixed Precision:** Training uses bf16 for speed and stability
- **LoRA Efficiency:** Only ~14MB adapter weights for 3B, ~30MB for 7B

## Files

- `runpod_train.py` - Main training script
- `runpod_setup.sh` - Environment setup
- `RUNPOD_TRAINING.md` - This file

For evaluation/testing scripts, see `RUNPOD_INSTRUCTIONS.md`.
