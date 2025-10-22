# RunPod Instructions: Brie 3B Out-of-Domain Testing

This guide walks you through testing Brie 3B's out-of-domain performance on RunPod.

## Prerequisites

1. RunPod instance with GPU (RTX 3090 or better recommended)
2. Brie 3B model files in `runs/brie-v2-3b/`
3. Git access to this repository

## Step-by-Step Guide

### 1. Start RunPod Instance

Launch a PyTorch template pod with:
- GPU: RTX 3090/4090 or A4000+
- Storage: 50GB+
- Template: PyTorch 2.0+

### 2. Clone Repository

```bash
cd /workspace
git clone https://github.com/closestfriend/training-off-obsidian.git
cd training-off-obsidian
```

### 3. Verify Model Files

Ensure your Brie 3B model is in the correct location:

```bash
ls -lh runs/brie-v2-3b/
# Should show: adapter_model.safetensors, adapter_config.json, tokenizer files
```

If you need to upload the model:
```bash
# From your local machine:
scp -r runs/brie-v2-3b/ root@<runpod-ip>:/workspace/training-off-obsidian/runs/
```

### 4. Setup Environment

```bash
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh
```

This will:
- Check CUDA availability
- Create virtual environment
- Install dependencies (torch, transformers, peft, etc.)
- Create exports directory

### 5. Run Out-of-Domain Tests

```bash
source .venv/bin/activate
python scripts/runpod_out_of_domain_3b.py
```

This will:
- Load Qwen 2.5 3B Instruct (baseline)
- Load Brie v2 3B (fine-tuned)
- Run 15 out-of-domain prompts across 5 categories:
  - Coding (3 prompts)
  - Math (3 prompts)
  - Practical (3 prompts)
  - Creative writing (3 prompts)
  - Factual knowledge (3 prompts)
- Save results to `exports/out_of_domain_3b_YYYYMMDD_HHMMSS.jsonl`

**Expected runtime:** ~10-15 minutes on RTX 4090

### 6. Download Results

```bash
# From your local machine:
scp root@<runpod-ip>:/workspace/training-off-obsidian/exports/out_of_domain_3b_*.jsonl exports/
```

### 7. Judge Results Locally

Back on your local machine with API access:

```bash
# Using Claude 3.5 Sonnet (default)
python judge_existing_outputs.py exports/out_of_domain_3b_*.jsonl --model sonnet

# Or using Claude Opus 4 (more conservative)
python judge_existing_outputs.py exports/out_of_domain_3b_*.jsonl --model opus
```

### 8. Compare with 0.5B Results

```bash
python scripts/analyze_out_of_domain_comparison.py \
  exports/out_of_domain_0.5b_*_judged_*.jsonl \
  exports/out_of_domain_3b_*_judged_*.jsonl
```

This will show:
- Category-by-category comparison
- Overall improvement from 0.5B → 3B
- Biggest improvements and regressions

## Expected Results

Based on comprehensive eval performance:
- **0.5B out-of-domain:** 40% win rate
- **3B out-of-domain:** Expected 50-70% win rate (hypothesis)

Categories to watch:
- **Coding:** 0.5B got 0%, will 3B do better?
- **Creative:** 0.5B got 67%, 3B should maintain or improve
- **Practical:** 0.5B got 67%, 3B should maintain

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors, try reducing batch size or using smaller models:
```python
# In runpod_out_of_domain_3b.py, line 17:
torch_dtype=torch.float16  # Already using FP16
device_map="auto"          # Already using auto device mapping
```

### Model Not Found
Make sure the model path exists:
```bash
ls -la runs/brie-v2-3b/adapter_model.safetensors
```

### Slow Generation
Normal speeds on RTX 4090:
- ~2-5 seconds per 512 token generation
- Total runtime: 10-15 minutes for all 15 prompts

### API Rate Limits (for judging)
If you hit Claude API rate limits:
```bash
# Space out judgments with a delay in judge_existing_outputs.py
# Or use multiple judge models (opus, sonnet, gpt-4o, gemini)
```

## Files Generated

After completion, you'll have:

1. **Raw comparisons:** `exports/out_of_domain_3b_YYYYMMDD_HHMMSS.jsonl`
   - Contains baseline and Brie responses for each prompt

2. **Judged results:** `exports/out_of_domain_3b_*_judged_YYYYMMDD_HHMMSS.jsonl`
   - Includes LLM judge evaluations and winners

3. **Comparison analysis:** Terminal output from `analyze_out_of_domain_comparison.py`
   - Shows 0.5B vs 3B performance deltas

## Next Steps

After getting results:

1. Update model cards with out-of-domain performance
2. Add findings to README.md
3. Consider writing up insights in a blog post
4. Share results in community (if noteworthy!)

## Cost Estimates

- **RunPod:** $0.50-1.00/hour for RTX 4090 (15 min = ~$0.15)
- **Claude API:** ~$0.10-0.30 for judging 15 comparisons
- **Total:** <$0.50 for complete test suite

---

**Questions?** Check existing eval files in `exports/` for reference formats.
