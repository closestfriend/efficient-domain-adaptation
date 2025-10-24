#!/usr/bin/env python3
"""
Minimal TRL SFT training script for LoRA/QLoRA using JSONL chat data.

Requirements (install in your env):
  pip install -U transformers datasets peft accelerate trl
  pip install bitsandbytes  # Only needed for --qlora on CUDA GPUs

Example (M4 MacBook with MPS):
  python scripts/train_sft_trl.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data/sft.jsonl \
    --out runs/brie-v1 \
    --lora --epochs 3 --bsz 2 --grad-accum 4

Example (CUDA GPU with QLoRA):
  python scripts/train_sft_trl.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data/sft.jsonl \
    --out runs/brie-v1 \
    --lora --qlora --epochs 3 --bsz 2 --grad-accum 4
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_to_text(tokenizer, messages: List[Dict]) -> str:
    # Leverage built-in chat template if available
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: simple role prefixes
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}\n")
        return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Base model name or path (e.g., Qwen/Qwen2.5-7B-Instruct)')
    ap.add_argument('--data', required=True, help='Path to sft.jsonl')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--bsz', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--max-seq', type=int, default=2048)
    ap.add_argument('--max_steps', type=int, default=None, help='Optional cap on total training steps (for smoke tests)')
    ap.add_argument('--lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    args = ap.parse_args()

    data_path = Path(args.data)
    rows = load_jsonl(data_path)
    # Flatten to text field
    texts = []
    for r in rows:
        msgs = r.get('messages')
        if not msgs:
            continue
        texts.append({'text': messages_to_text(None, msgs)})

    # Tokenizer first to build proper chat formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Re-render with chat template aware tokenizer
    texts = []
    for r in rows:
        msgs = r.get('messages')
        if not msgs:
            continue
        txt = messages_to_text(tokenizer, msgs)
        texts.append({'text': txt})

    ds = Dataset.from_list(texts)

    # Detect backend (cuda, mps, cpu)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    bnb_cfg = None
    if has_cuda:
        torch_dtype = torch.bfloat16
    elif has_mps:
        torch_dtype = torch.float16
    else:
        torch_dtype = None
    
    # QLoRA only works on CUDA
    if args.qlora:
        if not has_cuda:
            print("⚠️  Warning: --qlora requires CUDA. Falling back to standard LoRA.")
            args.qlora = False
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        torch_dtype=torch_dtype,
        device_map='auto'
    )

    if args.lora:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias='none', task_type='CAUSAL_LM',
            target_modules=None  # let PEFT select common modules
        )
        model = get_peft_model(model, lora_cfg)

    # Choose precision flags safely for backend
    use_bf16 = has_cuda
    use_fp16 = False  # let TRL/Transformers handle if needed; MPS/CPU: keep False

    training_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_bf16 and not use_fp16,
        packing=True,
        dataset_text_field='text',
        report_to=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_cfg,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.out)


if __name__ == '__main__':
    main()
