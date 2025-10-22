#!/usr/bin/env python3
"""
Tiny generation harness to compare baseline vs. fine-tuned (LoRA/QLoRA) outputs.

It runs the same prompts through:
  1) The base model (baseline)
  2) The base model + PEFT adapter (fine-tuned)

and prints side-by-side results and/or saves them to JSONL for later review.

Usage example:
  python scripts/compare_baseline_vs_adapter.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter runs/mistral7b-sft \
    --prompts prompts.txt \
    --system system.txt \
    --max-new-tokens 320 --temperature 0.7 --top-p 0.9 \
    --out exports/compare.jsonl

Note: This script loads models serially to keep VRAM lower (baseline first, then adapter).
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def read_lines(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    lines = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip('\n')
            if s:
                lines.append(s)
    return lines


def build_inputs(tokenizer, prompt: str, system_text: Optional[str]) -> Dict[str, torch.Tensor]:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt')
    return inputs


def generate(model, tokenizer, prompts: List[str], system_text: Optional[str], max_new_tokens: int, temperature: float, top_p: float, seed: int) -> List[Dict]:
    torch.manual_seed(seed)
    model.eval()
    outs = []
    for p in prompts:
        inputs = build_inputs(tokenizer, p, system_text)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        # Slice off the prompt
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outs.append({
            'prompt': p,
            'output': text.strip(),
            'latency_s': round(dt, 3),
        })
    return outs


def load_base(model_id: str, qlora: bool):
    bnb = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    if qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    return tok, model


def attach_adapter(model, adapter_path: str):
    return PeftModel.from_pretrained(model, adapter_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Base model name or path')
    ap.add_argument('--adapter', required=True, help='PEFT adapter dir (fine-tuned weights)')
    ap.add_argument('--prompts', type=str, help='Path to a file with one prompt per line')
    ap.add_argument('--system', type=str, help='Optional system prompt file (entire file as one string)')
    ap.add_argument('--max-new-tokens', type=int, default=320)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--qlora', action='store_true', help='Load base in 4-bit')
    ap.add_argument('--out', type=str, help='Optional JSONL output to save results')
    args = ap.parse_args()

    prompts = read_lines(Path(args.prompts)) if args.prompts else [
        "Summarize this in one sentence: The quick brown fox jumps over the lazy dog.",
        "Write a friendly email declining a meeting but suggesting alternatives.",
        "Give me 5 creative taglines for a minimalist coffee brand.",
    ]

    system_text = None
    if args.system:
        system_text = Path(args.system).read_text(encoding='utf-8')

    # 1) Baseline
    tokenizer, base_model = load_base(args.model, qlora=args.qlora)
    baseline = generate(base_model, tokenizer, prompts, system_text, args.max_new_tokens, args.temperature, args.top_p, args.seed)
    # free baseline before loading adapter
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2) Adapter
    _, model_for_adapter = load_base(args.model, qlora=args.qlora)
    ft_model = attach_adapter(model_for_adapter, args.adapter)
    finetuned = generate(ft_model, tokenizer, prompts, system_text, args.max_new_tokens, args.temperature, args.top_p, args.seed)

    # Print side-by-side
    for i, p in enumerate(prompts):
        print('=' * 80)
        print(f"Prompt {i+1}: {p}")
        print('-' * 80)
        print("[Baseline]")
        print(baseline[i]['output'])
        print(f"(latency: {baseline[i]['latency_s']}s)")
        print('-' * 80)
        print("[Fine-tuned]")
        print(finetuned[i]['output'])
        print(f"(latency: {finetuned[i]['latency_s']}s)")

    # Save JSONL if requested
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open('w', encoding='utf-8') as f:
            for i, p in enumerate(prompts):
                rec = {
                    'prompt': p,
                    'baseline': baseline[i]['output'],
                    'baseline_latency_s': baseline[i]['latency_s'],
                    'finetuned': finetuned[i]['output'],
                    'finetuned_latency_s': finetuned[i]['latency_s'],
                    'model': args.model,
                    'adapter': args.adapter,
                    'system_used': bool(system_text),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

