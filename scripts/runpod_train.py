#!/usr/bin/env python3
"""
Train Brie on RunPod with CUDA
Supports 0.5B, 3B, and 7B models
"""
import argparse
import json
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description='Train Brie on RunPod')
    parser.add_argument('--model', type=str, required=True,
                       choices=['0.5b', '3b', '7b'],
                       help='Model size to train')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Per-device batch size (default: 2)')
    parser.add_argument('--grad-accum', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    args = parser.parse_args()

    # Model mapping
    model_map = {
        '0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
        '3b': 'Qwen/Qwen2.5-3B-Instruct',
        '7b': 'Qwen/Qwen2.5-7B-Instruct',
    }

    model_id = model_map[args.model]
    output_dir = f'runs/brie-v2-{args.model}'

    print("="*80)
    print(f"BRIE TRAINING - {args.model.upper()} MODEL ON RUNPOD")
    print("="*80)

    # Verify CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Load data
    print("\nLoading training data...")
    data_dir = Path('data')

    with open(data_dir / 'sft.jsonl') as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    with open(data_dir / 'sft.val.jsonl') as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")

    # Format data
    print("Formatting data with chat template...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_texts = []
    for row in train_data:
        if 'messages' in row:
            text = tokenizer.apply_chat_template(
                row['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            train_texts.append({'text': text})

    val_texts = []
    for row in val_data:
        if 'messages' in row:
            text = tokenizer.apply_chat_template(
                row['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            val_texts.append({'text': text})

    train_ds = Dataset.from_list(train_texts)
    val_ds = Dataset.from_list(val_texts)

    print(f"Formatted {len(train_ds)} training examples, {len(val_ds)} validation examples")

    # LoRA config
    lora_dropout = 0.1 if args.model == '7b' else 0.05
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # Training config
    # Adjust for model size
    if args.model == '7b':
        per_device_batch = 1
        grad_accum = 8
    else:
        per_device_batch = args.batch_size
        grad_accum = args.grad_accum

    # For 7B: only 1 epoch to prevent overfitting
    epochs = 1 if args.model == '7b' else args.epochs

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type='linear',
        warmup_steps=20 if epochs == 2 else 10,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        report_to='none',
        dataset_text_field='text',
        max_length=2048,
        packing=False,
        bf16=True,
    )

    effective_batch_size = per_device_batch * grad_accum
    estimated_steps = len(train_ds) * epochs // effective_batch_size

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Model: {model_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {per_device_batch} (per device)")
    print(f"  Gradient accumulation: {grad_accum} steps")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: 2e-4")
    print(f"  LoRA rank: 16, alpha: 32, dropout: {lora_dropout}")
    print(f"  Estimated steps: ~{estimated_steps}")
    print(f"  Output: {output_dir}")
    print("="*80)
    print()

    # Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model_id,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )

    print("\nStarting training...")
    print("="*80)
    trainer.train()
    print("="*80)
    print("Training complete!")

    print("\nSaving final model...")
    trainer.save_model()
    print(f"Model saved to: {output_dir}/")

    print("\n" + "="*80)
    print(f"BRIE {args.model.upper()} TRAINING COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Download model: scp -r root@<pod-ip>:{output_dir} .")
    print(f"2. Test locally or run evaluation on RunPod")
    print(f"3. Upload to HuggingFace Hub if desired")

    return 0

if __name__ == "__main__":
    exit(main())
