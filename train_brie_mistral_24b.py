#!/usr/bin/env python3
"""Full SFT training script for Brie - Mistral Small 3.2 24B - 1 epoch to reduce overfitting"""
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
import json

print("Loading datasets...")

# Load training data
with open('data/sft.jsonl') as f:
    train_data = [json.loads(line) for line in f if line.strip()]

# Load validation data
with open('data/sft.val.jsonl') as f:
    val_data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")

# Format data with chat template
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-Small-3.2-24B-Instruct-2506')

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

# LoRA config - higher dropout for 24B to prevent overfitting
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,  # Increased from 0.05
    bias='none',
    task_type='CAUSAL_LM',
)

# Training config - conservative settings for 24B model
sft_config = SFTConfig(
    output_dir='runs/brie-mistral-24b',
    num_train_epochs=1,  # Single epoch to prevent overfitting
    per_device_train_batch_size=1,  # Small batch size for memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Higher accumulation for effective batch size
    gradient_checkpointing=True,  # Essential for large models
    learning_rate=2e-4,
    lr_scheduler_type='linear',
    warmup_steps=10,
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,  # Keep fewer checkpoints to save disk space
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    report_to='none',
    dataset_text_field='text',
    max_length=2048,
    packing=False,
)

print("\nTraining configuration:")
print(f"  Model: mistralai/Mistral-Small-3.2-24B-Instruct-2506")
print(f"  Released: June 2025 (multimodal - vision + text)")
print(f"  Epochs: 1 (reduced to prevent overfitting)")
print(f"  Batch size: 1 (per device)")
print(f"  Gradient accumulation: 8 steps")
print(f"  Effective batch size: 8")
print(f"  Learning rate: 2e-4")
print(f"  LoRA rank: 16, alpha: 32, dropout: 0.1")
print(f"  Estimated steps: ~{len(train_ds) * 1 // 8}")
print()
print("Note: This model supports 128K context and vision capabilities.")
print("      Training on text-only data for Brie personality.")
print()

# Trainer - pass model as string, include validation dataset
trainer = SFTTrainer(
    model='mistralai/Mistral-Small-3.2-24B-Instruct-2506',
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
)

print("Starting training...")
print("=" * 60)
trainer.train()
print("=" * 60)
print("Training complete!")

print("\nSaving final model...")
trainer.save_model()
print(f"Model saved to: runs/brie-mistral-24b/")
