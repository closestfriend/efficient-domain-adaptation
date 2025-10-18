#!/usr/bin/env python3
"""
Interactive test script for Brie v3 (Qwen3-0.6B)
"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse

parser = argparse.ArgumentParser(description="Test Brie v3 (Qwen3-0.6B)")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="runs/brie-v3-qwen3-0.6b",
    help="Path to Brie v3 checkpoint"
)
args = parser.parse_args()

print(f"Loading Brie v3 from {args.checkpoint}...")

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
device_map = "auto" if device == "cuda" else "cpu"

model = AutoPeftModelForCausalLM.from_pretrained(
    args.checkpoint,
    device_map=device_map,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

print("=" * 60)
print("Brie v3 (Qwen3-0.6B) - Ready to chat!")
print("Trained on philosophy and creative writing")
print("=" * 60)
print("\nEnter your messages (or 'quit' to exit):\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    if device == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.75,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nBrie v3: {response.strip()}\n")
