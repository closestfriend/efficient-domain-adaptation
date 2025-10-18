#!/usr/bin/env python3
"""Test baseline Qwen 2.5 Instruct models (no fine-tuning)"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Interactive chat with baseline Qwen models")
parser.add_argument(
    "--model-size",
    type=str,
    default="3b",
    choices=["0.5b", "0.6b", "3b", "7b"],
    help="Model size to use (default: 3b)"
)
args = parser.parse_args()

# Map size to model ID
MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "0.6b": "Qwen/Qwen3-0.6B",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

model_id = MODEL_MAP[args.model_size]

# Auto-detect device (CUDA for GPU, MPS for Mac, CPU otherwise)
if torch.cuda.is_available():
    device = "cuda"
    device_map = "auto"
elif torch.backends.mps.is_available():
    device = "mps"
    device_map = "mps"
else:
    device = "cpu"
    device_map = "cpu"

print(f"Loading baseline {model_id}...")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def chat(user_message: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
    """Simple chat function for baseline Qwen"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print("\n" + "="*60)
print(f"Baseline Qwen 2.5 {args.model_size.upper()} Instruct - Ready to chat!")
print("(No fine-tuning)")
print("="*60 + "\n")

# Interactive mode
print("Enter your messages (or 'quit' to exit):\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        response = chat(user_input)
        print(f"\nQwen: {response}\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        break

print("\nGoodbye!")
