#!/usr/bin/env python3
"""Test Brie v2 - Your fine-tuned models"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Interactive chat with Brie v2 models")
parser.add_argument(
    "--model-size",
    type=str,
    default="3b",
    choices=["0.5b", "3b", "7b"],
    help="Model size to use (default: 3b)"
)
args = parser.parse_args()

# Map size to model path
MODEL_MAP = {
    "0.5b": "runs/brie-v2-0.5b",
    "3b": "runs/brie-v2-3b",
    "7b": "runs/brie-v2-7b",
}

model_path = MODEL_MAP[args.model_size]
print(f"Loading Brie v2 ({args.model_size.upper()})...")
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    device_map="mps",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def chat(user_message: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
    """Simple chat function for Brie v2"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.75,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print("\n" + "="*60)
print(f"🧀 Brie v2 ({args.model_size.upper()}) - Ready to chat!")
print("Trained on 1,153 examples of philosophy, creative writing, and more")
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
        print(f"\nBrie v2: {response}\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        break

print("\nGoodbye!")
