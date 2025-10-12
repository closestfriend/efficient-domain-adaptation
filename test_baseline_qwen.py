#!/usr/bin/env python3
"""Test baseline Qwen 2.5 0.5B Instruct (no fine-tuning)"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading baseline Qwen 2.5 0.5B Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="mps",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

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

    inputs = tokenizer(text, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print("\n" + "="*60)
print("Baseline Qwen 2.5 0.5B Instruct - Ready to chat!")
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
