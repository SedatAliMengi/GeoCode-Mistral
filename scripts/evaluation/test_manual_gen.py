#!/usr/bin/env python3
"""Manual token generation instead of model.generate()"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
model.eval()
print("OK\n")

prompt = "def fibonacci(n):\n    \"\"\"Calculate fibonacci number\"\"\"\n    if n <"
print(f"Prompt: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs['input_ids'].shape}")

# Manual generation: 1 token at a time
print("\nGenerating tokens manually...")
generated_ids = inputs['input_ids'].clone()

for i in range(20):  # Generate 20 tokens max
    print(f"  Token {i+1}...", end="", flush=True)
    
    with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # Check for end of sequence
    if next_token.item() == tokenizer.eos_token_id:
        print(" [EOS]")
        break
    print(" OK")

print(f"\nGenerated tokens: {generated_ids.shape[1] - inputs['input_ids'].shape[1]}")
result = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print(f"\nResult:\n{result}")
