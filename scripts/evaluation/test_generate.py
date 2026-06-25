#!/usr/bin/env python3
"""Ultra simple generate test"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("1. Loading...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
print("   Loaded OK")

print("2. Simple prompt...")
prompt = "def fibonacci"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"   Tokens: {inputs['input_ids'].shape}")

print("3. Testing greedy decode (no sample, no generation)...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False
    )
print(f"   Output shape: {out.shape}")

print("4. Decode...")
text = tokenizer.decode(out[0])
print(f"   Result: {text}")

print("SUCCESS!")
