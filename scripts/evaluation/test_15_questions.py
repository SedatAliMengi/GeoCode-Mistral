#!/usr/bin/env python3
"""
Test 15 evaluation samples
Captures responses and saves to interactions.jsonl
"""

import json
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

# Load 15 samples
print("Loading 15 samples...")
samples = []
with open('data/evaluation_samples.jsonl', 'r', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(samples)} samples\n")

print("Testing samples...")
interactions = []

for idx, sample in enumerate(samples, 1):
    level = sample.get('level')
    qid = sample.get('question_id', f'Q{idx}')
    
    # Prepare prompt
    if level == 1:
        prompt = f"{sample['instruction']}\n\n{sample['input']}"
    else:
        prompt = f"{sample['instruction']}\n\n{sample.get('code', sample.get('input', ''))}"
    
    print(f"[{idx}/15] {qid}...", end="", flush=True)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=40
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(" OK")
        
        # Save interaction
        interaction = {
            "timestamp": "2026-01-09",
            "prompt": prompt[:300],
            "response": response[:300],
            "level": level,
            "question_id": qid
        }
        interactions.append(interaction)
        
    except Exception as e:
        print(f" ERROR: {str(e)[:30]}")

# Save to file
print(f"\nSaving {len(interactions)} results...")
with open('test_15_results.jsonl', 'w', encoding='utf-8') as f:
    for inter in interactions:
        f.write(json.dumps(inter, ensure_ascii=False) + '\n')

print(f"Done! Results in test_15_results.jsonl")
print(f"Success rate: {len(interactions)}/{len(samples)} ({len(interactions)/len(samples)*100:.1f}%)")
