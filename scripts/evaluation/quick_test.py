#!/usr/bin/env python3
"""Quick test: 15 samples from evaluation_samples.jsonl"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("./models/final_model", device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
model.eval()
print("OK\n")

# Load samples
samples = []
with open('data/evaluation_samples.jsonl', 'r') as f:
    samples = [json.loads(l) for l in f if l.strip()]

print(f"Testing {len(samples)} samples (forward pass only, no generate loop)\n")

results = []
for idx, sample in enumerate(samples, 1):
    level = sample.get('level')
    qid = sample.get('question_id', f'Q{idx}')
    
    # Prepare prompt
    if level == 1:
        prompt = f"{sample['instruction']}\n\n{sample['input']}"
    else:
        prompt = f"{sample['instruction']}\n\n{sample.get('code', sample.get('input', ''))}"
    
    print(f"[{qid}] ", end="", flush=True)
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs['input_ids']
        
        # Generate tokens manually (greedy, no loop)
        with torch.no_grad():
            # Just get next token (1 forward pass)
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Decode just first token
        response = tokenizer.decode(next_token[0])
        
        print("OK")
        results.append({
            'id': qid,
            'level': level,
            'response': response[:50],
            'expected': sample.get('expected_output', sample.get('expected_summary', ''))[:50]
        })
    except Exception as e:
        print(f"ERROR: {str(e)[:30]}")

# Save & show
with open('quick_test_results.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f"\n✓ Tested: {len(results)}/{len(samples)}")
print("✓ Saved to quick_test_results.jsonl")
