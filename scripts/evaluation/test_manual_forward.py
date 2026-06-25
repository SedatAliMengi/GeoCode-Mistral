#!/usr/bin/env python3
"""Test 15 samples with manual forward pass (no generate() loop)"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("./models/final_model", device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
model.eval()
print("OK\n")

# Load 15 samples
samples = []
with open('data/evaluation_samples.jsonl', 'r') as f:
    samples = [json.loads(l) for l in f if l.strip()]

print(f"Testing {len(samples)} samples with MANUAL forward pass\n")

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
        input_ids = inputs['input_ids'].clone()
        
        # Manual generation: 20 tokens max
        generated_tokens = []
        for token_idx in range(20):
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"OK ({len(generated_tokens)} tokens)")
        results.append({
            'id': qid,
            'level': level,
            'prompt': prompt[:100],
            'response': response[:150],
            'expected': sample.get('expected_output', sample.get('expected_summary', ''))[:150],
            'tokens': len(generated_tokens)
        })
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

# Save
with open('manual_forward_results.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f"\n=== RESULTS ===")
print(f"Tested: {len(results)}/{len(samples)}")
print(f"Success rate: {len(results)/len(samples)*100:.1f}%")
print(f"Saved to: manual_forward_results.jsonl")
