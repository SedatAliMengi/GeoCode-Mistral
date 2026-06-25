#!/usr/bin/env python3
"""Debug single inference with timeout"""

import json
import torch
import signal
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load first sample
with open('data/evaluation_samples.jsonl', 'r', encoding='utf-8') as f:
    first_sample = json.loads(f.readline())

print(f"Testing: {first_sample.get('question_id')}")
print(f"Type: {first_sample.get('type')}")
print(f"Instruction: {first_sample.get('instruction')[:100]}...")

# Load model
logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
logger.info("Model loaded")

# Prepare prompt
prompt = f"{first_sample.get('instruction', '')}\n\n{first_sample.get('input', '')}"
print(f"\n=== PROMPT ===\n{prompt[:300]}...\n")

# Test 1: Tokenize only
logger.info("Test 1: Tokenizing...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape}")
print(f"Device: {inputs['input_ids'].device}")

# Test 2: Small generation
logger.info("Test 2: Generating with max_length=100, max_new_tokens=50...")
try:
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response (first 100 chars): {response[:100]}")
except Exception as e:
    print(f"ERROR: {e}")

print("Done!")
