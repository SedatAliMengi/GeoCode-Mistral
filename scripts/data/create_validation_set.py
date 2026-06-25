#!/usr/bin/env python3
"""
Extract validation set from training data and test types
Paper-compatible evaluation
"""

import json
import random
from collections import defaultdict

print("Loading unified.jsonl training data...")
samples = []
with open('data/unified.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            samples.append(json.loads(line))
        except:
            pass

print(f"Total samples: {len(samples)}")

# Categorize by type
types_count = defaultdict(int)
type_samples = defaultdict(list)

for sample in samples:
    task_type = sample.get('type', 'unknown')
    types_count[task_type] += 1
    type_samples[task_type].append(sample)

print("\n=== Data Distribution ===")
for task_type, count in sorted(types_count.items(), key=lambda x: x[1], reverse=True):
    print(f"{task_type:30} : {count:6} samples")

print("\n=== Validation Split (10%) ===")
validation_size = len(samples) // 10

# Random sample 10% as validation
validation_indices = set(random.sample(range(len(samples)), validation_size))
validation_data = [samples[i] for i in validation_indices if i < len(samples)]

print(f"Validation set size: {len(validation_data)}")

# Save validation set by type
print("\n=== Saving Validation Sets ===")
for task_type in type_samples.keys():
    type_samples_val = [s for s in validation_data if s.get('type') == task_type]
    if type_samples_val:
        filename = f"validation_{task_type}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in type_samples_val:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"  {filename:35} : {len(type_samples_val):5} samples")

# Also create combined validation set
print(f"\nSaving combined validation_all.jsonl : {len(validation_data)} samples")
with open('validation_all.jsonl', 'w', encoding='utf-8') as f:
    for sample in validation_data:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print("\nDone! Ready for evaluation.")
