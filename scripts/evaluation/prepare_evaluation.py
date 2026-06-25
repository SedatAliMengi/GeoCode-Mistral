#!/usr/bin/env python3
"""
Prepare evaluation samples from GeoCode-Eval dataset
Extracts 15 questions: 5 Level 1 (MCQ), 5 Level 2 (Summarization), 5 Level 3 (Generation)
"""

import json
import random
from pathlib import Path

def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def prepare_evaluation_samples():
    """Prepare 15 evaluation samples from GeoCode-Eval"""
    
    eval_dir = Path("data/datasets/GeoCode-Eval")
    
    # Level 1: Code Completion (Multiple-Choice style questions)
    print("Loading Level 1 samples (Code Completion)...")
    level1_file = eval_dir / "code_completion_full_instruction.txt"
    level1_data = []
    
    if level1_file.exists():
        with open(level1_file, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                # Parse JSON array from file
                if content.startswith('['):
                    level1_data = json.loads(content)
            except:
                pass
    
    # Level 2: Code Summarization
    print("Loading Level 2 samples (Code Summarization)...")
    level2_file = eval_dir / "code_to_summary_instruction.txt"
    level2_data = []
    
    if level2_file.exists():
        with open(level2_file, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                if content.startswith('['):
                    level2_data = json.loads(content)
            except:
                pass
    
    # Level 3: Code Generation (from alpaca data)
    print("Loading Level 3 samples (Code Generation)...")
    level3_file = eval_dir / "alpaca_gpt4_data.json"
    level3_data = load_json_file(level3_file)
    
    # Filter Level 3 for code-related tasks
    level3_data = [item for item in level3_data 
                   if any(keyword in item.get('instruction', '').lower() 
                         for keyword in ['code', 'python', 'function', 'script', 'program'])]
    
    # Sample 5 from each level
    samples = []
    
    # Level 1
    if level1_data:
        selected_l1 = random.sample(level1_data, min(5, len(level1_data)))
        for idx, item in enumerate(selected_l1, 1):
            samples.append({
                "level": 1,
                "type": "code_completion",
                "question_id": f"L1_Q{idx}",
                "instruction": item.get("Instruction", ""),
                "input": item.get("Input", ""),
                "expected_output": item.get("Output", ""),
                "category": "operator_knowledge"
            })
    
    # Level 2
    if level2_data:
        selected_l2 = random.sample(level2_data, min(5, len(level2_data)))
        for idx, item in enumerate(selected_l2, 1):
            samples.append({
                "level": 2,
                "type": "code_summarization",
                "question_id": f"L2_Q{idx}",
                "instruction": item.get("Instruction", ""),
                "code": item.get("Input", ""),
                "expected_summary": item.get("Output", ""),
                "category": "code_understanding"
            })
    
    # Level 3
    if level3_data:
        selected_l3 = random.sample(level3_data, min(5, len(level3_data)))
        for idx, item in enumerate(selected_l3, 1):
            samples.append({
                "level": 3,
                "type": "code_generation",
                "question_id": f"L3_Q{idx}",
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "expected_output": item.get("output", ""),
                "category": "code_generation"
            })
    
    return samples

def save_samples(samples, output_file="data/evaluation_samples.jsonl"):
    """Save samples to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {output_file}")

if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING EVALUATION SAMPLES")
    print("=" * 80)
    
    samples = prepare_evaluation_samples()
    
    print(f"\nTotal samples prepared: {len(samples)}")
    print(f"  - Level 1 (Code Completion): {sum(1 for s in samples if s['level'] == 1)}")
    print(f"  - Level 2 (Code Summarization): {sum(1 for s in samples if s['level'] == 2)}")
    print(f"  - Level 3 (Code Generation): {sum(1 for s in samples if s['level'] == 3)}")
    
    if samples:
        save_samples(samples)
        print("\nReady for testing!")
    else:
        print("\nError: No samples prepared. Check GeoCode-Eval files.")
