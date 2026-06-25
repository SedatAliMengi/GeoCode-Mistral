#!/usr/bin/env python3
"""
Test model on GeoCode-Eval questions (Paper-compatible)
Level 1: 50 MCQ from alpaca_gpt4_data.json
Level 2: 30 Summarization from code_to_summary_instruction.txt
Level 3: 30 Code Generation from summary_to_code_instruction.txt
With timeout mechanism for freeze issues
"""

import json
import random
import torch
import os
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global for timeout
generation_complete = False
generation_result = None

# Load model
logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
model.eval()
logger.info("Model loaded\n")

def generate_response(prompt, max_new_tokens=50, timeout_sec=30):
    """Generate with timeout (threading-based for Windows)"""
    global generation_complete, generation_result
    
    generation_complete = False
    generation_result = None
    
    def generate_worker():
        global generation_result
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    top_k=40
                )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            generation_result = response
        except Exception as e:
            generation_result = None
    
    thread = threading.Thread(target=generate_worker, daemon=False)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        # Thread still running - timeout occurred
        return None
    
    return generation_result

def load_level1_mcq(limit=50):
    """Load MCQ from alpaca_gpt4_data.json"""
    logger.info("Loading Level 1 (MCQ)...")
    questions = []
    
    try:
        with open('./data/datasets/GeoCode-Eval/alpaca_gpt4_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data[:limit]:
            questions.append({
                'instruction': item.get('instruction', ''),
                'input': item.get('input', ''),
                'expected': item.get('output', '')
            })
    except Exception as e:
        logger.error(f"Error loading MCQ: {e}")
    
    logger.info(f"  Loaded {len(questions)} MCQ")
    return questions

def load_level2_summarization(limit=30):
    """Load Summarization"""
    logger.info("Loading Level 2 (Summarization)...")
    questions = []
    
    try:
        with open('./data/datasets/GeoCode-Eval/code_to_summary_instruction.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            items = content.split('[BEGIN]')[1:]
            
            for item in items[:limit]:
                parts = item.split('[END]')[0].strip().split('\n\n')
                if len(parts) >= 2:
                    questions.append({
                        'instruction': 'Summarize the following code',
                        'input': parts[0],
                        'expected': parts[1] if len(parts) > 1 else ''
                    })
    except Exception as e:
        logger.error(f"Error loading summarization: {e}")
    
    logger.info(f"  Loaded {len(questions)} summarization")
    return questions

def load_level3_generation(limit=30):
    """Load Code Generation"""
    logger.info("Loading Level 3 (Code Generation)...")
    questions = []
    
    try:
        with open('./data/datasets/GeoCode-Eval/summary_to_code_instruction.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            items = content.split('[BEGIN]')[1:]
            
            for item in items[:limit]:
                parts = item.split('[END]')[0].strip().split('\n\n')
                if len(parts) >= 2:
                    questions.append({
                        'instruction': 'Write code based on the following description',
                        'input': parts[0],
                        'expected': parts[1] if len(parts) > 1 else ''
                    })
    except Exception as e:
        logger.error(f"Error loading code generation: {e}")
    
    logger.info(f"  Loaded {len(questions)} code generation")
    return questions

def calculate_accuracy(response, expected):
    """Simple accuracy metric: word overlap"""
    if not response or not expected:
        return 0.0
    
    resp_words = set(response.lower().split())
    exp_words = set(expected.lower().split())
    
    if len(exp_words) == 0:
        return 0.0
    
    overlap = len(resp_words & exp_words)
    accuracy = overlap / len(exp_words)
    return min(accuracy, 1.0) * 100

def test_level(name, questions, timeout=30):
    """Test a level"""
    logger.info(f"\nTesting {name}...")
    logger.info("-" * 60)
    
    results = []
    stats = {'total': 0, 'success': 0, 'timeout': 0, 'accuracy': 0}
    
    for idx, question in enumerate(questions, 1):
        stats['total'] += 1
        
        prompt = f"{question['instruction']}\n\n{question['input']}"
        expected = question['expected']
        
        response = generate_response(prompt, timeout_sec=timeout)
        
        if response is None:
            logger.info(f"[{idx}/{len(questions)}] TIMEOUT")
            stats['timeout'] += 1
            continue
        
        accuracy = calculate_accuracy(response, expected)
        logger.info(f"[{idx}/{len(questions)}] OK - {accuracy:.1f}%")
        
        stats['success'] += 1
        stats['accuracy'] += accuracy
        
        results.append({
            'id': f"{name}_{idx}",
            'prompt': prompt[:200],
            'response': response[:200],
            'expected': expected[:200],
            'accuracy': accuracy
        })
    
    if stats['success'] > 0:
        stats['accuracy'] = stats['accuracy'] / stats['success']
    
    return results, stats

def main():
    logger.info("=" * 70)
    logger.info("GeoCode-GPT VALIDATION - Paper-Compatible Test")
    logger.info("=" * 70)
    logger.info("")
    
    # Load test data
    level1_q = load_level1_mcq(5)  # Small test first
    level2_q = []
    level3_q = []
    
    total = len(level1_q) + len(level2_q) + len(level3_q)
    logger.info(f"\nTotal questions loaded: {total}")
    logger.info("")
    
    # Run tests
    all_results = []
    all_stats = {}
    
    if level1_q:
        r1, s1 = test_level("LEVEL 1 (MCQ)", level1_q, timeout=30)
        all_results.extend(r1)
        all_stats['L1'] = s1
    
    if level2_q:
        r2, s2 = test_level("LEVEL 2 (Summarization)", level2_q, timeout=30)
        all_results.extend(r2)
        all_stats['L2'] = s2
    
    if level3_q:
        r3, s3 = test_level("LEVEL 3 (Code Generation)", level3_q, timeout=30)
        all_results.extend(r3)
        all_stats['L3'] = s3
    
    # Report
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    
    overall_accuracy = 0
    overall_count = 0
    
    for level, stats in sorted(all_stats.items()):
        if stats['total'] > 0:
            success_rate = (stats['success'] / stats['total']) * 100
            accuracy = stats['accuracy']
            
            logger.info(f"\n{level}:")
            logger.info(f"  Tested: {stats['total']}")
            logger.info(f"  Success: {stats['success']} ({success_rate:.1f}%)")
            logger.info(f"  Timeout: {stats['timeout']}")
            logger.info(f"  Accuracy: {accuracy:.1f}%")
            
            if stats['success'] > 0:
                overall_accuracy += accuracy * stats['success']
                overall_count += stats['success']
    
    if overall_count > 0:
        overall = overall_accuracy / overall_count
        logger.info(f"\nOverall Accuracy: {overall:.1f}%")
    
    logger.info("=" * 70)
    
    # Save results
    output_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
