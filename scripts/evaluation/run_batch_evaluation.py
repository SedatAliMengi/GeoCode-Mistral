#!/usr/bin/env python3
"""
Batch evaluation runner - Test multiple samples sequentially
Uses prepared evaluation_samples.jsonl
"""

import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import logging
from datetime import datetime
from io import StringIO

log_file = f"evaluation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model once
logger.info("Loading model and tokenizer...")
logger.info("Optimizing for low memory usage...")
model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
logger.info("Model loaded successfully")

def generate_response(prompt, max_length=None, temperature=0.5, max_new_tokens=50):
    """Generate response from model"""
    
    if max_length is None:
        prompt_tokens = len(tokenizer.encode(prompt))
        max_length = min(prompt_tokens + max_new_tokens, 2048)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output directly
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove prompt from response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        # Clean up
        del inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()
        
        return response_text
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return None

def run_batch_evaluation(samples_file="data/evaluation_samples.jsonl", output_file="evaluation_results.jsonl", limit_samples=3):
    """Run evaluation on all samples"""
    
    logger.info("=" * 80)
    logger.info("BATCH EVALUATION RUNNER (DEBUG MODE - FIRST 3 SAMPLES)")
    logger.info("=" * 80)
    
    results = []
    
    # Read samples
    try:
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {samples_file}")
        return
    
    logger.info(f"Loaded {len(samples)} samples from {samples_file}\n")
    
    # Group by level
    levels = {}
    for sample in samples:
        level = sample.get('level')
        if level not in levels:
            levels[level] = []
        levels[level].append(sample)
    
    # Process each level
    for level in sorted(levels.keys()):
        level_samples = levels[level]
        # DEBUG: Limit to first 3 total samples
        if sum(len(v) for v in levels.values()) > limit_samples:
            level_samples = level_samples[:1]  # 1 örnek per level
        
        logger.info(f"LEVEL {level}: Processing {len(level_samples)} samples")
        logger.info("-" * 80)
        
        for idx, sample in enumerate(level_samples, 1):
            question_id = sample.get('question_id', f'L{level}_Q{idx}')
            
            # Prepare prompt based on level
            if level == 1:  # Code completion
                prompt = f"{sample.get('instruction', '')}\n\n{sample.get('input', '')}"
            elif level == 2:  # Code summarization
                prompt = f"{sample.get('instruction', '')}\n\nCode:\n{sample.get('code', '')}"
            else:  # Code generation
                prompt = sample.get('instruction', '')
            
            logger.info(f"\n[{question_id}] Testing...")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            # Generate response
            response = generate_response(prompt, temperature=0.5, max_new_tokens=50)
            
            if response:
                logger.info(f"Response: {response[:100]}...\n")
                
                result = {
                    "question_id": question_id,
                    "level": level,
                    "type": sample.get('type', ''),
                    "prompt": prompt,
                    "response": response,
                    "expected": sample.get('expected_output', '') or sample.get('expected_summary', ''),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
            else:
                logger.warning(f"Failed to generate response for {question_id}")
        
        # Memory cleanup after each level
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("-" * 80)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"\nSaved {len(results)} results to {output_file}")
    logger.info("=" * 80)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_batch_evaluation()
