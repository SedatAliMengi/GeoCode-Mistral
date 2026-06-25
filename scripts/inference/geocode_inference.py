#!/usr/bin/env python3
"""
GeoCode-Mistral-7B Inference Script
Geospatial code generation queries
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import logging
import json
from datetime import datetime
from io import StringIO

# Log hem console'a hem dosyaya
log_file = f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model and tokenizer
logger.info("[*] Loading model and tokenizer (Float16 Hybrid Mode)...")

# Float16 Mode Configuration:
# - device_map="auto": Intelligent GPU-CPU split
# - torch_dtype=torch.float16: Half precision for memory efficiency
# - No quantization complexity (avoids meta tensor issues)
# Result: GPU ~7GB + CPU ~7GB = ~14GB total
#         Stable and proven to work

model = AutoModelForCausalLM.from_pretrained(
    "./models/final_model",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
logger.info("[OK] Model loaded successfully (Float16 - Stable Hybrid Mode)!")

def generate_response(prompt, max_length=None, temperature=0.5):
    """Generate response from model with token streaming and output capture"""
    
    # Dinamik max_length: prompt boyutuna göre ayarla
    if max_length is None:
        prompt_tokens = len(tokenizer.encode(prompt))
        # Prompt + min 150 token output
        max_length = min(prompt_tokens + 300, 2048)  # Max 2048 token
    
    logger.info(f"\n[>] Input: {prompt}\n")
    logger.info(f"[~] Generating output (max_length={max_length}, temp={temperature})...\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Capture output
    output_buffer = StringIO()
    streamer = TextStreamer(
        tokenizer, 
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0  # 60 saniye timeout
    )
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                streamer=streamer,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Get captured output
        response_text = output_buffer.getvalue()
        
        # Print and log
        logger.info("[OUTPUT]:\n")
        logger.info(response_text)
        logger.info("\n" + "="*80 + "\n")
        
        # Also save to JSON
        save_interaction(prompt, response_text, max_length, temperature)
        
        return response_text
    
    except Exception as e:
        logger.error(f"[ERROR] Generation failed: {e}")
        return None

def save_interaction(prompt, response, max_length, temperature):
    """Save prompt-response pairs to JSON"""
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "max_length": max_length,
        "temperature": temperature
    }
    
    # Append to interactions file
    interactions_file = "interactions.jsonl"
    try:
        with open(interactions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        logger.info(f"[OK] Saved to {interactions_file}")
    except Exception as e:
        logger.error(f"[ERROR] Error saving interaction: {e}")

# Example prompts
examples = [
    "Write Python code using Google Earth Engine to calculate NDVI for a specific region and visualize the results",
    "How can I perform land use classification using satellite imagery with Python?",
    "What is the Sentinel-2 dataset and how is it used in remote sensing?",
    "Explain how to create a buffer zone around geographic features using GeoPandas"
]

if __name__ == "__main__":
    print("=" * 80)
    print("[GEOCODE] GeoCode-Mistral-7B Inference Interface")
    print("=" * 80)
    
    while True:
        print("\nOptions:")
        print("1. Try example prompts")
        print("2. Enter custom prompt")
        print("3. Exit")
        
        choice = input("\nChoose (1/2/3): ").strip()
        
        if choice == "1":
            print("\nExample prompts:")
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example}")
            
            try:
                ex_choice = int(input(f"\nSelect example (1-{len(examples)}): "))
                if 1 <= ex_choice <= len(examples):
                    prompt = examples[ex_choice - 1]
                    generate_response(prompt)
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == "2":
            print("\nEnter your prompt (type END on new line to finish):")
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            prompt = " ".join(lines).strip()
            
            if prompt:
                generate_response(prompt)
            else:
                print("Prompt cannot be empty!")
        
        elif choice == "3":
            print("\n[+] Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")
