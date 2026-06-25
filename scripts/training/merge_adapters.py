#!/usr/bin/env python3
"""
Merge script for GeoCode-GPT Phase 1 + Phase 2 adapters
Merges: Phase1 (QLoRA) + Phase2 (LoRA) + Base Model (Mistral 7B)
Output: Final merged model ready for inference
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_MODEL = "./models/base_model"
PHASE1_ADAPTER = "./models/adapters/phase1_pretrain/final_adapter"
PHASE2_ADAPTER = "./models/adapters/phase2_sft/final_adapter"
OUTPUT_DIR = "./models/final_model"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("=" * 60)
logger.info("🚀 GeoCode-GPT MERGE: Phase1 + Phase2 + Base Model")
logger.info("=" * 60)

# Step 1: Load base model
logger.info(f"📦 Loading base model from {BASE_MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",  # CPU'ye offload - GPU belleği yetmiyorsa
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Step 2: Load Phase 1 adapter (QLoRA)
logger.info(f"🔗 Loading Phase 1 adapter (pre-training)...")
model = PeftModel.from_pretrained(model, PHASE1_ADAPTER, is_trainable=False)

# Step 3: Merge Phase 1 adapter into base model
logger.info("⚙️  Merging Phase 1 adapter into base model...")
model = model.merge_and_unload()

# Step 4: Load Phase 2 adapter (LoRA)
logger.info(f"🔗 Loading Phase 2 adapter (fine-tuning)...")
model = PeftModel.from_pretrained(model, PHASE2_ADAPTER, is_trainable=False)

# Step 5: Merge Phase 2 adapter into base model
logger.info("⚙️  Merging Phase 2 adapter into base model...")
model = model.merge_and_unload()

# Step 6: Save final merged model
logger.info(f"💾 Saving final merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

# Step 7: Copy tokenizer
logger.info("📝 Copying tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("=" * 60)
logger.info("✅ MERGE COMPLETE!")
logger.info(f"📍 Final model saved to: {OUTPUT_DIR}")
logger.info("=" * 60)

# Quick test
logger.info("\n🧪 Quick inference test...")
try:
    test_input = "Merhaba, nasılsın?"
    inputs = tokenizer(test_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Input: {test_input}")
    logger.info(f"Output: {response}")
    logger.info("✅ Inference works!")
except Exception as e:
    logger.error(f"❌ Inference test failed: {e}")

logger.info("\n🎉 GeoCode-GPT is ready for deployment!")
