#!/usr/bin/env python3
"""
GeoCode-GPT Showcase Inference
Runs 5 geospatial questions, saves full outputs to showcase_outputs.jsonl
"""

import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/final_model"
OUTPUT_FILE = "showcase_outputs.jsonl"

QUESTIONS = [
    {
        "id": 1,
        "topic": "NDVI Calculation",
        "question": "Write Python code to calculate NDVI from Landsat 8 imagery using rasterio."
    },
    {
        "id": 2,
        "topic": "Shapefile Filtering",
        "question": "Write Python code to load a shapefile using GeoPandas and filter features by a specific attribute value."
    },
    {
        "id": 3,
        "topic": "GEE Export",
        "question": "Write a Google Earth Engine JavaScript script to filter a Sentinel-2 image collection by date and region, then export the result to Google Drive."
    },
    {
        "id": 4,
        "topic": "Raster Reading",
        "question": "Write Python code to open a GeoTIFF file using rasterio, read the first band, and print the coordinate reference system and resolution."
    },
    {
        "id": 5,
        "topic": "Buffer Zone",
        "question": "Write Python code to create a 5km buffer zone around points in a GeoPandas GeoDataFrame and visualize the result with matplotlib."
    },
]

def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer

def generate(model, tokenizer, question):
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    # Decode only the new tokens (fixes the prompt-repeating bug)
    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

def main():
    model, tokenizer = load_model()
    results = []

    for q in QUESTIONS:
        print(f"[{q['id']}/5] {q['topic']}")
        print(f"  Q: {q['question']}")

        start = datetime.now()
        response = generate(model, tokenizer, q["question"])
        elapsed = (datetime.now() - start).total_seconds()

        print(f"  A: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"  ({elapsed:.1f}s)\n")

        results.append({
            "id": q["id"],
            "topic": q["topic"],
            "question": q["question"],
            "response": response,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat()
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
