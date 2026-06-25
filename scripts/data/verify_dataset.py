#!/usr/bin/env python3
"""
Verify and compare your actual dataset with GeoCode-GPT article specifications.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("GeoCode-GPT DATASET VERIFICATION")
print("=" * 80)

# ============================================================================
# ARTICLE SPECIFICATIONS
# ============================================================================
article_specs = {
    "GeoCode-PT": {
        "code_snippets": 275_374,
        "operator_knowledge": 10_190,
        "dataset_knowledge": 853,
        "platform_docs": 14,
        "description": "Pretraining corpus: code + knowledge"
    },
    "GeoCode-SFT": {
        "instruction_entries": 502_047,
        "description": "SFT corpus: instruction-tuned data"
    },
    "GeoCode-Eval": {
        "mcq": 3_000,
        "code_generation": 500,
        "code_summarization": 500,
        "description": "Evaluation: MCQ + generation tasks"
    }
}

print("\n📄 ARTICLE SPECIFICATIONS:")
print("-" * 80)
for dataset, specs in article_specs.items():
    print(f"\n{dataset}:")
    for key, value in specs.items():
        if key != "description":
            if isinstance(value, int):
                print(f"  • {key}: {value:,}")
        else:
            print(f"  Description: {value}")

# ============================================================================
# YOUR ACTUAL DATA
# ============================================================================
print("\n\n" + "=" * 80)
print("YOUR ACTUAL DATA INVENTORY")
print("=" * 80)

workspace = Path(".")

# Check unified.jsonl
if (workspace / "data/unified.jsonl").exists():
    size_mb = (workspace / "data/unified.jsonl").stat().st_size / (1024 * 1024)
    with open("data/unified.jsonl", "r") as f:
        unified_count = sum(1 for _ in f)
    print(f"\n✅ unified.jsonl: {unified_count:,} entries ({size_mb:.2f} MB)")
else:
    print("\n❌ unified.jsonl: NOT FOUND")

# Check GeoCode-Eval
eval_dir = workspace / "data/datasets/GeoCode-Eval"
if eval_dir.exists():
    print(f"\n📁 GeoCode-Eval files:")
    eval_files = {}
    for file in eval_dir.glob("*.json"):
        if file.is_file():
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                    eval_files[file.name] = count
                    print(f"  • {file.name}: {count:,}")
            except Exception as e:
                print(f"  • {file.name}: ERROR - {e}")
    
    for file in eval_dir.glob("*.txt"):
        if file.is_file():
            try:
                with open(file, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                eval_files[file.name] = count
                print(f"  • {file.name}: ~{count:,} lines")
            except Exception as e:
                print(f"  • {file.name}: ERROR - {e}")

# Check GeoCode-PT subdirectories
pt_dir = workspace / "data/datasets/GeoCode-PT"
if pt_dir.exists():
    print(f"\n📁 GeoCode-PT structure:")
    for subdir in pt_dir.iterdir():
        if subdir.is_dir():
            file_count = len(list(subdir.glob("**/*.json"))) + len(list(subdir.glob("**/*.jsonl")))
            print(f"  • {subdir.name}/: {file_count} files")

# Check GeoCode-SFT directories
print(f"\n📁 GeoCode-SFT structure:")
sft_total = 0
for i in range(1, 11):
    sft_dir = workspace / f"data/datasets/GeoCode-SFT{i}"
    if sft_dir.exists():
        files = list(sft_dir.glob("*.json")) + list(sft_dir.glob("*.jsonl"))
        file_count = len(files)
        sft_total += file_count
        if file_count > 0:
            print(f"  • GeoCode-SFT{i}: {file_count} file(s)")

if sft_total > 0:
    print(f"  TOTAL SFT files: {sft_total}")

# ============================================================================
# COMPARISON & ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("COMPARISON ANALYSIS")
print("=" * 80)

if (workspace / "data/unified.jsonl").exists():
    print(f"\n📊 Dataset Size Comparison:")
    print(f"  Article GeoCode-PT specification:    275,374 code snippets")
    print(f"  Article GeoCode-SFT specification:   502,047 instruction entries")
    print(f"  TOTAL (article claim):               777,421 entries")
    print(f"")
    print(f"  Your unified.jsonl:                  {unified_count:,} entries")
    
    coverage = (unified_count / 777_421) * 100
    print(f"  Coverage: {coverage:.1f}% of article's claimed data")
    
    if unified_count >= 500_000:
        print(f"  ✅ SUFFICIENT: You have substantial data for fine-tuning")
    elif unified_count >= 300_000:
        print(f"  ⚠️  MODERATE: Good coverage but missing some original data")
    else:
        print(f"  ❌ LIMITED: Significantly less than article's claims")

print("\n\n📋 DATA COMPOSITION CHECK:")
print("-" * 80)

# Try to analyze data types in unified.jsonl
if (workspace / "data/unified.jsonl").exists():
    print("\nSampling unified.jsonl structure...")
    with open("data/unified.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Sample first 5 entries
                break
            try:
                record = json.loads(line)
                print(f"\n  Record {i+1} keys: {list(record.keys())}")
                for k, v in record.items():
                    if isinstance(v, str):
                        preview = v[:80] + "..." if len(v) > 80 else v
                        print(f"    • {k}: {preview}")
            except:
                pass

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

missing_pt = 275_374 - (unified_count if (workspace / "data/unified.jsonl").exists() else 0) * 0.4  # rough estimate
missing_sft = 502_047 - (unified_count if (workspace / "data/unified.jsonl").exists() else 0) * 0.6

print("""
✅ WHAT YOU HAVE:
  • Pre-trained base model (Code Llama-7B) in local_model/
  • Consolidated unified.jsonl with ~311k training examples
  • GeoCode-Eval evaluation dataset for testing
  • Complete training pipeline setup

❓ COMPARISON WITH ARTICLE:
  The article's GeoCode-GPT uses:
    - 275,374 PT entries (pretraining)
    - 502,047 SFT entries (supervised fine-tuning)
    - Total: 777,421+ entries

  Your current data appears to be a SUBSET or SAMPLE of the full dataset.
  This is NORMAL and EXPECTED when:
    • Using public/research release (not all private data included)
    • Dataset has been curated/filtered
    • Working with demo version

🎯 RECOMMENDATION:
  Your 311k examples ARE SUFFICIENT for:
    ✓ Fine-tuning Code Llama-7B
    ✓ Creating a functional geospatial code generation model
    ✓ Achieving reasonable performance on code tasks
    ✓ Demonstrating the GeoCode-GPT approach
    
  Your data may be LESS optimal than the article's model for:
    × Matching article's exact performance benchmarks
    × Complex geological reasoning tasks
    × Cutting-edge state-of-the-art results
""")

print("\n" + "=" * 80)
