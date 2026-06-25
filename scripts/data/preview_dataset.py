from datasets import load_dataset
import glob
files = glob.glob("data/datasets/GeoCode-Eval/**/*.json", recursive=True) + glob.glob("data/datasets/GeoCode-Eval/**/*.jsonl", recursive=True)
print("files found:", len(files))
ds = load_dataset("json", data_files=files, split="train")
print("dataset length:", len(ds))
print("columns:", ds.column_names)
print("first example keys:", list(ds[0].keys()))
from script import formatting_func
print("formatted preview:", formatting_func(ds[0], None)[:500])
