# GeoCode-Mistral 🌍

> **Domain-Specific LLM for Geospatial Code Generation on Consumer Hardware**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Mistral-7B](https://img.shields.io/badge/Base%20Model-Mistral--7B-orange.svg)](https://mistral.ai/)
[![QLoRA](https://img.shields.io/badge/Training-QLoRA%20%2B%20LoRA-green.svg)](https://arxiv.org/abs/2305.14314)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Thesis](https://img.shields.io/badge/Thesis-Düzce%20University-purple.svg)](#citation)

GeoCode-Mistral is a fine-tuned language model for geospatial code generation, targeting platforms such as **Google Earth Engine**, **ArcPy**, **QGIS**, and **GeoPandas**. It adapts the methodology of **GeoCode-GPT** (Hou et al., 2025) — which used Code Llama — by substituting **Mistral-7B** as the backbone and training entirely on a **consumer-grade laptop GPU (8 GB VRAM)**.

This work was completed as an undergraduate thesis at **Düzce University, Department of Computer Engineering**.

> ### ⚠️ Honest-results notice
> This is a **constrained reproduction with a limited result.** On a 25-question slice of the GeoCode-Eval benchmark, this model scored **20% overall (5/25)**, versus the original paper's reported ~73%. During evaluation I traced the underperformance to a **specific data-formatting bug in my own training pipeline** (not only hardware/data limits). I document this openly because diagnosing *why* a model underperforms is part of the contribution. See [Results](#results) and [Key Finding](#key-finding-a-training-data-formatting-bug).

---

## Table of Contents

- [Motivation](#motivation)
- [What This Project Is (and Is Not)](#what-this-project-is-and-is-not)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Base Model Selection](#base-model-selection)
  - [Two-Stage Training Pipeline](#two-stage-training-pipeline)
  - [Hyperparameters (as actually configured)](#hyperparameters-as-actually-configured)
- [Hardware Setup](#hardware-setup)
- [Dataset](#dataset)
- [Results](#results)
- [Comparison with GeoCode-GPT](#comparison-with-geocode-gpt)
- [Key Finding: a Training-Data Formatting Bug](#key-finding-a-training-data-formatting-bug)
- [Gradio UI](#gradio-ui)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing the Evaluation](#reproducing-the-evaluation)
- [Training Your Own Model](#training-your-own-model)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Motivation

General-purpose LLMs exhibit two recurring failure modes when generating geospatial code:

| Failure Mode | Description |
|---|---|
| **Refusal to Code** | The model declines to generate code for spatial operations, citing insufficient context or capability. |
| **Coding Hallucinations** | The model produces syntactically plausible but semantically incorrect code — referencing non-existent API methods, wrong libraries, or invalid band names. |

These problems stem from the relative scarcity of geospatial code in general pre-training corpora. GeoCode-GPT (Hou et al., 2025) showed that domain-specific fine-tuning reduces both failure modes. This project attempts that adaptation using Mistral-7B on consumer hardware — and, just as importantly, evaluates the result honestly and diagnoses where it fell short.

---

## What This Project Is (and Is Not)

**This project IS:**
- A methodological adaptation of GeoCode-GPT to a different base model (Mistral-7B vs. Code Llama) on consumer hardware (8 GB VRAM, local).
- An **honest, end-to-end study**: a working two-stage QLoRA→LoRA pipeline, adapter merging, a streaming Gradio UI, and a resumable evaluation harness.
- A demonstration of **diagnosis under a limited result** — the evaluation surfaced a concrete pipeline bug, which is documented rather than hidden.

**This project IS NOT:**
- A claim to match or beat GeoCode-GPT. The original reports ~73%; this run scored **20%** on a 25-question subset.
- A clean reproduction — base model, data fraction, quantization, and the bug above all diverge from the original. These are documented in the [comparison table](#comparison-with-geocode-gpt).
- A production system.

---

## Repository Structure

```
GeoCode-Mistral/
├── chatbot.py                     # Streaming Gradio UI (gr.Blocks + gr.Chatbot)
├── models/
│   ├── base_model/                # Mistral-7B base weights (local)
│   ├── final_model/               # Merged model used for inference
│   └── adapters/
│       ├── phase1_pretrain/        # Stage 1 QLoRA adapter
│       └── phase2_sft/             # Stage 2 LoRA adapter
├── scripts/
│   ├── training/                  # script.py (two-stage training), merge_adapters.py, download_model.py
│   ├── inference/                 # geocode_inference.py, showcase_inference.py
│   ├── data/                      # dataset conversion / verification utilities
│   └── evaluation/                # batch + manual evaluation scripts
├── data/
│   ├── datasets/                  # GeoCode-PT, GeoCode-Eval, GeoCode-SFT1..10
│   ├── unified.jsonl              # Normalized Stage-1 corpus
│   └── evaluation_samples.jsonl
├── testing/                       # Self-contained evaluation harness (see below)
│   ├── run_tests.py
│   ├── test_questions.jsonl       # 25 sampled GeoCode-Eval questions
│   └── results/                   # responses.jsonl, evaluation_report.md, final_scores.md
└── outputs/                       # logs and generated artifacts
```

---

## Methodology

### Base Model Selection

Mistral-7B was selected over Code Llama (the original base model) primarily for memory efficiency under an 8 GB VRAM budget. Mistral-7B-v0.1 uses **Sliding Window Attention**, which keeps peak VRAM growth more manageable for longer code sequences than standard full attention. The trade-off is that Mistral-7B is a *general* model, not a code-specialized one — a factor relevant to the results below.

### Two-Stage Training Pipeline

**Stage 1 — Domain Pre-Training (QLoRA)**
- Objective: expose the model to geospatial code/knowledge corpora (causal LM objective).
- Data: a normalized `unified.jsonl` derived from GeoCode-PT.
- Method: QLoRA — base weights quantized to 4-bit NF4; only adapter weights trained.

**Stage 2 — Supervised Fine-Tuning (LoRA)**
- Objective: teach instruction-following and code generation from natural-language descriptions.
- Data: GeoCode-SFT instruction files (subset).
- Method: LoRA via HuggingFace TRL `SFTTrainer`, resuming from the Stage-1 adapter.

> **Important:** Stage 2 is where the [data-formatting bug](#key-finding-a-training-data-formatting-bug) occurred — the geospatial instruction data was silently malformed before training. Read that section before drawing conclusions from the results.

### Hyperparameters (as actually configured)

These values are taken **directly from [`scripts/training/script.py`](scripts/training/script.py)**, not idealized.

| Parameter | Value | Source |
|---|---|---|
| **Base Model** | Mistral-7B (loaded 4-bit NF4) | `BASE_MODEL` |
| **LoRA Rank (r)** | **64** | `LoraConfig(r=64)` |
| **LoRA Alpha (α)** | **128** (α/r = 2.0) | `lora_alpha=128` |
| **LoRA Dropout** | 0.05 | `lora_dropout=0.05` |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all attn + MLP) | `target_modules=[...]` |
| **Quantization** | 4-bit NF4 + double quant, compute dtype bf16 | `BitsAndBytesConfig` |
| **Mixed Precision** | bf16 | `bf16=True` |
| **Per-device Batch Size** | **2** | `per_device_train_batch_size=2` |
| **Gradient Accumulation** | **1** (effective batch = 2) | `gradient_accumulation_steps=1` |
| **Learning Rate** | 2e-4 | `learning_rate` |
| **Warmup Ratio** | 0.05 | `warmup_ratio=0.05` |
| **Epochs** | 1 | `num_train_epochs=1` |
| **Gradient Checkpointing** | Enabled | `gradient_checkpointing=True` |
| **Trainer** | TRL `SFTTrainer` | — |
| **Dataset Fraction** | **20%** (`DATA_PERCENTAGE=0.20`) | — |

> Note: the in-code comment says "%80" but the value is `0.20`. The actual fraction used was **20%**. Multilingual SFT files (`GeoCode-SFT5`–`SFT10`) were excluded to save time.

---

## Hardware Setup

All training and evaluation ran on a **local consumer laptop** — no cloud GPU.

| Component | Specification |
|---|---|
| **CPU** | Intel Core i7-14700HX |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM) |
| **RAM** | 32 GB |
| **OS** | Windows 11 |
| **PyTorch / CUDA** | 2.5.1 + CUDA 12.1 |

Because the merged 7B model exceeds 8 GB VRAM, inference uses `device_map="auto"` with **CPU offloading** of the layers that don't fit. The original GeoCode-GPT used A100 (80 GB) GPUs on the full dataset; the gap in resources is large and intentional — the goal was to test feasibility on accessible hardware.

---

## Dataset

This project uses the **GeoCode-PT**, **GeoCode-SFT**, and **GeoCode-Eval** datasets from Hou et al. (2025).

| Dataset | Purpose | Subset Used |
|---|---|---|
| **GeoCode-PT** | Stage-1 domain pre-training corpus | ~20% |
| **GeoCode-SFT** | Stage-2 instruction tuning | ~20%, multilingual files dropped |
| **GeoCode-Eval** | Evaluation benchmark | 25-question stratified sample |

---

## Results

Evaluation was run on a **25-question stratified sample** of GeoCode-Eval using the harness in [`testing/`](testing/). Multiple-choice questions were auto-scored where the model gave a clear answer; open-ended (code-generation and summarization) questions were graded by manual human review against the expected outputs.

**Sample composition:** 10 code-generation (`summary_to_code`), 5 code-summarization (`code_to_summary`), 5 API-knowledge MCQ, 5 platform-identification MCQ.

### Overall: **5 / 25 = 20%**

### By task type

| Task Type | Questions | Pass | Success Rate |
|---|---|---|---|
| Platform identification (MCQ) | 5 | 3 | **60%** |
| API knowledge (MCQ) | 5 | 1 | **20%** |
| Code → summary | 5 | 1 | **20%** |
| Summary → code (generation) | 10 | 0 | **0%** |
| **Overall** | **25** | **5** | **20%** |

### By difficulty level

| Level | Task | Questions | Success Rate |
|---|---|---|---|
| **1** | MCQ (knowledge / identification) | 10 | **40%** |
| **2** | Code summarization | 5 | **20%** |
| **3** | Code generation | 10 | **0%** |

### Observed failure modes

The results show a clear **difficulty gradient**: the model can sometimes *recognize* what code does (60% at identifying the platform) but **cannot generate** correct geospatial code (0%). The dominant failure on generation tasks was **library hallucination** — the model confidently imported non-existent packages (e.g. `googleearthapi`, `geopoints`, `lspacenow`, `geosim`) or fell back to unrelated machine-learning libraries (`scikit-learn`, `pandas`) instead of the Earth Engine `ee` API. On MCQ, it frequently ignored the "answer A/B/C/D" instruction and wrote prose instead.

These behaviors are consistent with the [training-data bug](#key-finding-a-training-data-formatting-bug) described below.

---

## Comparison with GeoCode-GPT

| Dimension | GeoCode-GPT (Hou et al., 2025) | GeoCode-Mistral (this project) |
|---|---|---|
| **Base Model** | Code Llama 7B (code-specialized) | Mistral-7B-v0.1 (general) |
| **Hardware** | NVIDIA A100 (80 GB), cloud | RTX 4070 Laptop (8 GB), local |
| **Training** | (per paper) | QLoRA (Stage 1) + LoRA (Stage 2) |
| **Quantization** | — | 4-bit NF4 |
| **Dataset Coverage** | Full | ~20% subset; multilingual dropped |
| **Evaluation Set** | Full GeoCode-Eval | 25-question stratified subset |
| **Reported Success** | **~73%** | **20%** |

**Honest interpretation of the gap.** The 73% → 20% difference is **not** explained by hardware alone. The largest single contributor appears to be a **data-formatting bug** that malformed the geospatial training data during Stage 2 (next section). Resource limits (general-purpose base model, 20% data, 4-bit quantization, dropped multilingual data) are real but secondary. Isolating each factor would require fixing the bug and re-running controlled ablations — the clear next step.

---

## Key Finding: a Training-Data Formatting Bug

The most important outcome of this project was discovering, *during evaluation*, why the model underperformed.

**The bug.** In [`scripts/training/script.py`](scripts/training/script.py), the Stage-2 formatter reads **lowercase** field names:

```python
desc = example.get('instruction') or example.get('description') or example.get('input') or ""
code = example.get('output')      or example.get('code')        or example.get('response') or ""
text = f"<s>[INST] {desc} [/INST] {code}</s>" if desc and code else str(example)
```

But the GeoCode-SFT instruction files use **capitalized** keys — `Instruction`, `Input`, `Output`. Because Python's `dict.get()` is case-sensitive, `desc` and `code` both became empty strings for every geospatial example, so the `if desc and code` check failed and each example fell through to `str(example)` — i.e. the model was trained on the **raw Python dictionary string**:

```
{'Instruction': 'You are an expert...', 'Input': '...', 'Output': 'var ndvi = ...'}
```

instead of a proper `[INST] instruction [/INST] code` pair.

**The compounding factor.** The only Stage-2 file with *lowercase* keys was `alpaca_gpt4_data.json` — a **generic, non-geospatial** instruction set. So the only data that was formatted correctly into clean `[INST]` pairs was generic chatbot data. This explains the observed behavior precisely: the model learned generic instruction-following and never properly learned geospatial code generation in the right format.

**The fix** (for future work) is one of:
- normalize keys to lowercase before formatting, or
- read the capitalized keys directly (`example.get('Instruction')`, `example.get('Output')`).

**Why this is reported, not hidden.** Finding a bug in one's own pipeline through honest evaluation — and tracing the failure to a root cause — is a core engineering skill. The corrected, retrained result is expected to be substantially higher, and quantifying that improvement is the primary future-work item.

---

## Gradio UI

[`chatbot.py`](chatbot.py) provides a streaming chat interface (`gr.Blocks` + `gr.Chatbot`) with:
- token-by-token streaming (`TextIteratorStreamer` + background thread),
- a domain selector (Auto-detect / Google Earth Engine / ArcPy / QGIS / GeoPandas) that injects per-domain system prompts,
- the Mistral `[INST]` prompt format used in training,
- example prompts and a dark GitHub-style theme.

**Launch:**
```bash
python chatbot.py
# Opens at http://127.0.0.1:7860
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8 GB VRAM workable with CPU offload)
- CUDA Toolkit 12.x

### Setup
```bash
git clone https://github.com/SedatAliMengi/GeoCode-Mistral.git
cd GeoCode-Mistral

python -m venv venv
source venv/bin/activate          # Linux/macOS
# .\venv\Scripts\activate         # Windows

# Install a CUDA build of torch FIRST (see requirements.txt header), then:
pip install -r requirements.txt
```

> **Note:** `torch` must be a CUDA build, not the default PyPI wheel. Install it from the PyTorch index first:
> ```bash
> pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
> ```
> Then `pip install -r requirements.txt` for the rest. Pinned versions match the tested environment (PyTorch 2.5.1+cu121, Transformers 4.57, Gradio 6.3).

---

## Usage

### Chatbot (Gradio)
```bash
python chatbot.py
```

### Direct inference (CLI)
```bash
python scripts/inference/geocode_inference.py
```

### Example prompts

| Domain | Example Prompt |
|---|---|
| **Google Earth Engine** | "Compute NDVI from a Sentinel-2 collection over a region and add the greenest-pixel composite to the map." |
| **GeoPandas** | "Read a shapefile, reproject to EPSG:3857, and compute each polygon's area in km²." |
| **ArcPy** | "Create a 500 m buffer around a feature class and save it to a file geodatabase." |
| **QGIS (PyQGIS)** | "Load a shapefile as a vector layer, add it to the project, and apply categorized symbology." |

---

## Reproducing the Evaluation

The evaluation is fully self-contained in [`testing/`](testing/) and is resumable.

```bash
# 1. Run all 25 questions (saves after each; safe to Ctrl+C and resume)
python testing/run_tests.py

# 2. Open testing/results/evaluation_report.md and grade the open-ended
#    questions by ticking the Pass/Fail checkboxes.

# 3. Compute the final score from your grades
python testing/run_tests.py --tally   # writes testing/results/final_scores.md
```

Multiple-choice questions with a clear answer are auto-scored; everything else is graded by human review (the harness leaves checkboxes to fill).

---

## Training Your Own Model

Training is driven by [`scripts/training/script.py`](scripts/training/script.py) (two-stage) and [`scripts/training/merge_adapters.py`](scripts/training/merge_adapters.py) (adapter merge). Adjust `DATA_PERCENTAGE`, the `LoraConfig`, and the data paths at the top of the script.

```bash
# Two-stage training (edit paths/flags inside the script first)
python scripts/training/script.py

# Merge Stage-1 + Stage-2 adapters into ./models/final_model
python scripts/training/merge_adapters.py
```

> If you reproduce this, **fix the field-name bug first** (see [Key Finding](#key-finding-a-training-data-formatting-bug)) — otherwise the geospatial data will not train correctly.

---

## Limitations

1. **Training-data formatting bug (primary).** Geospatial Stage-2 data was malformed before training; this is the leading suspected cause of the low score. See [Key Finding](#key-finding-a-training-data-formatting-bug).
2. **Reduced data (20%) + dropped multilingual files.** Less coverage than intended; the in-code comment ("%80") does not match the actual setting (`0.20`).
3. **General-purpose base model.** Mistral-7B is not code-specialized, unlike Code Llama in the original paper.
4. **Small evaluation set.** 25 cases is directional, not statistically robust.
5. **4-bit NF4 quantization** introduces some quality loss versus higher precision.
6. **Manual + format-based scoring.** Open-ended grading is human-judgment based; MCQ auto-scoring only triggers when the model gives a clear letter (it often wrote prose instead). No sandboxed code execution was used.
7. **Single base model / single run.** No ablations across models, ranks, or seeds.

---

## Future Work

- **Fix the formatting bug and retrain** — the highest-priority item; quantify the resulting improvement (this is the natural "before/after" experiment).
- **Restore full data + multilingual files**, and set `DATA_PERCENTAGE` to the intended value.
- **Try a code-specialized base** (Code Llama, or a current code model) for a fairer comparison to the paper.
- **Automated execution-based evaluation** — a sandboxed runner that executes generated code instead of manual inspection.
- **LoRA rank / data ablations** to isolate each factor's contribution to the gap.
- **Retrieval-augmented inference** over GEE/ArcPy/QGIS documentation to curb hallucinated APIs.

---

## Citation

**This project:**
```bibtex
@misc{mengi2026geocodemistral,
  title        = {GeoCode-Mistral: Domain-Specific Geospatial Code Generation on Consumer Hardware},
  author       = {Mengi, Sedat Ali},
  year         = {2026},
  institution  = {Düzce University, Department of Computer Engineering},
  note         = {Undergraduate thesis},
  url          = {https://github.com/SedatAliMengi/GeoCode-Mistral}
}
```

**Original GeoCode-GPT paper:**
```bibtex
@article{hou2025geocodegpt,
  title   = {GeoCode-GPT: A Large Language Model for Geospatial Code Generation},
  author  = {Hou, et al.},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year    = {2025},
  note    = {https://doi.org/10.1016/j.jag.2025.104456}
}
```

---

## Acknowledgements

- **Hou et al. (2025)** — for the GeoCode-GPT methodology and the GeoCode-PT / GeoCode-SFT / GeoCode-Eval datasets this project builds on.
- **Mistral AI** — for releasing Mistral-7B openly.
- **HuggingFace** — for the Transformers, PEFT, and TRL libraries.

---

## License

Released under the [MIT License](LICENSE). The GeoCode datasets are subject to the terms of the original GeoCode-GPT paper and their respective licenses.
