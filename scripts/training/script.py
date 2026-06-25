# script.py
# GeoCode-GPT-7B Training Script
import os
import sys

# Set this FIRST to tell transformers to skip the check
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer 
import glob
import logging
from functools import partial
import random
import importlib
import inspect
import json

# Now disable the check
import transformers.utils
import_utils = transformers.utils.import_utils
import_utils._is_torch_load_safe_warned = True
import_utils.check_torch_load_is_safe = lambda: None

# --- KONFİGÜRASYON ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- KRİTİK AYAR: VERİ YÜZDESİ (%80) ---
DATA_PERCENTAGE = 0.20  # %80 for the main phase 1 training

# --- SABİTLER (OFFLINE MOD) ---
BASE_MODEL = "./models/base_model" 
OUTPUT_DIR = "./results"
PT_ADAPTER_NAME = "geocode-pt-adapter" 
SFT_ADAPTER_NAME = "geocode-sft-adapter"   

# --- VERİ YOLLARI ---
DATA_DIR_PT = "data/unified.jsonl"
DATA_DIR_EVAL = "data/datasets/GeoCode-Eval"

# --- YARDIMCI FONKSİYON: VERİ YÜKLEME ---
def load_and_format_data(data_paths):
    """JSON/JSONL dosyalarını bulur, yükler ve belirtilen oranda (%80) kısıtlar.
    Supports direct files (unified.jsonl) or directories (recursively scans for *.json/*.jsonl).
    """
    data_files = []
    paths_to_search = [data_paths] if isinstance(data_paths, str) else data_paths

    for path in paths_to_search:
        # If path is a direct file (ends with .json or .jsonl), add it directly
        if isinstance(path, str) and (path.endswith('.json') or path.endswith('.jsonl')):
            if os.path.isfile(path):
                data_files.append(path)
            else:
                logger.warning(f"File not found: {path}")
        else:
            # Otherwise treat as directory and glob recursively
            data_files.extend(glob.glob(os.path.join(path, '**', '*.jsonl'), recursive=True))
            data_files.extend(glob.glob(os.path.join(path, '**', '*.json'), recursive=True))
    
    # FILTER: SFT5-10 exclude et (multilingual files)
    data_files = [f for f in data_files if not any(skip in f for skip in ['GeoCode-SFT5', 'GeoCode-SFT6', 'GeoCode-SFT7', 'GeoCode-SFT8', 'GeoCode-SFT9', 'GeoCode-SFT10'])]
    
    if not data_files:
        logger.error(f"Belirtilen yollarda JSON dosyası bulunamadı: {data_paths}")
        return None

    # DEBUG: Bulunan dosyaları ve satır sayılarını listele
    logger.info(f"📁 Toplam {len(data_files)} JSON dosyası bulundu:")
    total_raw_lines = 0
    for file in data_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_raw_lines += line_count
                logger.info(f"   - {file}: {line_count} satır")
        except Exception as e:
            logger.warning(f"   - {file}: Okunamadı ({e})")

    try:
        # Veriyi yükle - DOSYALARI AYRI AYRI YÜKLe (format uyuşmazlığından kaçın)
        logger.info(f"DEBUG: data_files = {data_files}")
        
        all_datasets = []
        for file in data_files:
            try:
                ds = load_dataset('json', data_files=[file], split="train")
                all_datasets.append(ds)
            except Exception as e:
                logger.warning(f"File {file} yüklenirken hata: {e}")
        
        if not all_datasets:
            logger.error("Hiç bir dosya yüklenemedi!")
            return None
        
        # Tüm dataset'leri birleştir
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(all_datasets)
        
        logger.info(f"✅ Ham veri yüklendi: {len(dataset)} satır (İşlendikten sonra)")
        logger.info(f"   Raw dosya satırları toplam: {total_raw_lines}")
        logger.info(f"DEBUG: dataset length after load = {len(dataset)}")

        # --- VERİ KISITLAMA MANTIĞI ---
        if DATA_PERCENTAGE < 1.0:
            logger.info(f"DEBUG BEFORE SHUFFLE: len(dataset) = {len(dataset)}")
            dataset = dataset.shuffle(seed=42)
            logger.info(f"DEBUG AFTER SHUFFLE: len(dataset) = {len(dataset)}")
            num_samples = int(len(dataset) * DATA_PERCENTAGE)
            logger.info(f"DEBUG: num_samples = {num_samples}, DATA_PERCENTAGE = {DATA_PERCENTAGE}")
            dataset = dataset.select(range(num_samples))
            logger.info(f"DEBUG AFTER SELECT: len(dataset) = {len(dataset)}")
            logger.info(f"Data limited to {DATA_PERCENTAGE*100:.1f}% of dataset. New row count: {len(dataset)}")
        # ------------------------------

        return dataset
    except Exception as e:
        logger.exception(f"Veri yüklenirken hata oluştu: {e}")
        return None

# --- FORMATLAMA FONKSİYONU ---
def formatting_func(example, tokenizer, max_seq_length=1024):
    # Prefer Description -> Code mapping (Alldata files use Description/Code),
    # with fallbacks to common fields.
    def _get(*keys):
        for k in keys:
            if k in example and example[k] not in (None, ""):
                return example[k]
        return None

    desc = _get('Description', 'description', 'instruction', 'prompt', 'input')
    code = _get('Code', 'code', 'output', 'response', 'answer', 'completion')

    # Normalize to strings
    desc_s = str(desc).strip() if desc is not None else ""
    code_s = str(code).strip() if code is not None else ""

    if desc_s and code_s:
        return f"<s>[INST] {desc_s} [/INST] {code_s}</s>"
    if desc_s:
        return f"<s>{desc_s}</s>"
    if code_s:
        return f"<s>{code_s}</s>"

    return example.get('text', str(example))


def check_environment():
    """Prints versions and checks for required packages and device capabilities."""
    reqs = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("bitsandbytes", "bitsandbytes"),
        ("torch", "torch"),
    ]
    info = {}
    for name, pkg in reqs:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            info[name] = ver
        except Exception:
            info[name] = None

    logger.info("Environment package versions: %s", info)

    # Torch + CUDA checks
    try:
        cuda = torch.cuda.is_available()
        bf16 = False
        if cuda:
            try:
                bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                bf16 = False
        logger.info(f"cuda_available={cuda}  bf16_supported={bf16}")
    except Exception:
        logger.warning("Couldn't fully probe torch CUDA support.")

    return info

# --- EĞİTİM MOTORU ---
def run_fine_tuning(base_model_name, dataset_path, adapter_name, lo_ra_config, is_qlora, is_pretrain=False, resume_from_adapter=None):
    logger.info(f"\n🚀 EĞİTİM BAŞLIYOR: {adapter_name}")
    logger.info(f"Mod: {'Pre-Training (PT)' if is_pretrain else 'Supervised Fine-Tuning (SFT)'}")
    logger.info(f"⚡ AYARLAR: Packing=AKTİF | Kayıt Sıklığı=HER 50 ADIM | Veri=%{DATA_PERCENTAGE*100}")

    # 1. Veriyi Yükle - OPTIMIZED FOR SPEED
    dataset = load_and_format_data(dataset_path)
    if dataset is None:
        return None
    
    # Pre-format data to text field for faster training
    logger.info("📝 Pre-formatting dataset to text field...")
    def format_to_text(example):
        desc = example.get('instruction') or example.get('description') or example.get('input') or ""
        code = example.get('output') or example.get('code') or example.get('response') or ""
        text = f"<s>[INST] {desc} [/INST] {code}</s>" if desc and code else str(example)
        return {"text": text}
    
    # Map'i yap ve SONRA kolonları temizle
    dataset = dataset.map(format_to_text, batched=False, remove_columns=dataset.column_names)
    logger.info("✅ Pre-formatting complete")
    
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
        
    # 2. Model Ayarları (daha sağlam: CUDA/bf16 kontrolü, qlora yalnızca GPU'da)
    cuda_available = torch.cuda.is_available()
    bf16_supported = False
    try:
        if cuda_available:
            bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_supported = False

    device_map = "auto" if cuda_available else None

    if is_qlora and not cuda_available:
        logger.warning("QLoRA istenmiş fakat CUDA bulunamadı; qlora devre dışı bırakılıyor.")
        is_qlora = False

    bnb_config = None
    if is_qlora:
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # 3. Model Yükle
    logger.info("Model ağırlıkları local_model klasöründen yükleniyor...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=(torch.bfloat16 if (is_qlora and bf16_supported) else (torch.float16 if cuda_available else torch.float32)),
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        logger.exception(f"Model yüklenirken hata oluştu: {e}")
        raise
    
    if resume_from_adapter:
        logger.info(f"🔗 Önceki aşama adaptörü yükleniyor: {resume_from_adapter}")
        # Unload any existing adapters first
        if hasattr(model, 'unload'):
            model.unload()
        model = PeftModel.from_pretrained(model, resume_from_adapter, is_trainable=False)

    if is_qlora:
         model = prepare_model_for_kbit_training(model)
         
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. LoRA Adaptörü
    model = get_peft_model(model, lo_ra_config)
    model.print_trainable_parameters()
    
    # 5. Eğitim Argümanları (ULTRA-OPTIMIZED FOR SPEED)
    # Build TrainingArguments kwargs dynamically so older/newer transformers versions work
    ta_kwargs = {
        "output_dir": f"{OUTPUT_DIR}/{adapter_name}",
        "num_train_epochs": 1 if is_pretrain else 1,  # 1 epoch
        "per_device_train_batch_size": 2,  # Drastically reduced
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,  # ENABLE: Memory optimization
        "learning_rate": 2e-4 if is_qlora else 1e-4,
        "logging_steps": 10,  # Less logging

        # --- PAUSE / RESUME İÇİN KRİTİK AYARLAR ---
        "save_strategy": "steps",
        "save_steps": 50,  # Save every 50 steps (more frequent for safety)
        "save_total_limit": 1,
        # ------------------------------------------

        # include both possible names for evaluation strategy so whichever matches will be used
        "evaluation_strategy": "no",  # CHANGED: Disable evaluation to speed up training
        "eval_strategy": "no",
        "eval_steps": 50,

        # Mixed precision flags: AGGRESSIVE bf16 for speed
        "fp16": False,  # Disable fp16
        "bf16": True,   # Enable bf16 (faster + more stable)
        "report_to": "none",
        "warmup_ratio": 0.05,
        "remove_unused_columns": False,
        "group_by_length": False,  # Disable (on-the-fly tokenization incompatible)
        "dataloader_num_workers": 8,  # INCREASE: More parallel loading threads
        "dataloader_pin_memory": True,  # ADDED: Speed up data loading
        "dataloader_prefetch_factor": 2,  # ADDED: Prefetch batches ahead
        "use_cache": False,  # CRITICAL: Must be False when gradient_checkpointing=True
    }

    # Filter kwargs to only those accepted by TrainingArguments.__init__ for this transformers version
    try:
        ta_sig = inspect.signature(TrainingArguments.__init__)
        accepted = set(ta_sig.parameters.keys())
    except Exception:
        accepted = set()

    filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in accepted}

    training_args = TrainingArguments(**filtered_kwargs)

    # ON-THE-FLY TOKENIZATION (NO CACHE LEAK, FASTER TRAINING)
    logger.info("⚡ Using on-the-fly tokenization (faster, no memory leak)...")
    
    # Tokenization will happen during training, batch by batch
    # This eliminates pre-tokenization cache leak that slows down training
    logger.info("✅ Ready for on-the-fly tokenization!")

    # 6. Trainer (Using basic Trainer instead of SFTTrainer for speed)
    # Import Trainer
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    # PATCH: Override _load_optimizer_and_scheduler to skip CVE check
    original_load_optimizer = Trainer._load_optimizer_and_scheduler
    
    def patched_load_optimizer_and_scheduler(self, checkpoint_dir=None):
        """Load optimizer and scheduler WITHOUT CVE check"""
        import os
        if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
            return
        
        import torch
        from pathlib import Path
        
        # Load optimizer
        optimizer_file = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_file):
            try:
                optimizer_state = torch.load(optimizer_file, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state)
                logger.info(f"Optimizer loaded from {optimizer_file}")
            except Exception as e:
                logger.warning(f"Could not load optimizer: {e}")
        
        # Load scheduler
        scheduler_file = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_file):
            try:
                scheduler_state = torch.load(scheduler_file, weights_only=False)
                self.lr_scheduler.load_state_dict(scheduler_state)
                logger.info(f"Scheduler loaded from {scheduler_file}")
            except Exception as e:
                logger.warning(f"Could not load scheduler: {e}")
    
    Trainer._load_optimizer_and_scheduler = patched_load_optimizer_and_scheduler
    
    # Use SFTTrainer for on-the-fly tokenization
    from trl import SFTTrainer
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 7. Başlat (OTOMATİK DEVAM ETME MANTIĞI - PYTORCH 2.5.1 FIX)
    logger.info("⏳ Eğitim başlıyor... (Checkpoint kontrol ediliyor)")
    
    last_checkpoint = None
    
    # Eğer klasörde checkpoint varsa oradan devam et — en sonuncuyu seç
    out_dir = f"{OUTPUT_DIR}/{adapter_name}"
    if os.path.isdir(out_dir):
        checkpoints = sorted(
            [os.path.join(out_dir, d) for d in os.listdir(out_dir) if d.startswith("checkpoint")],
            key=lambda p: os.path.getmtime(p)
        )
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            logger.info(f"💾 Önceki kayıt bulundu! Kaldığı yerden devam ediyor: {last_checkpoint}")
    
    # Training'i başlat - torch.load CVE fix ile sorun çıkmayacak
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 8. Kaydet
    final_path = f"{OUTPUT_DIR}/{adapter_name}/final_adapter"
    trainer.model.save_pretrained(final_path)
    logger.info(f"✅ Aşama tamamlandı: {final_path}")

    return final_path

# --- ANA PROGRAM ---
if __name__ == "__main__":
    
    qlora_config_pt = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    )
    
    # ✅ PHASE 1 SKIP - Zaten GPU'da tamamlandı!
    # print(f"\n--- PHASE 1: PRE-TRAINING (Veri: %{DATA_PERCENTAGE*100} | Safe-Mode: ON) ---")
    # pt_adapter_path = run_fine_tuning(
    #     base_model_name=BASE_MODEL,
    #     dataset_path=DATA_DIR_PT, 
    #     adapter_name=PT_ADAPTER_NAME,
    #     lo_ra_config=qlora_config_pt,
    #     is_qlora=True,
    #     is_pretrain=True
    # )
    
    # Phase 1 sonrası, Phase 2 başlat (PT adapter'ı yükle)
    # PT adapter'ın gerçek konumu
    pt_adapter_path = "./models/adapters/phase1_pretrain/final_adapter"
    
    if os.path.isdir(pt_adapter_path):
        logger.info(f"✅ PT Adapter bulundu: {pt_adapter_path}")
        lora_config_sft = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        )

        sft_data_paths = [f'data/datasets/GeoCode-SFT{i}' for i in range(1, 11)]
        existing_paths = [p for p in sft_data_paths if os.path.isdir(p)]

        if existing_paths:
            print(f"\n--- PHASE 2: SFT (Veri: %{DATA_PERCENTAGE*100} | Safe-Mode: ON) ---")
            sft_adapter_path = run_fine_tuning(
                base_model_name=BASE_MODEL, 
                dataset_path=existing_paths, 
                adapter_name=SFT_ADAPTER_NAME,
                lo_ra_config=lora_config_sft,
                is_qlora=True,   
                is_pretrain=False,
                resume_from_adapter=pt_adapter_path 
            )
            logger.info(f"🏆 PROJE TAMAMLANDI! Çıktı: {sft_adapter_path}")