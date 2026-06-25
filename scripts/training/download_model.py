# download_model.py
import os
import sys

# Kodun çalıştığını anlamak için ilk sinyal
print("--- PYTHON DEVREDE: Kod baslatiliyor... ---")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("--- Kutuphaneler yuklendi, indirme basliyor... ---")
except ImportError as e:
    print(f"HATA: Kutuphaneler eksik! Lutfen sunu calistir: pip install torch transformers acceleration")
    sys.exit()

# --- AYARLAR ---
MODEL_ID = "codellama/CodeLlama-7b-hf"
SAVE_DIRECTORY = "./models/base_model"

# Klasörü oluştur
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

print(f"Hedef Model: {MODEL_ID}")
print(f"Kayit Yeri: {SAVE_DIRECTORY}")
print("NOT: Bu islem internet hizina gore zaman alabilir...")

try:
    # 1. TOKENIZER
    print("\n[1/2] Tokenizer indiriliyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    print(">>> Tokenizer TAMAM.")

    # 2. MODEL
    print("\n[2/2] Model agirliklari indiriliyor (En uzun kisim burasi)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.save_pretrained(SAVE_DIRECTORY)
    print(f">>> Model basariyla '{SAVE_DIRECTORY}' klasorune kaydedildi.")

    print("\n------------------------------------------------")
    print("ISLEM BASARILI! Artik script.py dosyasina gecebilirsin.")
    print("------------------------------------------------")

except Exception as e:
    print(f"\n!!! HATA OLUSTU !!!\nDetay: {e}")