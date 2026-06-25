#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT TXT dosyalarını JSONL formatına dönüştür
GeoCode-SFT1-10 klasörlerindeki .txt → .jsonl
"""

import os
import json
import glob

def convert_txt_to_jsonl():
    """GeoCode-SFT klasörlerindeki .txt dosyalarını JSONL'ye dönüştür"""
    
    print("🚀 SFT TXT → JSONL Dönüştürme Başlıyor...\n")
    
    # Tüm SFT klasörlerini bulma
    sft_dirs = sorted([d for d in glob.glob("data/datasets/GeoCode-SFT*") if os.path.isdir(d)])
    print(f"📁 Bulunan SFT Klasörleri: {len(sft_dirs)}")
    print(f"   {sft_dirs}\n")
    
    total_lines = 0
    
    for sft_dir in sft_dirs:
        print(f"\n📂 İşleniyor: {sft_dir}/")
        
        # Klasördeki .txt dosyalarını bulma
        txt_files = glob.glob(os.path.join(sft_dir, "*.txt"))
        print(f"   📄 Dosya Sayısı: {len(txt_files)}")
        
        if not txt_files:
            print(f"   ⚠️  .txt dosyası bulunamadı!")
            continue
        
        # Her .txt dosyasını işle
        for txt_file in txt_files:
            jsonl_file = txt_file.replace(".txt", ".jsonl")
            
            print(f"   ├─ {os.path.basename(txt_file)} → {os.path.basename(jsonl_file)}")
            
            try:
                # TXT dosyasını oku
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # JSONL dosyasına yaz
                with open(jsonl_file, 'w', encoding='utf-8') as out:
                    line_count = 0
                    for line in lines:
                        line = line.strip()
                        if not line:  # Boş satırları atla
                            continue
                        
                        try:
                            # JSON parse et (eğer JSON ise)
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            # JSON değilse, text olarak sar
                            data = {
                                "text": line,
                                "instruction": line
                            }
                        
                        # JSONL formatında yaz
                        out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        line_count += 1
                
                print(f"      ✅ {line_count} satır dönüştürüldü")
                total_lines += line_count
            
            except Exception as e:
                print(f"      ❌ HATA: {e}")
    
    print(f"\n🎉 TAMAMLANDI!")
    print(f"   Toplam İşlenen Satır: {total_lines}")
    print(f"   ✅ Tüm .jsonl dosyaları oluşturuldu!")

if __name__ == "__main__":
    convert_txt_to_jsonl()
