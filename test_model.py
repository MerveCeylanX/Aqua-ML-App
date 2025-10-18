#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Test Scripti
Bu script, verilen girdilerle modele direkt istek atar ve sonucu gösterir.
Arayüz sonucu ile karşılaştırma yapmak için kullanılır.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

def load_model_and_meta():
    """Model ve metadata'yı yükle"""
    try:
        pipe = joblib.load("best_model.joblib")
        with open("best_model.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            features = meta.get("features", [])
        return pipe, features
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None, None

def create_test_data():
    """Test verisi oluştur - arayüzdeki girdilerle aynı (ham girdiler)"""
    
    # Test verisi - arayüzdeki girdilerle aynı (model kendi preprocessing'ini yapacak)
    test_data = {
        # Synthesis parametreleri
        "Agent/Sample(g/g)": 1.0,
        "Soaking_Time(min)": 240.0,
        "Soaking_Temp(K)": 360,
        "Activation_Time(min)": 240.0,
        "Activation_Temp(K)": 673.15,
        "Activation_Heating_Rate (K/min)": 5.0,
        
        # Adsorbent parametreleri
        "BET_Surface_Area(m2/g)": 796.0,
        "Total_Pore_Volume(cm3/g)": 0.342696991,
        "Micropore_Volume(cm3/g)": 0.22,
        "Average_Pore_Diameter(nm)": 2.71,
        "pHpzc": 5.4,
        
        # Percent değerleri (model bunları molar değerlere çevirecek)
        "C_percent": 74.3,
        "H_percent": 3.5,
        "O_percent": 5.4,
        "N_percent": 2.5,
        "S_percent": 2.5,
        
        # Process parametreleri
        "Solution_pH": 7.0,
        "Temperature(K)": 323.0,
        "Initial_Concentration(mg/L)": 475.0,
        "Dosage(g/L)": 4.1,
        "Contact_Time(min)": 71.0,
        "Agitation_speed(rpm)": 240.0,
        
        # Kategorik değişkenler (model bunları işleyecek)
        "Activation_Atmosphere": "N2",  # "N2", "Air", "SG"
        "Target_Phar": "CIP"  # Ciprofloxacin kodu
    }
    
    return test_data

def prepare_data_for_model(data):
    """Veriyi model için hazırla - sadece DataFrame oluştur, model kendi preprocessing'ini yapacak"""
    # DataFrame oluştur - model pipeline'ı tüm preprocessing'i yapacak
    df = pd.DataFrame([data])
    return df

def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("MODEL TEST SCRİPTİ")
    print("=" * 60)
    
    # Model yükle
    print("1. Model yükleniyor...")
    pipe, features = load_model_and_meta()
    if pipe is None:
        print("❌ Model yüklenemedi!")
        return
    
    print(f"✅ Model yüklendi. Feature sayısı: {len(features)}")
    print(f"📋 Features: {features}")
    print()
    
    # Test verisi oluştur
    print("2. Test verisi oluşturuluyor...")
    test_data = create_test_data()
    print("✅ Test verisi oluşturuldu:")
    for key, value in test_data.items():
        print(f"   {key}: {value}")
    print()
    
    # Veriyi model için hazırla
    print("3. Veri model için hazırlanıyor...")
    df = prepare_data_for_model(test_data)
    print(f"✅ Veri hazırlandı. Shape: {df.shape}")
    print(f"📊 DataFrame kolonları: {df.columns.tolist()}")
    print("💡 Model pipeline'ı tüm preprocessing'i (percent→molar, solute params, vs.) yapacak")
    print()
    
    # Model tahmini yap
    print("4. Model tahmini yapılıyor...")
    try:
        prediction = pipe.predict(df)
        qe_value = float(prediction[0])
        print(f"🎯 Tahmini qe değeri: {qe_value:.3f} mg/g")
        print()
        
        # Sonuç özeti
        print("=" * 60)
        print("SONUÇ ÖZETİ")
        print("=" * 60)
        print(f"Seçilen İlaç: Ciprofloxacin (CIP)")
        print(f"Activation Atmosphere: {test_data['Activation_Atmosphere']}")
        print(f"Percent Değerleri:")
        print(f"  C%: {test_data['C_percent']}")
        print(f"  H%: {test_data['H_percent']}")
        print(f"  O%: {test_data['O_percent']}")
        print(f"  N%: {test_data['N_percent']}")
        print(f"  S%: {test_data['S_percent']}")
        print(f"💡 Model pipeline'ı bu değerleri molar değerlere çevirdi ve solute parametrelerini ekledi")
        print(f"Tahmini qe: {qe_value:.3f} mg/g")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Tahmin sırasında hata: {e}")
        import traceback
        print("Tam hata detayı:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
