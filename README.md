# Aqua-ML: Aktif Karbon Adsorpsiyon Kapasitesi Tahmin Uygulaması

**Makine Öğrenmesi Tabanlı Adsorpsiyon Kapasitesi Tahmin Sistemi**

Bu uygulama, aktif karbon ile farmasötik kirleticilerin adsorpsiyonunu tahmin eden bir Streamlit web arayüzüdür. Sentez koşulları, adsorban özellikleri ve proses parametrelerini kullanarak adsorpsiyon kapasitesi (qe) tahmini yapar.

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Girdileri](#-model-girdileri)
- [Desteklenen İlaçlar](#-desteklenen-ilaçlar)
- [Test ve Doğrulama](#-test-ve-doğrulama)
- [Geliştirme](#-geliştirme)
- [Dosya Açıklamaları](#-dosya-açıklamaları)

---

## 🚀 Özellikler

### Ana Özellikler
- ✅ **Tekil Parametre Girişi**: Kullanıcı dostu form ile tek seferde tahmin
- ✅ **Toplu Tahmin**: Excel/CSV dosyası ile çoklu veri tahmini
- ✅ **21 Farmasötik Desteklenir**: APAP, CIP, SMX, TC, OTC ve 16 ilaç daha
- ✅ **İnteraktif Grafikler**: Plotly tabanlı duyarlılık analizleri
- ✅ **Karşılaştırmalı Analiz**: Farklı ilaçlar için adsorpsiyon kapasitesi karşılaştırması

### Duyarlılık Analizleri
Uygulamada aşağıdaki parametrelerin adsorpsiyon kapasitesi üzerindeki etkisini görselleştirebilirsiniz:

#### Sentez Koşulları:
- 🧪 Ajan/Numune Oranı
- ⏰ Emdirim Süresi
- ⏲️ Aktivasyon Süresi
- 🔥 Aktivasyon Sıcaklığı

#### Proses Koşulları:
- 📈 Başlangıç Konsantrasyonu
- 🌡️ Çözelti Sıcaklığı
- 🧪 pH Değeri
- ⚖️ Adsorban Dozajı
- ⏱️ Temas Süresi

---

## 📁 Proje Yapısı

```
aqua_ml_app/
│
├── aqua_ml_app.py              # Ana Streamlit uygulaması
├── test_model.py               # Model doğrulama scripti
├── requirements.txt            # Python bağımlılıkları
│
├── best_model.joblib           # Eğitilmiş ML modeli (Pipeline)
├── best_model.meta.json        # Model metadata (features, model bilgileri)
│
├── header.jpg                  # Uygulama başlık görseli
├── header.jpeg                 # (Alternatif başlık görseli)
│
├── ui_specs/                   # UI ve referans dosyaları
│   └── drug_map.xlsx           # İlaç adı-kod eşleştirmeleri (Kullanımda)
│
└── src/                        # Model geliştirme kaynak kodları
    ├── __init__.py
    ├── config.py               # Konfigürasyon ve sabitler
    ├── data_io.py              # Veri giriş/çıkış işlemleri
    ├── estimators.py           # ML model wrapper'ları
    ├── evaluation.py           # Model değerlendirme
    ├── evaluation1.py          # Ek değerlendirme modülü
    ├── features.py             # Özellik mühendisliği
    ├── imports.py              # Import yönetimi
    ├── pipelines.py            # ML pipeline tanımları
    ├── preprocessing.py        # Veri ön işleme
    ├── tunning.py              # Hiperparametre optimizasyonu
    └── README.md               # Kaynak kod modül dokümantasyonu
```

---

## 🛠️ Kurulum

### 1. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### Gerekli Kütüphaneler:

**Temel Kütüphaneler:**
- `streamlit` - Web arayüzü
- `pandas` - Veri işleme
- `numpy` - Sayısal hesaplamalar
- `plotly` - İnteraktif grafikler
- `scikit-learn` - ML modeli
- `joblib` - Model yükleme
- `openpyxl` - Excel desteği

**Opsiyonel ML Kütüphaneleri:**
- `catboost`, `lightgbm`, `xgboost` - Gradient boosting modelleri
- `interpret` - Explainable Boosting Machine (EBM)

### 2. Gerekli Dosyaları Kontrol Edin

Aşağıdaki dosyaların mevcut olduğundan emin olun:

```
✅ best_model.joblib          # ML modeli
✅ best_model.meta.json       # Model metadata
✅ ui_specs/drug_map.xlsx     # İlaç haritası
✅ header.jpg                 # Başlık görseli (opsiyonel)
```

---

## 🎯 Kullanım

### Streamlit Uygulamasını Başlatın

```bash
streamlit run aqua_ml_app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### 1️⃣ Tekil Giriş Modu

1. Sol panelde **"Tekil Giriş"** sekmesini seçin
2. Parametreleri girin:
   - **Sentez Koşulları**: Aktivasyon parametreleri
   - **Adsorban Özellikleri**: BET yüzey alanı, gözenek özellikleri, element analizi
   - **Proses Koşulları**: pH, sıcaklık, konsantrasyon, dozaj, vb.
   - **Hedef İlaç**: İlaç seçimi (zorunlu)
   - **Aktivasyon Atmosferi**: N₂, Air, veya SG (zorunlu)
3. **"Tahmin Et"** butonuna basın
4. Sonuçları görüntüleyin:
   - Tahmini qe değeri (mg/g)
   - Antibiyotik karşılaştırma grafiği
   - Duyarlılık analizleri (pH, sıcaklık, konsantrasyon, vb.)

### 2️⃣ Excel Yükleme Modu

1. **"Excel Yükle"** sekmesini seçin
2. Şablon dosyasını indirin (CSV veya Excel)
3. Parametreleri doldurun
4. Dosyayı yükleyin
5. Tahmin sonuçlarını indirin (CSV veya Excel)

**Zorunlu Kolonlar:**
- `Target_Phar` - İlaç kodu (örn: CIP, SMX, TC)
- `Activation_Atmosphere` - Atmosfer tipi (N2, Air, SG)

---

## 📊 Model Girdileri

### Sentez Koşulları
| Parametre | Açıklama | Birim |
|-----------|----------|-------|
| `Agent/Sample(g/g)` | Aktivasyon ajanı/numune oranı | g/g |
| `Soaking_Time(min)` | Emdirim süresi | dakika |
| `Soaking_Temp(K)` | Emdirim sıcaklığı | Kelvin |
| `Activation_Time(min)` | Aktivasyon süresi | dakika |
| `Activation_Temp(K)` | Aktivasyon sıcaklığı | Kelvin |
| `Activation_Heating_Rate (K/min)` | Isıtma hızı | K/dakika |
| `Activation_Atmosphere` | Atmosfer tipi | N2/Air/SG |

### Adsorban Özellikleri
| Parametre | Açıklama | Birim |
|-----------|----------|-------|
| `BET_Surface_Area(m2/g)` | BET yüzey alanı | m²/g |
| `Total_Pore_Volume(cm3/g)` | Toplam gözenek hacmi | cm³/g |
| `Micropore_Volume(cm3/g)` | Mikrogözenek hacmi | cm³/g |
| `Average_Pore_Diameter(nm)` | Ortalama gözenek çapı | nm |
| `pHpzc` | Sıfır yük noktası pH'ı | - |
| `C_percent` | Karbon yüzdesi | % (wt.) |
| `H_percent` | Hidrojen yüzdesi | % (wt.) |
| `O_percent` | Oksijen yüzdesi | % (wt.) |
| `N_percent` | Azot yüzdesi | % (wt.) |
| `S_percent` | Kükürt yüzdesi | % (wt.) |

### Proses Koşulları
| Parametre | Açıklama | Birim |
|-----------|----------|-------|
| `Solution_pH` | Çözelti pH'ı | - |
| `Temperature(K)` | Sıcaklık | Kelvin |
| `Initial_Concentration(mg/L)` | Başlangıç konsantrasyonu | mg/L |
| `Dosage(g/L)` | Adsorban dozajı | g/L |
| `Contact_Time(min)` | Temas süresi | dakika |
| `Agitation_speed(rpm)` | Karıştırma hızı | rpm |

### Hedef İlaç
`Target_Phar` - İlaç kodu (bkz: [Desteklenen İlaçlar](#-desteklenen-ilaçlar))

**Not:** Model, percent değerlerini otomatik olarak molar değerlere çevirir ve seçilen ilaca göre solute parametrelerini (E, S, A, B, V) ekler.

---

## 💊 Desteklenen İlaçlar

Aşağıdaki 21 farmasötik kirletici desteklenmektedir:

| Kod | İlaç Adı | Kategori |
|-----|----------|----------|
| **APAP** | Acetaminophen | Analjezik |
| **ASA** | Aspirin | Anti-inflamatuar |
| **BENZ** | Benzocaine | Lokal anestezik |
| **CAF** | Caffeine | Stimülan |
| **CIP** | Ciprofloxacin | Antibiyotik |
| **CIT** | Citalopram | Antidepresan |
| **DCF** | Diclofenac | Anti-inflamatuar |
| **FLX** | Fluoxetine | Antidepresan |
| **IBU** | Ibuprofen | Analjezik |
| **MTZ** | Metronidazole | Antibiyotik |
| **NPX** | Naproxen | Anti-inflamatuar |
| **NOR** | Norfloxacin | Antibiyotik |
| **OTC** | Oxytetracycline | Antibiyotik |
| **SA** | Salicylic Acid | Anti-inflamatuar |
| **SDZ** | Sulfadiazine | Antibiyotik |
| **SMR** | Sulfamerazine | Antibiyotik |
| **SMT** | Sulfamethazine | Antibiyotik |
| **SMX** | Sulfamethoxazole | Antibiyotik |
| **TC** | Tetracycline | Antibiyotik |
| **CBZ** | Carbamazepine | Antiepileptik |
| **PHE** | Phenol | Dezenfektan |

Her ilaç için solute parametreleri (E, S, A, B, V) model tarafından otomatik olarak eklenir.

---

## 🧪 Test ve Doğrulama

### Model Doğrulama Scripti

`test_model.py` scripti, modelin doğruluğunu kontrol etmek için kullanılır. Arayüzdeki tahminlerin model ile aynı olduğunu doğrular.

#### Kullanım:

```bash
python test_model.py
```

#### Ne Yapar?

1. **Model Yükleme**: `best_model.joblib` ve `best_model.meta.json` dosyalarını yükler
2. **Test Verisi**: Önceden tanımlanmış test parametreleri ile veri oluşturur
3. **Tahmin**: Model ile tahmin yapar
4. **Sonuç**: Tahmini qe değerini (mg/g) ekrana yazdırır

#### Örnek Çıktı:

```
============================================================
MODEL TEST SCRİPTİ
============================================================
1. Model yükleniyor...
✅ Model yüklendi. Feature sayısı: 23

2. Test verisi oluşturuluyor...
✅ Test verisi oluşturuldu:
   Agent/Sample(g/g): 1.0
   Soaking_Time(min): 240.0
   ...
   Target_Phar: CIP

3. Veri model için hazırlanıyor...
✅ Veri hazırlandı. Shape: (1, 23)
💡 Model pipeline'ı tüm preprocessing'i yapacak

4. Model tahmini yapılıyor...
🎯 Tahmini qe değeri: 125.456 mg/g
============================================================
```

#### Doğrulama Süreci:

1. `test_model.py` ile modelden tahmin alın
2. Aynı parametreleri Streamlit arayüzüne girin
3. Her iki sonucu karşılaştırın
4. Sonuçlar aynı olmalıdır ✅

---

## 🔧 Geliştirme

### Yeni İlaç Ekleme

1. `aqua_ml_app.py` dosyasında `solute_params` sözlüğüne yeni ilaç parametrelerini ekleyin:

```python
solute_params = {
    ...
    'YENİ_KOD': {'E': 0.00, 'S': 0.00, 'A': 0.00, 'B': 0.00, 'V': 0.00},
}
```

2. `ui_specs/drug_map.xlsx` dosyasına yeni ilaç kaydı ekleyin:
   - `Code`: İlaç kodu (örn: YENİ_KOD)
   - `Display_Name`: Görünen isim (örn: Yeni İlaç Adı)

### Model Güncelleme

1. Yeni model eğitimi sonrası:
   - `best_model.joblib` dosyasını değiştirin
   - `best_model.meta.json` dosyasını güncelleyin
   
2. Feature listesinin `meta.json` dosyasında doğru olduğundan emin olun

3. `test_model.py` ile doğrulama yapın

### Kaynak Kod Modülleri

`src/` klasöründeki modüller model geliştirme sürecinde kullanılır:

- **Data Processing**: `data_io.py`, `preprocessing.py`
- **Feature Engineering**: `features.py`
- **Model Training**: `pipelines.py`, `estimators.py`
- **Hyperparameter Tuning**: `tunning.py`
- **Evaluation**: `evaluation.py`, `evaluation1.py`
- **Configuration**: `config.py`

Detaylı bilgi için: `src/README.md`

---

## 📄 Dosya Açıklamaları

### Ana Dosyalar

#### `aqua_ml_app.py`
Ana Streamlit uygulaması. İki sekme içerir:
- **Tekil Giriş**: Form tabanlı tek tahmin
- **Excel Yükle**: Toplu tahmin

**Özellikler:**
- Modern UI tasarımı (Inter font, custom CSS)
- Otomatik solute parametre ekleme
- Percent → Molar dönüşümü (model pipeline'da)
- İnteraktif Plotly grafikleri
- Excel/CSV export desteği

#### `test_model.py`
Model doğrulama scripti. Modele direkt istek atarak arayüz sonuçlarını doğrular.

**Kullanım Senaryosu:**
- Model güncellemesi sonrası test
- Arayüz-model uyumluluk kontrolü
- Debugging ve hata ayıklama

#### `best_model.joblib`
Eğitilmiş makine öğrenmesi modeli (scikit-learn Pipeline).

**Pipeline İçeriği:**
1. Preprocessing (percent → molar dönüşümü, solute params ekleme)
2. Feature engineering
3. Scaling/encoding
4. ML modeli (örn: CatBoost, XGBoost)

#### `best_model.meta.json`
Model metadata dosyası.

**İçerik:**
```json
{
  "features": ["feature1", "feature2", ...],
  "model_type": "CatBoostRegressor",
  "trained_date": "2024-XX-XX",
  ...
}
```

#### `ui_specs/drug_map.xlsx`
İlaç kodu ve görünen isim eşleştirmeleri.

**Kolonlar:**
- `Code`: İlaç kodu (CIP, SMX, TC, vb.)
- `Display_Name`: Görünen isim (Ciprofloxacin, Sulfamethoxazole, vb.)

**Kullanım:**
- Selectbox'ta ilaç isimleri gösterme
- Kod → İsim ve İsim → Kod dönüşümleri

#### `requirements.txt`
Python bağımlılıkları listesi. Yorum satırları ile açıklanmıştır.

---

## ⚠️ Önemli Notlar

### Veri Doğrulama
- ✅ `Micropore_Volume` < `Total_Pore_Volume` olmalı
- ✅ `Target_Phar` ve `Activation_Atmosphere` zorunlu
- ⚠️ Eksik parametrelerle de tahmin yapılabilir ama sonuç etkilenebilir

### Model Pipeline
Model, aşağıdaki işlemleri otomatik yapar:
1. Element yüzdelerini (C%, H%, O%, N%, S%) molar değerlere çevirir
2. Seçilen ilaca göre solute parametrelerini (E, S, A, B, V) ekler
3. Kategorik değişkenleri (Activation_Atmosphere) encode eder
4. Gerekli scaling ve transformasyonları uygular

**Kullanıcı ham değerleri girmelidir!** Model preprocessing'i kendisi yapacaktır.

### Performans
- İlk çalıştırmada model yüklenir (`@st.cache_resource`)
- Duyarlılık analizleri her parametrede 20-40 tahmin yapar
- Büyük Excel dosyaları (1000+ satır) işlem süresini artırabilir

---

## 📞 Destek ve İletişim

**Proje Hakkında:**
- Model: Gradient Boosting tabanlı adsorpsiyon kapasitesi tahmini
- Framework: Streamlit + scikit-learn
- Veri: Farmasötik kirleticiler ve aktif karbon adsorpsiyonu

**Versiyon:** 1.0  
**Son Güncelleme:** 2024

---

## 📜 Lisans

Bu proje araştırma amaçlı geliştirilmiştir.

---

## 🎨 Ekran Görüntüleri

### Ana Sayfa
- Modern ve temiz arayüz
- İnter font ailesi ile profesyonel görünüm
- Başlık görseli ve açıklama

### Tekil Giriş Sekmesi
- 4 kolonlu responsive form düzeni
- Slider ve number input widget'ları
- Ilaç seçimi için searchable dropdown
- Aktivasyon atmosferi radio button'ları

### Sonuç Ekranı
- Tahmini qe değeri (mg/g)
- Antibiyotik karşılaştırma bar chart
- 2 kolonlu duyarlılık analizleri:
  - Sol: Sentez koşulları (ajan oranı, emdirim, aktivasyon)
  - Sağ: Proses koşulları (konsantrasyon, pH, sıcaklık, dozaj, temas süresi)
- İnteraktif Plotly line chart'ları

### Excel Yükleme Sekmesi
- Şablon indirme butonları (CSV & Excel)
- Drag-drop dosya yükleme
- Sonuç tablosu önizleme
- İndirme butonları (CSV & Excel)

---

**🎯 Aqua-ML ile çevre dostu çözümler için akıllı tahminler!**

