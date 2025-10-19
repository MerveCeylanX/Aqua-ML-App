# aqua_ml_app.py — Aqua-ML: Aktif Karbon Adsorpsiyon Kapasitesi Tahmin Uygulaması
# -*- coding: utf-8 -*-
"""
Aqua-ML: Makine Öğrenmesi Tabanlı Adsorpsiyon Kapasitesi Tahmin Sistemi

Bu uygulama, aktif karbon ile farmasötik kirleticilerin adsorpsiyonunu tahmin eden
bir makine öğrenmesi arayüzüdür.

Özellikler:
- Tekil parametre girişi ile tahmin
- Excel/CSV toplu veri yükleme ve tahmin
- 21 farklı farmasötik kirletici desteği
- Sentez koşulları, adsorban özellikleri ve proses parametreleri ile tahmin
- Interaktif duyarlılık analizleri (pH, sıcaklık, konsantrasyon, vb.)
- Antibiyotik karşılaştırma grafikleri

Model Girdileri:
- Sentez Koşulları: Aktivasyon sıcaklığı, süresi, atmosfer, ajan oranı, vb.
- Adsorban Özellikleri: BET yüzey alanı, gözenek hacmi, element analizi, vb.
- Proses Koşulları: pH, sıcaklık, konsantrasyon, dozaj, temas süresi, vb.
- Hedef İlaç: APAP, CIP, SMX, TC, OTC ve 16 farmasötik daha

Gereksinimler:
- best_model.joblib: Eğitilmiş ML modeli
- best_model.meta.json: Model metadata ve feature listesi
- ui_specs/drug_map.xlsx: İlaç adı ve kod eşleştirmeleri
"""

from pathlib import Path
import json
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# (Yalnızca Plotly'nin "keyword args deprecated" FutureWarning'ini sustur)
warnings.filterwarnings(
    "ignore",
    message=r"The keyword arguments have been deprecated.*Use config.*Plotly",
    category=FutureWarning
)

# -------------------------------------------------
# SAYFA AYARLARI (ilk satırlarda olmalı)
# -------------------------------------------------
st.set_page_config(
    page_title="Aqua-ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Plotly yardımcı (config=... standardına geçirir)
# -------------------------------------------------
def show_plotly(fig, config=None, **maybe_old_kwargs):
    """
    Streamlit 1.30+ için doğru kullanım:
      - Plotly ayarları config={} içinde verilmeli.
      - Eski keyword'ler (**kwargs) gelirse config'e aktarılır.
    """
    merged = dict(config or {})
    merged.update(maybe_old_kwargs)
    st.plotly_chart(fig, config=merged, use_container_width=True)

# -------------------------------------------------
# TEMA / STİL
# -------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Family */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }

/**************** Base ****************/
.stApp {
    background:#f7fafc;
    max-width: 100% !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 100% !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, h4, h5 {
    color:#0f172a;
    font-weight: 600;
}
/* butonlar */
.stButton>button, .stDownloadButton>button {
  background:#2563eb; color:#fff; border-radius:10px; border:0; padding:0.6rem 1rem;
  font-weight: 500;
}
.stButton>button:hover, .stDownloadButton>button:hover { filter:brightness(0.95); }
/* expander */
.stExpander > summary {
  background:#e2e8f0 !important; color:#0f172a; border:1px solid #cbd5e1;
  border-radius:12px; padding:12px 16px;
  font-weight: 500; font-size: 16px;
  list-style: none !important; cursor: pointer; position: relative; display: block !important;
}
.stExpander > summary::marker,
.stExpander > summary::-webkit-details-marker,
.stExpander > summary::after { display: none !important; content: none !important; }
.stExpander > summary::before {
  content: ''; position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
  width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 8px solid #0f172a; transition: transform 0.2s ease;
}
.stExpander[open] > summary::before { transform: translateY(-50%) rotate(180deg); }
.stExpander div[role='region'] {
  background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; padding:12px; margin-top:6px;
}
/* Hide any text content that might be showing */
.stExpander > summary > * { display: none !important; }
.stExpander > summary { text-indent: 0 !important; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background:#e2e8f0; border-radius: 10px; padding: 8px 12px; font-weight: 500;
}
/* help text */
.small-note { font-size:0.9rem; color:#334155; }
/* Grafikler için geniş alan */
.plotly { width: 100% !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# HEADER GÖRSELİ
# -------------------------------------------------
if Path("header.jpeg").exists():
    st.image(
        "header.jpeg",
        use_container_width=True,
        caption="A real-world adsorption dataset and ML model for qe prediction",
    )

st.title("Aqua-ML")
st.markdown("""
<div style="margin-top: -15px; margin-bottom: 30px;">
    <p style="font-size: 1.1rem; color: #64748b; font-weight: 400; letter-spacing: 0.5px;">
        Çevre dostu çözümler için akıllı tahmin motoru
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL / META YÜKLEME
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    """best_model.joblib ve best_model.meta.json dosyalarını yükle."""
    try:
        pipe = joblib.load("best_model.joblib")  # fit edilmiş sklearn Pipeline
    except FileNotFoundError:
        st.error("Model dosyası bulunamadı: best_model.joblib")
        st.stop()
    except Exception as e:
        st.error(f"Model yüklenemedi: {type(e).__name__}: {e}")
        st.stop()

    try:
        with open("best_model.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            feats = meta.get("features", [])
    except FileNotFoundError:
        st.error("best_model.meta.json bulunamadı. Lütfen meta dosyasını ekleyin.")
        st.stop()
    except Exception as e:
        st.error(f"Meta okunamadı: {type(e).__name__}: {e}")
        st.stop()

    if not feats:
        st.error("Meta içinde 'features' anahtarı boş görünüyor.")
        st.stop()
    return pipe, feats

@st.cache_data
def load_drug_mapping():
    """İlaç haritasını yükle."""
    try:
        df = pd.read_excel("ui_specs/drug_map.xlsx")
        return df
    except Exception as e:
        st.error(f"İlaç haritası yüklenemedi: {type(e).__name__}: {e}")
        st.stop()

pipe, FEATURES = load_artifacts()
drug_mapping = load_drug_mapping()

# Solute parametreleri (E, S, A, B, V değerleri)
solute_params = {
    'APAP': {'E': 1.16, 'S': 1.35, 'A': 0.49, 'B': 0.20, 'V': 1.1566},
    'ASA': {'E': 1.13, 'S': 1.12, 'A': 0.72, 'B': 0.31, 'V': 1.0000},
    'BENZ': {'E': 0.60, 'S': 0.90, 'A': 0.00, 'B': 0.39, 'V': 1.0000},
    'CAF': {'E': 1.50, 'S': 1.33, 'A': 0.00, 'B': 0.64, 'V': 1.0000},
    'CIP': {'E': 1.98, 'S': 2.50, 'A': 0.05, 'B': 2.39, 'V': 2.2724},
    'CIT': {'E': 1.77, 'S': 1.96, 'A': 0.00, 'B': 1.12, 'V': 2.0000},
    'DCF': {'E': 1.28, 'S': 1.44, 'A': 0.00, 'B': 0.68, 'V': 1.0000},
    'FLX': {'E': 1.77, 'S': 1.96, 'A': 0.00, 'B': 1.12, 'V': 2.0000},
    'IBU': {'E': 0.62, 'S': 0.79, 'A': 0.00, 'B': 0.40, 'V': 1.0000},
    'MTZ': {'E': 1.28, 'S': 1.44, 'A': 0.00, 'B': 0.68, 'V': 1.0000},
    'NPX': {'E': 1.51, 'S': 2.02, 'A': 0.60, 'B': 0.67, 'V': 1.7821},
    'NOR': {'E': 1.98, 'S': 2.50, 'A': 0.05, 'B': 2.39, 'V': 2.2724},
    'OTC': {'E': 3.60, 'S': 3.05, 'A': 1.65, 'B': 3.50, 'V': 3.1579},
    'SA': {'E': 0.90, 'S': 0.85, 'A': 0.73, 'B': 0.37, 'V': 0.9904},
    'SDZ': {'E': 2.08, 'S': 2.55, 'A': 0.65, 'B': 1.37, 'V': 1.7225},
    'SMR': {'E': 2.10, 'S': 2.65, 'A': 0.65, 'B': 1.42, 'V': 1.8634},
    'SMT': {'E': 2.13, 'S': 2.53, 'A': 0.59, 'B': 1.53, 'V': 2.0043},
    'SMX': {'E': 1.89, 'S': 2.23, 'A': 0.58, 'B': 1.29, 'V': 1.7244},
    'TC': {'E': 3.50, 'S': 3.60, 'A': 1.35, 'B': 3.29, 'V': 3.0992},
    'CBZ': {'E': 2.15, 'S': 1.90, 'A': 0.50, 'B': 1.15, 'V': 1.8106},
    'PHE': {'E': 0.60, 'S': 0.90, 'A': 0.00, 'B': 0.39, 'V': 1.0000}
}

# -------------------------------------------------
# FORM GRUPLARI
# -------------------------------------------------
synthesis = [
    "Agent/Sample(g/g)", "Soaking_Time(min)", "Soaking_Temp(K)",
    "Activation_Time(min)", "Activation_Temp(K)", "Activation_Heating_Rate (K/min)",
]
adsorbent = [
    "BET_Surface_Area(m2/g)", "Total_Pore_Volume(cm3/g)", "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)", "pHpzc", "C_percent", "H_percent", "O_percent", "N_percent", "S_percent",
]
process_ = [
    "Solution_pH", "Temperature(K)", "Initial_Concentration(mg/L)",
    "Dosage(g/L)", "Contact_Time(min)", "Agitation_speed(rpm)",
]
solute = ["E", "S", "A", "B", "V"]
categorical = ["Activation_Atmosphere"]
target_phar = ["Target_Phar"]

# Slider tanımları (min, max, step)
SLIDER_SPEC = {
    "Solution_pH": (0.5, 13.5, 0.1),
    "Temperature(K)": (290.0, 340.0, 1.0),
    "Dosage(g/L)": (0.05, 18.0, 0.1),
    "Contact_Time(min)": (0.0, 6000.0, 1.0),
    "Initial_Concentration(mg/L)": (0.0, 1000.0, 1.0),
    "Agitation_speed(rpm)": (0.0, 700.0, 10.0),
    "C_percent": (0.0, 100.0, 0.1),
    "H_percent": (0.0, 10.0, 0.1),
    "O_percent": (0.0, 50.0, 0.1),
    "N_percent": (0.0, 20.0, 0.1),
    "S_percent": (0.0, 5.0, 0.1),
}

# Number input (oklu kutu) tanımları (min, max, step, default)
NUMBER_INPUT_SPEC = {
    "BET_Surface_Area(m2/g)": (0.0, 3000.0, 10.0, None),
    "Total_Pore_Volume(cm3/g)": (0.0, 5.0, 0.01, None),
    "Micropore_Volume(cm3/g)": (0.0, 2.0, 0.01, None),
    "Average_Pore_Diameter(nm)": (0.0, 50.0, 0.1, None),
    "pHpzc": (0.0, 14.0, 0.1, None),
    "Agent/Sample(g/g)": (0.0, 10.0, 0.01, None),
    "Soaking_Time(min)": (0.0, 6000.0, 5.0, None),
    "Soaking_Temp(K)": (273.0, 500.0, 1.0, None),
    "Activation_Time(min)": (0.0, 360.0, 5.0, None),
    "Activation_Temp(K)": (550.0, 1200.0, 10.0, None),
    "Activation_Heating_Rate (K/min)": (0.0, 50.0, 1.0, None),
}

# Özel başlangıç değerleri (orta değer yerine)
SLIDER_DEFAULTS = {
    "H_percent": 1.0,
    "N_percent": 1.0,
    "S_percent": 1.0,
}

def slider_default(lo: float, hi: float, name: str = None) -> float:
    if name and name in SLIDER_DEFAULTS:
        return float(SLIDER_DEFAULTS[name])
    return float((float(lo) + float(hi)) / 2.0)

# -------------------------------------------------
# YARDIMCILAR
# -------------------------------------------------
def _to_float_or_none(txt: str):
    """Boş stringi None, sayı metnini float yapar; aksi halde None döndürür."""
    if txt is None:
        return None
    txt = str(txt).strip().replace(",", ".")
    if txt == "":
        return None
    try:
        return float(txt)
    except Exception:
        return None

def render_block(title: str, fields: list[str], values: dict):
    percent_fields = ["C_percent", "H_percent", "O_percent", "N_percent", "S_percent"]
    present = [f for f in fields if f in FEATURES or f == "Target_Phar" or f in percent_fields]

    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <h3 style="margin:0; color:#1e293b; font-weight: 600; font-size: 18px; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; display: inline-block;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, name in enumerate(present):
        c = cols[i % 4]
        display_name = name.replace("_", " ")
        if "(" in display_name and not " (" in display_name:
            display_name = display_name.replace("(", " (")

        element_map = {
            "C percent": "Carbon % (wt.)",
            "H percent": "Hydrogen % (wt.)",
            "O percent": "Oxygen % (wt.)",
            "N percent": "Nitrogen % (wt.)",
            "S percent": "Sulfur % (wt.)"
        }
        if display_name in element_map:
            display_name = element_map[display_name]

        display_name = display_name.replace("m2/g", "m²/g").replace("cm3/g", "cm³/g")

        if name in NUMBER_INPUT_SPEC:
            lo, hi, stp, default_val = NUMBER_INPUT_SPEC[name]
            values[name] = c.number_input(
                display_name,
                min_value=float(lo),
                max_value=float(hi),
                step=float(stp),
                value=default_val,
                help=f"Ok tuşları ile {stp} artırıp azaltabilirsiniz (virgül veya nokta kullanabilirsiniz)"
            )
        elif name in SLIDER_SPEC:
            lo, hi, stp = SLIDER_SPEC[name]
            values[name] = c.slider(
                display_name,
                min_value=float(lo),
                max_value=float(hi),
                step=float(stp),
                value=slider_default(lo, hi, name),
            )
        elif name == "Target_Phar":
            drug_options = drug_mapping['Display_Name'].tolist()
            selected_display = c.selectbox(
                "Target Pharmaceutical",
                options=drug_options,
                index=None,
                placeholder="İlaç seçiniz... (örn: Ciprofloxacin, Norfloxacin)",
                help="İlaç adını yazarak arama yapabilirsiniz. Seçilen ilaca göre solute parametreleri otomatik eklenir."
            )
            if selected_display:
                selected_code = drug_mapping[drug_mapping['Display_Name'] == selected_display]['Code'].iloc[0]
                values[name] = selected_code
                c.success(f"✅ Seçilen: **{selected_display}** ({selected_code})")
            else:
                values[name] = None
        elif name == "Activation_Atmosphere":
            atmosphere_options = ["N2", "Air", "SG"]
            atmosphere_labels = ["Nitrogen (N₂)", "Air", "Self-generated atmosphere"]
            selected_atmosphere = c.radio(
                display_name,
                options=atmosphere_options,
                format_func=lambda x: atmosphere_labels[atmosphere_options.index(x)],
                index=None,
                horizontal=True,
                help="Aktivasyon sırasında kullanılan atmosfer türünü seçiniz"
            )
            values[name] = selected_atmosphere
        else:
            values[name] = c.text_input(display_name, value="", placeholder="değer giriniz")

def align_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Veriyi model için hazırla - model kendi preprocessing'ini yapacak"""
    return df

# -------------------------------------------------
# SEKMELER
# -------------------------------------------------
tab1, tab2 = st.tabs(["Tekil Giriş", "Excel Yükle"])

# -------- Tekil Giriş --------
with tab1:
    st.subheader("Tekil Giriş")
    st.markdown('<div class="small-note">💡 İpucu: Parametreleri doldurun ve \'Tahmin Et\' butonuna basın.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.5])
    vals: dict[str, float | str] = {}

    with col_left:
        st.markdown("### 📝 Model Girdileri")
        with st.form("single"):
            render_block("Sentez Koşulları", synthesis, vals)
            render_block("Adsorban Özellikleri", adsorbent, vals)
            render_block("Proses Koşulları", process_, vals)

            st.markdown("""
            <div style="margin-bottom:20px;">
                <h3 style="margin:0; color:#1e293b; font-weight: 600; font-size: 18px; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; display: inline-block;">🎯 Hedef İlaç</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**İlaç Seçimi:** Seçilen ilaca göre solute parametreleri (E, S, A, B, V) otomatik olarak eklenir.")

            drug_options = drug_mapping['Display_Name'].tolist()
            selected_display = st.selectbox(
                "Target Pharmaceutical",
                options=drug_options,
                index=None,
                placeholder="İlaç seçiniz... (örn: Ciprofloxacin, Norfloxacin)",
                help="İlaç adını yazarak arama yapabilirsiniz.",
                key="target_phar_select"
            )

            if selected_display:
                selected_code = drug_mapping[drug_mapping['Display_Name'] == selected_display]['Code'].iloc[0]
                vals["Target_Phar"] = selected_code
                st.success(f"✅ **Seçilen:** {selected_display} ({selected_code})")
            else:
                vals["Target_Phar"] = None
                st.warning("⚠️ Lütfen bir ilaç seçiniz.")

            st.markdown("""
            <div style="margin-bottom:20px;">
                <h3 style="margin:0; color:#1e293b; font-weight: 600; font-size: 18px; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; display: inline-block;">🌡️ Aktivasyon Atmosferi</h3>
            </div>
            """, unsafe_allow_html=True)

            atmosphere_options = ["N2", "Air", "SG"]
            atmosphere_labels = ["Nitrogen (N₂)", "Air", "Self-generated atmosphere"]
            selected_atmosphere = st.radio(
                "Aktivasyon sırasında kullanılan atmosfer türünü seçiniz",
                options=atmosphere_options,
                format_func=lambda x: atmosphere_labels[atmosphere_options.index(x)],
                index=None,
                horizontal=True,
                help="Aktivasyon atmosferi seçimi"
            )
            vals["Activation_Atmosphere"] = selected_atmosphere

            submitted = st.form_submit_button("🎯 Tahmin Et", use_container_width=True)

    with col_right:
        st.markdown("### 📊 Tahmin Sonuçları")

    if submitted:
        row: dict[str, float | str] = {}
        for k in [*synthesis, *adsorbent, *process_, *categorical, *target_phar]:
            v = vals.get(k, None)
            if isinstance(v, str) and k not in ["Target_Phar", "Activation_Atmosphere"]:
                v = _to_float_or_none(v)
            row[k] = v

        X = pd.DataFrame([row])
        X = align_and_cast(X)
        base_params = {k: v for k, v in row.items() if k != "Target_Phar"}

        with col_right:
            if not vals.get("Target_Phar"):
                st.error("⚠️ Lütfen bir ilaç seçiniz!")
                st.stop()

            if not vals.get("Activation_Atmosphere"):
                st.error("⚠️ Lütfen aktivasyon atmosferini seçiniz!")
                st.stop()

            missing_fields = []
            element_map = {
                "C percent": "Carbon % (wt.)",
                "H percent": "Hydrogen % (wt.)",
                "O percent": "Oxygen % (wt.)",
                "N percent": "Nitrogen % (wt.)",
                "S percent": "Sulfur % (wt.)"
            }
            for k in [*synthesis, *adsorbent, *process_]:
                v = row.get(k, None)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    readable_name = k.replace("_", " ")
                    if "(" in readable_name and not " (" in readable_name:
                        readable_name = readable_name.replace("(", " (")
                    if readable_name in element_map:
                        readable_name = element_map[readable_name]
                    readable_name = readable_name.replace("m2/g", "m²/g").replace("cm3/g", "cm³/g")
                    missing_fields.append(readable_name)

            if missing_fields:
                st.warning("⚠️ **Uyarı:** Bazı alanlar boş. Model tahmin yapacak ancak sonuç eksik veriden etkilenebilir.")

            total_pore = row.get("Total_Pore_Volume(cm3/g)", None)
            micro_pore = row.get("Micropore_Volume(cm3/g)", None)
            if total_pore is not None and micro_pore is not None:
                if not pd.isna(total_pore) and not pd.isna(micro_pore):
                    if micro_pore > total_pore:
                        st.error("⚠️ **Hata:** Micropore Volume, Total Pore Volume'den büyük olamaz!")
                        st.stop()

            selected_display = drug_mapping[drug_mapping['Code'] == vals["Target_Phar"]]['Display_Name'].iloc[0]
            st.info(f"🎯 **Seçilen İlaç:** {selected_display} ({vals['Target_Phar']})")

            try:
                yhat = float(pipe.predict(X)[0])
                st.success(f"🎯 **Model Tahmini:** {yhat:.3f} mg/g")

                # ==== Plotly: karşılaştırma ve duyarlılık grafikleri ====
                import plotly.express as px

                # 1) Antibiyotik karşılaştırması
                st.markdown("---")
                st.markdown("### 📊 Antibiyotik Karşılaştırması")
                st.caption("Girdiğiniz parametreler sabit tutularak, farklı antibiyotikler için adsorpsiyon kapasitesi tahminleri karşılaştırılır.")

                comparison_results = []
                for drug_code, _ in solute_params.items():
                    test_row = base_params.copy()
                    test_row["Target_Phar"] = drug_code
                    test_df = pd.DataFrame([test_row])
                    test_df = align_and_cast(test_df)
                    try:
                        pred_qe = float(pipe.predict(test_df)[0])
                        drug_name = drug_mapping[drug_mapping['Code'] == drug_code]['Display_Name'].iloc[0] if not drug_mapping[drug_mapping['Code'] == drug_code].empty else drug_code
                        comparison_results.append({'Drug_Code': drug_code, 'Drug_Name': drug_name, 'Predicted_qe': pred_qe})
                    except Exception as e:
                        st.warning(f"⚠️ {drug_code} için tahmin yapılamadı: {e}")

                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results).sort_values('Predicted_qe', ascending=False)
                    fig1 = px.bar(
                        comparison_df,
                        x='Drug_Name',
                        y='Predicted_qe',
                        color='Predicted_qe',
                        color_continuous_scale='Turbo',
                        labels={'Drug_Name': 'Antibiyotik', 'Predicted_qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'}
                    )
                    max_qe = comparison_df['Predicted_qe'].max()
                    fig1.update_layout(
                        xaxis_tickangle=-45,
                        height=480,
                        showlegend=False,
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='white',
                        margin=dict(l=60, r=60, t=30, b=60),
                        font=dict(size=12, family='Inter, sans-serif'),
                        xaxis=dict(
                            showgrid=False, showline=True, linewidth=2, linecolor='#2c3e50', mirror=True,
                            title=dict(text='Antibiyotik', font=dict(size=14, color='#2c3e50'))
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor='#e0e0e0', gridwidth=1, showline=True, linewidth=2, linecolor='#2c3e50', mirror=True,
                            title=dict(text='Adsorpsiyon Kapasitesi, qe (mg/g)', font=dict(size=14, color='#2c3e50')),
                            range=[0, max_qe * 1.1]
                        ),
                        coloraxis_colorbar=dict(title_text='qe (mg/g)', thickness=15, len=0.7)
                    )
                    fig1.update_traces(marker=dict(line=dict(width=0.5, color='white')))
                    show_plotly(fig1)

                # ==== Duyarlılık Analizleri ====
                st.markdown("---")
                st.markdown("### 📈 Duyarlılık Analizleri")
                st.info("ℹ️ Tüm parametreler model girdilerindeki değerinde sabit tutulup, sadece analiz edilen parametre değiştirilerek adsorpsiyon kapasitesindeki değişim incelenir.")

                col_synthesis, col_process = st.columns([1, 1])

                with col_synthesis:
                    st.markdown("""
                    <div style="margin: 20px 0;">
                        <h2 style="color:#1e293b; font-weight: 600; font-size: 20px; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; display: inline-block;">⚗️ Sentez Koşulları</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    # Agent/Sample ratio
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">🧪 Ajan/Numune Oranı Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Aktivasyon ajanı oranının adsorpsiyon kapasitesi üzerindeki etkisi")
                    agent_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
                    agent_results = []
                    for ratio in agent_ratios:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Agent/Sample(g/g)"] = ratio
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            agent_results.append({'Agent_Ratio': ratio, 'qe': pred_qe})
                        except:
                            pass
                    if agent_results:
                        import plotly.express as px
                        agent_df = pd.DataFrame(agent_results)
                        fig7 = px.line(
                            agent_df, x='Agent_Ratio', y='qe',
                            labels={'Agent_Ratio': 'Ajan/Numune Oranı (g/g)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#e74c3c']
                        )
                        fig7.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig7.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig7)

                    # Soaking Time
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">⏰ Emdirim Süresi Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Emdirim süresinin adsorpsiyon kapasitesi üzerindeki etkisi")
                    soaking_times = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 720, 900, 1200, 1500, 1800, 2000]
                    soaking_results = []
                    for time in soaking_times:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Soaking_Time(min)"] = time
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            soaking_results.append({'Soaking_Time': time, 'qe': pred_qe})
                        except:
                            pass
                    if soaking_results:
                        soaking_df = pd.DataFrame(soaking_results)
                        fig8 = px.line(
                            soaking_df, x='Soaking_Time', y='qe',
                            labels={'Soaking_Time': 'Emdirim Süresi (dk)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#9b59b6']
                        )
                        fig8.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig8.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig8)

                    # Activation Time
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">⏲️ Aktivasyon Süresi Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Aktivasyon süresinin adsorpsiyon kapasitesi üzerindeki etkisi")
                    activation_times = [30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 270, 300, 330, 360]
                    act_time_results = []
                    for time in activation_times:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Activation_Time(min)"] = time
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            act_time_results.append({'Activation_Time': time, 'qe': pred_qe})
                        except:
                            pass
                    if act_time_results:
                        act_time_df = pd.DataFrame(act_time_results)
                        fig10 = px.line(
                            act_time_df, x='Activation_Time', y='qe',
                            labels={'Activation_Time': 'Aktivasyon Süresi (dk)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#f39c12']
                        )
                        fig10.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig10.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig10)

                    # Activation Temperature
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">🔥 Aktivasyon Sıcaklığı Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Aktivasyon sıcaklığının adsorpsiyon kapasitesi üzerindeki etkisi")
                    activation_temps = [550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200]
                    act_temp_results = []
                    for temp in activation_temps:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Activation_Temp(K)"] = temp
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            act_temp_results.append({'Activation_Temp': temp, 'qe': pred_qe})
                        except:
                            pass
                    if act_temp_results:
                        act_temp_df = pd.DataFrame(act_temp_results)
                        fig9 = px.line(
                            act_temp_df, x='Activation_Temp', y='qe',
                            labels={'Activation_Temp': 'Aktivasyon Sıcaklığı (K)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#3498db']
                        )
                        fig9.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig9.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig9)

                with col_process:
                    st.markdown("""
                    <div style="margin: 20px 0;">
                        <h2 style="color:#1e293b; font-weight: 600; font-size: 20px; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; display: inline-block;">🔬 Proses Koşulları</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    # Konsantrasyon
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">📈 Konsantrasyon Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Başlangıç konsantrasyonunun adsorpsiyon kapasitesi üzerindeki etkisi")
                    concentrations = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
                    conc_results = []
                    for conc in concentrations:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Initial_Concentration(mg/L)"] = conc
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            conc_results.append({'Concentration': conc, 'qe': pred_qe})
                        except:
                            pass
                    if conc_results:
                        conc_df = pd.DataFrame(conc_results)
                        fig2 = px.line(
                            conc_df, x='Concentration', y='qe',
                            labels={'Concentration': 'Başlangıç Konsantrasyonu (mg/L)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#1abc9c']
                        )
                        fig2.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig2.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig2)

                    # Sıcaklık
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">🌡️ Sıcaklık Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Çözelti sıcaklığının adsorpsiyon kapasitesi üzerindeki etkisi")
                    temperatures = [290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340]
                    temp_results = []
                    for temp in temperatures:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Temperature(K)"] = temp
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            temp_results.append({'Temperature': temp, 'qe': pred_qe})
                        except:
                            pass
                    if temp_results:
                        temp_df = pd.DataFrame(temp_results)
                        fig3 = px.line(
                            temp_df, x='Temperature', y='qe',
                            labels={'Temperature': 'Sıcaklık (K)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#e67e22']
                        )
                        fig3.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig3.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig3)

                    # pH
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">🧪 pH Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Çözelti pH'ının adsorpsiyon kapasitesi üzerindeki etkisi")
                    ph_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
                    ph_results = []
                    for ph in ph_values:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Solution_pH"] = ph
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            ph_results.append({'pH': ph, 'qe': pred_qe})
                        except:
                            pass
                    if ph_results:
                        ph_df = pd.DataFrame(ph_results)
                        fig4 = px.line(
                            ph_df, x='pH', y='qe',
                            labels={'pH': 'pH', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#27ae60']
                        )
                        fig4.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig4.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig4)

                    # Dozaj
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">⚖️ Dozaj Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Adsorban dozajının adsorpsiyon kapasitesi üzerindeki etkisi")
                    dosages = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
                    dosage_results = []
                    for dosage in dosages:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Dosage(g/L)"] = dosage
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            dosage_results.append({'Dosage': dosage, 'qe': pred_qe})
                        except:
                            pass
                    if dosage_results:
                        dosage_df = pd.DataFrame(dosage_results)
                        fig5 = px.line(
                            dosage_df, x='Dosage', y='qe',
                            labels={'Dosage': 'Dozaj (g/L)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#c0392b']
                        )
                        fig5.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig5.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig5)

                    # Contact Time
                    st.markdown("""
                    <div style="margin: 15px 0;">
                        <h3 style="color:#374151; font-weight: 500; font-size: 16px; margin: 0;">⏱️ Temas Süresi Duyarlılığı</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Temas süresinin adsorpsiyon kapasitesi üzerindeki etkisi")
                    contact_times = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 600, 720, 900, 1200, 1500, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000]
                    time_results = []
                    for time in contact_times:
                        test_row = base_params.copy()
                        test_row["Target_Phar"] = vals["Target_Phar"]
                        test_row["Contact_Time(min)"] = time
                        test_df = pd.DataFrame([test_row])
                        test_df = align_and_cast(test_df)
                        try:
                            pred_qe = float(pipe.predict(test_df)[0])
                            time_results.append({'Contact_Time': time, 'qe': pred_qe})
                        except:
                            pass
                    if time_results:
                        time_df = pd.DataFrame(time_results)
                        fig6 = px.line(
                            time_df, x='Contact_Time', y='qe',
                            labels={'Contact_Time': 'Temas Süresi (dk)', 'qe': 'Adsorpsiyon Kapasitesi, qe (mg/g)'},
                            line_shape='spline', markers=True, color_discrete_sequence=['#8e44ad']
                        )
                        fig6.update_layout(
                            height=350, showlegend=False, plot_bgcolor='#f8f9fa', paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=30, b=50),
                            font=dict(size=11, family='Inter, sans-serif'),
                            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True),
                            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', showline=True, linewidth=2, linecolor='#2c3e50', mirror=True)
                        )
                        fig6.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1.5, color='white')))
                        show_plotly(fig6)

            except Exception as e:
                error_msg = str(e)
                if "NaN" in error_msg or "missing" in error_msg.lower() or "nan" in error_msg.lower():
                    st.error("⚠️ Bazı gerekli alanlar eksik! Lütfen tüm parametreleri doldurunuz.")
                elif "cat_feature" in error_msg.lower() or "categorical" in error_msg.lower():
                    st.error("⚠️ Kategorik değişken hatası! Lütfen aktivasyon atmosferini seçtiğinizden emin olun.")
                else:
                    st.error(f"⚠️ Tahmin sırasında bir hata oluştu. Lütfen tüm alanları kontrol ediniz.")
                    with st.expander("Teknik Detaylar (Geliştiriciler için)"):
                        import traceback
                        st.code(traceback.format_exc())

# -------- Excel Yükle --------
with tab2:
    st.subheader("Excel (.xlsx) Yükle")
    st.markdown('<div class="small-note">💡 İpucu: Önce örnek şablonu indirin, parametrelerinizi doldurup yükleyin.</div>', unsafe_allow_html=True)
    st.markdown("")

    template_data = {
        'Agent/Sample(g/g)': [1.0, 2.0],
        'Soaking_Time(min)': [120, 180],
        'Soaking_Temp(K)': [373, 373],
        'Activation_Time(min)': [60, 90],
        'Activation_Temp(K)': [700, 800],
        'Activation_Heating_Rate (K/min)': [5, 10],
        'BET_Surface_Area(m2/g)': [1000, 1200],
        'Total_Pore_Volume(cm3/g)': [0.5, 0.6],
        'Micropore_Volume(cm3/g)': [0.3, 0.4],
        'Average_Pore_Diameter(nm)': [2.5, 3.0],
        'pHpzc': [7.0, 7.5],
        'C_percent': [80, 85],
        'H_percent': [1.0, 1.5],
        'O_percent': [15, 12],
        'N_percent': [1.0, 1.0],
        'S_percent': [1.0, 0.5],
        'Solution_pH': [7.0, 6.5],
        'Temperature(K)': [298, 308],
        'Initial_Concentration(mg/L)': [100, 150],
        'Dosage(g/L)': [1.0, 1.5],
        'Contact_Time(min)': [60, 120],
        'Agitation_speed(rpm)': [200, 250],
        'Target_Phar': ['CIP', 'SMX'],
        'Activation_Atmosphere': ['N2', 'Air']
    }
    template_df = pd.DataFrame(template_data)

    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Şablon İndir (CSV)",
            data=template_csv,
            file_name="aquaml_template.csv",
            mime="text/csv",
            help="CSV şablonunu indirin",
            use_container_width=True
        )
    with col_temp2:
        template_buffer = BytesIO()
        with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Template')
        template_buffer.seek(0)
        st.download_button(
            label="📥 Şablon İndir (Excel)",
            data=template_buffer,
            file_name="aquaml_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Excel şablonunu indirin",
            use_container_width=True
        )

    st.markdown("---")
    file = st.file_uploader("Dosya seç", type=["xlsx", "csv"])

    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Dosya okunamadı: {type(e).__name__}: {e}")
            st.stop()

        st.info("💡 **Not:** Model pipeline'ı tüm preprocessing'i yapacak (percent→molar, solute params, vs.)")

        required_cols = ["Target_Phar", "Activation_Atmosphere"]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error(f"Eksik gerekli kolon(lar): {missing_required}")
            st.info("Excel'inizde Target_Phar ve Activation_Atmosphere kolonları olmalı.")
            st.stop()

        extra = [c for c in df.columns if c not in required_cols and not c.startswith(('Agent', 'Soaking', 'Activation', 'BET', 'Total', 'Micropore', 'Average', 'pHpzc', 'C_percent', 'H_percent', 'O_percent', 'N_percent', 'S_percent', 'Solution', 'Temperature', 'Initial', 'Dosage', 'Contact', 'Agitation'))]
        if extra:
            st.warning(f"Tanınmayan kolon(lar) yoksayılacak: {extra}")

        X = align_and_cast(df)

        try:
            yhat = pipe.predict(X)
        except Exception as e:
            st.error(f"Tahmin sırasında hata: {type(e).__name__}: {e}")
            st.stop()

        out = df.copy()
        out["Pred_qe"] = yhat
        st.success(f"✅ {len(out)} satır tahmin edildi.")
        st.dataframe(out.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Sonuçları İndir (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="aquaml_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                out.to_excel(writer, index=False, sheet_name='Predictions')
            buffer.seek(0)
            st.download_button(
                label="📥 Sonuçları İndir (Excel)",
                data=buffer,
                file_name="aquaml_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# -------------------------------------------------
# İPUÇLARI - Sayfa Altı
# -------------------------------------------------
st.markdown("---")
st.markdown("### 💡 Kullanım Kılavuzu")

col_guide1, col_guide2 = st.columns(2)

with col_guide1:
    st.markdown("""
    **📝 Tekil Giriş:**
    - İlaç ve atmosfer seçimi zorunludur.
    - Eksik parametrelerle de tahmin yapılabilir.
    - Ok tuşları ile değerleri ayarlayabilirsiniz.
    """)

with col_guide2:
    st.markdown("""
    **📊 Excel Yükleme:**
    - Şablon dosyasını indirin ve doldurun.
    - Target_Phar ve Activation_Atmosphere zorunludur.
    - Sonuçlar CSV veya Excel olarak indirilebilir.
    """)

st.info("⚠️ **Önemli:** Micropore Volume, Total Pore Volume'den küçük olmalıdır.")
