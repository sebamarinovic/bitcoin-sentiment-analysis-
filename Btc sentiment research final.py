# -*- coding: utf-8 -*-
"""
================================================================================
BTC SENTIMENT ANALYSIS - PROFESSIONAL RESEARCH NOTEBOOK
================================================================================
Proyecto: Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin
Autores: Sebastián Marinovic, Ricardo Lizana, Luis Gutiérrez
Universidad de Las Américas - Magíster en Data Science

Este notebook genera todos los resultados, tablas y figuras necesarias para 
el paper de investigación en formato LaTeX/Overleaf.

Ejecutar en Google Colab con GPU para mejor rendimiento.
================================================================================
"""

# =============================================================================
# SECCIÓN 0: INSTALACIÓN Y CONFIGURACIÓN
# =============================================================================

# Descomentar estas líneas en Google Colab:
# !pip install kagglehub textblob nltk scikit-learn tensorflow --quiet
# !pip install scipy statsmodels --quiet

import os
import re
import math
import warnings
import random
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Configuración de estilo para figuras publicables
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Crear directorio para guardar figuras
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("BTC SENTIMENT ANALYSIS - RESEARCH NOTEBOOK")
print("=" * 70)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE PARÁMETROS
# =============================================================================

@dataclass
class Config:
    """Configuración central del experimento"""
    # Rutas de datos
    BTC_OHLCV_CSV: str = "BTCUSDT_1h_2021-02-05_2021-08-21.csv"
    TWEETS_LOCAL_FALLBACK: str = "bitcoin_tweets1000000.csv"
    
    # Horizontes de predicción (en horas)
    HORIZONS_HOURS: tuple = (1, 6, 24)
    
    # Ventanas para features técnicas
    ROLL_WINDOWS: tuple = (6, 12, 24, 48)
    EWMA_SPAN: int = 12
    
    # Train/Test split
    TEST_SIZE: float = 0.2
    
    # Features de sentimiento
    SENT_LAGS: tuple = (1, 3, 6, 12, 24)
    SENT_ROLL: tuple = (6, 12, 24, 48)
    MIN_TEXT_LEN: int = 3
    
    # Validación cruzada
    N_SPLITS_CV: int = 5
    
    # Random Forest
    RF_N_ESTIMATORS: int = 500
    RF_MAX_DEPTH: int = 12

CFG = Config()

print("\n📊 Configuración del Experimento:")
print(f"   - Horizontes de predicción: {CFG.HORIZONS_HOURS} horas")
print(f"   - Test size: {CFG.TEST_SIZE*100:.0f}%")
print(f"   - CV Folds: {CFG.N_SPLITS_CV}")

# =============================================================================
# SECCIÓN 2: CARGA DE DATOS
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 2: CARGA DE DATOS")
print("="*70)

# 2A) Cargar Tweets
tweets_csv_path = None

try:
    from kagglehub import dataset_download
    path = dataset_download("gautamchettiar/bitcoin-sentiment-analysis-twitter-data")
    candidates = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
    preferred = [p for p in candidates if os.path.basename(p).lower() == "bitcoin_tweets1000000.csv"]
    tweets_csv_path = preferred[0] if preferred else (candidates[0] if candidates else None)
    print(f"✅ Tweets (kagglehub): {tweets_csv_path}")
except Exception as e:
    print(f"ℹ️ kagglehub no disponible: {e}")
    print(f"   Intentando ruta local: {CFG.TWEETS_LOCAL_FALLBACK}")

if tweets_csv_path is None:
    if os.path.exists(CFG.TWEETS_LOCAL_FALLBACK):
        tweets_csv_path = CFG.TWEETS_LOCAL_FALLBACK
        print(f"✅ Tweets (local): {tweets_csv_path}")
    else:
        # Intentar en /content/ para Colab
        colab_path = f"/content/{CFG.TWEETS_LOCAL_FALLBACK}"
        if os.path.exists(colab_path):
            tweets_csv_path = colab_path
            print(f"✅ Tweets (Colab): {tweets_csv_path}")
        else:
            raise FileNotFoundError(
                "No se encontró el CSV de tweets. "
                "Sube 'bitcoin_tweets1000000.csv' o instala kagglehub."
            )

df_tw = pd.read_csv(tweets_csv_path, low_memory=False, encoding="latin1")
print(f"   Shape tweets: {df_tw.shape}")

# 2B) Cargar OHLCV
btc_paths = [
    CFG.BTC_OHLCV_CSV,
    f"/content/{CFG.BTC_OHLCV_CSV}",
    "/mnt/data/BTCUSDT_1h_2021-02-05_2021-08-21.csv"
]

df_btc = None
for p in btc_paths:
    if os.path.exists(p):
        df_btc = pd.read_csv(p)
        print(f"✅ OHLCV: {p}")
        break

if df_btc is None:
    raise FileNotFoundError(
        f"No se encontró OHLCV. Sube '{CFG.BTC_OHLCV_CSV}'"
    )

print(f"   Shape OHLCV: {df_btc.shape}")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO DE DATOS
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 3: PREPROCESAMIENTO")
print("="*70)

# 3A) Normalización temporal
TIME_CANDS = ["date", "datetime", "created_at", "timestamp", "time", "Date", "Datetime"]
TEXT_CANDS = ["cleanText", "text", "tweet", "content", "Text", "body"]

time_col = next((c for c in TIME_CANDS if c in df_tw.columns), None)
text_col = next((c for c in TEXT_CANDS if c in df_tw.columns), None)

print(f"   Columna temporal: {time_col}")
print(f"   Columna texto: {text_col}")

if time_col is None or text_col is None:
    raise ValueError("No se detectaron columnas de tiempo/texto.")

# Convertir timestamps
df_tw[time_col] = pd.to_datetime(df_tw[time_col], errors="coerce", utc=True)
df_tw = df_tw.dropna(subset=[time_col]).sort_values(time_col)
df_tw["ts_hour"] = df_tw[time_col].dt.floor("H")

df_btc["date"] = pd.to_datetime(df_btc["date"], errors="coerce", utc=True)
df_btc = df_btc.dropna(subset=["date"]).sort_values("date")
df_btc = df_btc.rename(columns={"date": "ts_hour"})

# 3B) Limpieza de texto
def clean_text_basic(s: str) -> str:
    """Limpieza básica de texto para análisis de sentimiento"""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)  # URLs
    s = re.sub(r"@[\w_]+", " ", s)          # Menciones
    s = re.sub(r"#[\w_]+", " ", s)          # Hashtags (mantener palabras)
    s = re.sub(r"[^a-z0-9\s$%]", " ", s)    # Solo alfanuméricos
    s = re.sub(r"\s+", " ", s).strip()
    return s

df_tw["text_clean"] = df_tw[text_col].apply(clean_text_basic)
df_tw = df_tw[df_tw["text_clean"].str.len() >= CFG.MIN_TEXT_LEN]
df_tw = df_tw.drop_duplicates(subset=["ts_hour", "text_clean"])

print(f"   Tweets tras limpieza: {df_tw.shape[0]:,}")

# Estadísticas del período
date_min = df_tw["ts_hour"].min()
date_max = df_tw["ts_hour"].max()
print(f"   Período tweets: {date_min.date()} a {date_max.date()}")

# =============================================================================
# SECCIÓN 4: ANÁLISIS DE SENTIMIENTO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 4: ANÁLISIS DE SENTIMIENTO")
print("="*70)

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

def vader_scores(text: str):
    """Calcula scores VADER para un texto"""
    s = sia.polarity_scores(text)
    return s["neg"], s["neu"], s["pos"], s["compound"]

def textblob_scores(text: str):
    """Calcula polaridad y subjetividad con TextBlob"""
    tb = TextBlob(text)
    return tb.sentiment.polarity, tb.sentiment.subjectivity

print("   Calculando VADER scores...")
df_tw[["vader_neg", "vader_neu", "vader_pos", "vader_comp"]] = df_tw["text_clean"].apply(
    lambda t: pd.Series(vader_scores(t))
)

print("   Calculando TextBlob scores...")
df_tw[["tb_polarity", "tb_subjectivity"]] = df_tw["text_clean"].apply(
    lambda t: pd.Series(textblob_scores(t))
)

# Etiquetas de sentimiento
df_tw["vader_label"] = np.where(
    df_tw["vader_comp"] > 0.05, "positive",
    np.where(df_tw["vader_comp"] < -0.05, "negative", "neutral")
)

# Keywords FOMO/FUD
FOMO_WORDS = ["moon", "pump", "bull", "ath", "buy", "long", "lambo", "rocket", "to the moon", "bullish"]
FUD_WORDS = ["dump", "bear", "crash", "sell", "short", "scam", "fear", "rekt", "dead", "bearish"]

def count_keywords(text, words):
    return sum(1 for w in words if w in text)

df_tw["fomo"] = df_tw["text_clean"].apply(lambda t: count_keywords(t, FOMO_WORDS))
df_tw["fud"] = df_tw["text_clean"].apply(lambda t: count_keywords(t, FUD_WORDS))

# Estadísticas de sentimiento
sent_dist = df_tw["vader_label"].value_counts(normalize=True)
print(f"\n   Distribución de sentimiento:")
for label, pct in sent_dist.items():
    print(f"      {label}: {pct*100:.1f}%")

# =============================================================================
# SECCIÓN 5: AGREGACIÓN TEMPORAL
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 5: AGREGACIÓN TEMPORAL")
print("="*70)

# Agregación por hora
agg = {
    "vader_comp": ["mean", "std", "median", "min", "max"],
    "vader_pos": ["mean"],
    "vader_neg": ["mean"],
    "tb_polarity": ["mean", "std"],
    "tb_subjectivity": ["mean"],
    "fomo": ["sum"],
    "fud": ["sum"],
    "text_clean": ["count"]
}

df_sent = df_tw.groupby("ts_hour").agg(agg)
df_sent.columns = ["_".join([c for c in col if c]).strip() for col in df_sent.columns.values]
df_sent = df_sent.rename(columns={"text_clean_count": "volume_social"}).sort_index()

# Proporciones por categoría
label_counts = pd.crosstab(df_tw["ts_hour"], df_tw["vader_label"])
label_props = label_counts.div(label_counts.sum(axis=1), axis=0).add_prefix("prop_")
df_sent = df_sent.join(label_props, how="left")

# Completar índice horario
full_index = pd.date_range(df_btc["ts_hour"].min(), df_btc["ts_hour"].max(), freq="H", tz="UTC")
df_sent = df_sent.reindex(full_index)
df_sent.index.name = "ts_hour"

# Manejo de NaN
df_sent["volume_social"] = df_sent["volume_social"].fillna(0)
df_sent["has_tweets"] = (df_sent["volume_social"] > 0).astype(int)

for c in ["prop_negative", "prop_neutral", "prop_positive"]:
    if c in df_sent.columns:
        df_sent[c] = df_sent[c].fillna(0)

for col in ["fomo_sum", "fud_sum"]:
    if col in df_sent.columns:
        df_sent[col] = df_sent[col].fillna(0)
        df_sent[col + "_ewma"] = df_sent[col].ewm(span=CFG.EWMA_SPAN, adjust=False).mean()

# EWMA de promedios
for col in ["vader_comp_mean", "tb_polarity_mean"]:
    if col in df_sent.columns:
        base = df_sent[col].copy()
        base_ff = base.ffill(limit=24)
        df_sent[col + "_ewma"] = base_ff.ewm(span=CFG.EWMA_SPAN, adjust=False).mean()

coverage = df_sent["has_tweets"].mean()
print(f"   Cobertura temporal (horas con tweets): {coverage*100:.1f}%")
print(f"   Promedio tweets/hora: {df_sent['volume_social'].mean():.1f}")

# =============================================================================
# SECCIÓN 6: FEATURES FINANCIERAS Y MERGE
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 6: FEATURES FINANCIERAS")
print("="*70)

df_btc2 = df_btc.copy()
df_btc2["ts_hour"] = pd.to_datetime(df_btc2["ts_hour"], utc=True)
df_btc2 = df_btc2.set_index("ts_hour").sort_index()

for c in ["open", "high", "low", "close", "volume"]:
    df_btc2[c] = pd.to_numeric(df_btc2[c], errors="coerce")
df_btc2 = df_btc2.dropna(subset=["close"])

# Retornos logarítmicos
df_btc2["ret_1h"] = np.log(df_btc2["close"]).diff()

# Features técnicas
for w in CFG.ROLL_WINDOWS:
    df_btc2[f"vol_{w}h"] = df_btc2["ret_1h"].rolling(w).std()
    df_btc2[f"ma_{w}h"] = df_btc2["close"].rolling(w).mean()
    df_btc2[f"mom_{w}h"] = df_btc2["close"].pct_change(w)

# RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

df_btc2["rsi_14"] = calculate_rsi(df_btc2["close"], 14)

# Merge con sentimiento
df = df_btc2.join(df_sent, how="left")
df["volume_social"] = df["volume_social"].fillna(0)
df["has_tweets"] = df["has_tweets"].fillna(0).astype(int)

# Variables objetivo (targets)
for h in CFG.HORIZONS_HOURS:
    df[f"close_t+{h}h"] = df["close"].shift(-h)
    df[f"ret_t+{h}h"] = np.log(df[f"close_t+{h}h"]) - np.log(df["close"])
    df[f"dir_t+{h}h"] = (df[f"ret_t+{h}h"] > 0).astype(int)

df = df.dropna(subset=[f"ret_t+{h}h" for h in CFG.HORIZONS_HOURS])

print(f"   Dataset final: {df.shape}")
print(f"   Período: {df.index.min().date()} a {df.index.max().date()}")
print(f"   Observaciones: {len(df):,}")

# =============================================================================
# SECCIÓN 7: FEATURES DE SENTIMIENTO CON LAGS Y ROLLING
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 7: FEATURES DE SENTIMIENTO")
print("="*70)

sent_base_cols = [
    "vader_comp_mean", "tb_polarity_mean",
    "vader_comp_mean_ewma", "tb_polarity_mean_ewma",
    "fomo_sum", "fud_sum", "fomo_sum_ewma", "fud_sum_ewma",
    "prop_negative", "prop_neutral", "prop_positive",
    "volume_social", "has_tweets"
]
sent_base_cols = [c for c in sent_base_cols if c in df.columns]

# Crear lags
for col in sent_base_cols:
    for lag in CFG.SENT_LAGS:
        df[f"{col}_lag{lag}h"] = df[col].shift(lag)

# Crear rolling features
for col in ["vader_comp_mean_ewma", "tb_polarity_mean_ewma", "volume_social", "fomo_sum_ewma", "fud_sum_ewma"]:
    if col in df.columns:
        for w in CFG.SENT_ROLL:
            df[f"{col}_roll{w}h_mean"] = df[col].rolling(w, min_periods=max(3, w//4)).mean()
            df[f"{col}_roll{w}h_std"] = df[col].rolling(w, min_periods=max(3, w//4)).std()

df_model = df.copy()
df_model = df_model.dropna(subset=["ret_1h", "rsi_14"])
df_model = df_model.dropna(subset=[f"ret_t+{h}h" for h in CFG.HORIZONS_HOURS])

n_features_base = len([c for c in df_model.columns if c.startswith(("vol_", "ma_", "mom_", "rsi"))])
n_features_sent = len([c for c in df_model.columns if "vader" in c or "tb_" in c or "fomo" in c or "fud" in c])

print(f"   Features técnicas (base): {n_features_base}")
print(f"   Features sentimiento: {n_features_sent}")
print(f"   Dataset modelado: {df_model.shape}")

# =============================================================================
# SECCIÓN 8: VISUALIZACIONES PARA EL PAPER
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 8: GENERANDO FIGURAS")
print("="*70)

# --- FIGURA 1: Serie temporal de precio BTC ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_model.index, df_model["close"], linewidth=0.8, color='#1f77b4')
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio de cierre (USD)")
ax.set_title("Precio de cierre diario de Bitcoin (BTC) durante el período analizado")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_btc_price.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig1_btc_price.pdf")
print("   ✅ Figura 1: Precio BTC")
plt.show()

# --- FIGURA 2: Distribución de sentimiento ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e74c3c', '#95a5a6', '#27ae60']
labels = ['Negativo', 'Neutral', 'Positivo']
counts = df_tw["vader_label"].value_counts()[['negative', 'neutral', 'positive']]
bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Categoría de Sentimiento")
ax.set_ylabel("Número de Tweets")
ax.set_title("Distribución del Sentimiento en Tweets relacionados con Bitcoin")
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
            f'{count:,}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_sentiment_distribution.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig2_sentiment_distribution.pdf")
print("   ✅ Figura 2: Distribución de sentimiento")
plt.show()

# --- FIGURA 3: Precio vs Sentimiento (dual axis) ---
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# Resample para visualización más clara
daily_price = df_model["close"].resample("D").mean()
daily_sent = df_model["vader_comp_mean_ewma"].resample("D").mean()

ax1.plot(daily_price.index, daily_price, 'b-', linewidth=1.2, label='Precio BTC', alpha=0.8)
ax2.plot(daily_sent.index, daily_sent, 'r-', linewidth=1.2, label='Sentimiento (EWMA)', alpha=0.8)

ax1.set_xlabel('Fecha')
ax1.set_ylabel('Precio BTC (USD)', color='b')
ax2.set_ylabel('Sentimiento VADER (compound)', color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Evolución temporal: Precio BTC vs Sentimiento Social')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_price_vs_sentiment.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig3_price_vs_sentiment.pdf")
print("   ✅ Figura 3: Precio vs Sentimiento")
plt.show()

# --- FIGURA 4: Matriz de correlación ---
corr_cols = ['close', 'ret_1h', 'volume', 'vader_comp_mean', 'tb_polarity_mean', 
             'volume_social', 'fomo_sum', 'fud_sum', 'rsi_14', 'vol_24h']
corr_cols = [c for c in corr_cols if c in df_model.columns]
corr_matrix = df_model[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Matriz de Correlación: Variables Financieras y de Sentimiento')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_correlation_matrix.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig4_correlation_matrix.pdf")
print("   ✅ Figura 4: Matriz de correlación")
plt.show()

# --- FIGURA 5: Rolling correlation ---
fig, ax = plt.subplots(figsize=(12, 5))
window = 48
rolling_corr = df_model["vader_comp_mean_ewma"].rolling(window).corr(df_model["ret_1h"])
ax.plot(rolling_corr.index, rolling_corr, linewidth=0.8, color='purple', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.fill_between(rolling_corr.index, rolling_corr, 0, alpha=0.3, color='purple')
ax.set_xlabel('Fecha')
ax.set_ylabel('Correlación')
ax.set_title(f'Correlación Rolling ({window}h): Sentimiento vs Retorno')
ax.set_ylim(-0.5, 0.5)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_rolling_correlation.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig5_rolling_correlation.pdf")
print("   ✅ Figura 5: Correlación rolling")
plt.show()

# --- FIGURA 6: Heatmap cobertura social ---
tmp = df_model.copy()
tmp["dow"] = tmp.index.dayofweek
tmp["hour"] = tmp.index.hour
pivot = tmp.pivot_table(values="volume_social", index="dow", columns="hour", aggfunc="mean")

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Tweets promedio/hora'})
ax.set_yticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], rotation=0)
ax.set_xlabel('Hora (UTC)')
ax.set_ylabel('Día de la semana')
ax.set_title('Cobertura Social: Promedio de Tweets por Hora y Día')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_social_coverage.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig6_social_coverage.pdf")
print("   ✅ Figura 6: Cobertura social")
plt.show()

# =============================================================================
# SECCIÓN 9: MODELADO Y EVALUACIÓN
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 9: MODELADO Y EVALUACIÓN")
print("="*70)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Definir conjuntos de features
base_cols = ["open", "high", "low", "close", "volume", "ret_1h", "rsi_14"] + \
            [c for c in df_model.columns if c.startswith(("vol_", "ma_", "mom_"))]

sent_feat_cols = [c for c in df_model.columns if (
    "vader_" in c or "tb_" in c or "fomo" in c or "fud" in c or 
    "prop_" in c or "volume_social" in c or "has_tweets" in c
)]
sent_feat_cols = [c for c in sent_feat_cols if not c.startswith(("close_t+", "ret_t+", "dir_t+"))]

# Limpiar dataset
all_target_reg_cols = [f"ret_t+{h}h" for h in CFG.HORIZONS_HOURS]
all_target_dir_cols = [f"dir_t+{h}h" for h in CFG.HORIZONS_HOURS]
all_cols_to_consider = list(set(base_cols + sent_feat_cols + all_target_reg_cols + all_target_dir_cols))

# Filtrar columnas existentes
all_cols_to_consider = [c for c in all_cols_to_consider if c in df_model.columns]
df_model_cleaned = df_model[all_cols_to_consider].dropna()

# Actualizar base_cols y sent_feat_cols
base_cols = [c for c in base_cols if c in df_model_cleaned.columns]
sent_feat_cols = [c for c in sent_feat_cols if c in df_model_cleaned.columns]

X_base = df_model_cleaned[base_cols].copy()
X_sent = df_model_cleaned[list(set(base_cols + sent_feat_cols))].copy()

print(f"   Features BASE: {len(base_cols)}")
print(f"   Features BASE+SENT: {X_sent.shape[1]}")
print(f"   Observaciones: {len(df_model_cleaned):,}")

# Funciones de evaluación
def eval_regression_walkforward(X, y, n_splits=5):
    """Evaluación walk-forward para regresión"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        
        models = {
            "Naive_Zero": None,
            "Ridge": Pipeline([("scaler", StandardScaler()), 
                              ("model", Ridge(alpha=1.0, random_state=SEED))]),
            "RandomForest": RandomForestRegressor(
                n_estimators=CFG.RF_N_ESTIMATORS, 
                max_depth=CFG.RF_MAX_DEPTH, 
                random_state=SEED, 
                n_jobs=-1
            ),
        }
        
        for name, model in models.items():
            if name == "Naive_Zero":
                y_pred = np.zeros(len(yte))
            else:
                model.fit(Xtr, ytr)
                y_pred = model.predict(Xte)
            
            rows.append({
                "fold": fold,
                "model": name,
                "MAE": mean_absolute_error(yte, y_pred),
                "RMSE": math.sqrt(mean_squared_error(yte, y_pred)),
                "R2": r2_score(yte, y_pred),
            })
    
    return pd.DataFrame(rows)

def eval_direction_walkforward(X, y_dir, n_splits=5):
    """Evaluación walk-forward para clasificación direccional"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_dir.iloc[tr], y_dir.iloc[te]
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED))
        ])
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(yte, pred)
        })
    
    return pd.DataFrame(rows)

# Almacenar resultados
all_results = []
all_direction_results = []

for H in CFG.HORIZONS_HOURS:
    print(f"\n   📊 Evaluando horizonte {H}h...")
    
    y_reg = df_model_cleaned[f"ret_t+{H}h"]
    y_dir = df_model_cleaned[f"dir_t+{H}h"]
    
    # Evaluación regresión
    res_base = eval_regression_walkforward(X_base, y_reg, CFG.N_SPLITS_CV)
    res_sent = eval_regression_walkforward(X_sent, y_reg, CFG.N_SPLITS_CV)
    
    # Evaluación dirección
    acc_base = eval_direction_walkforward(X_base, y_dir, CFG.N_SPLITS_CV)
    acc_sent = eval_direction_walkforward(X_sent, y_dir, CFG.N_SPLITS_CV)
    
    # Agregar resultados
    for _, row in res_base.groupby("model")[["MAE", "RMSE", "R2"]].mean().iterrows():
        all_results.append({
            "Horizon": f"{H}h",
            "Model": _,
            "Features": "BASE",
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"]
        })
    
    for _, row in res_sent.groupby("model")[["MAE", "RMSE", "R2"]].mean().iterrows():
        all_results.append({
            "Horizon": f"{H}h",
            "Model": _,
            "Features": "BASE+SENT",
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"]
        })
    
    all_direction_results.append({
        "Horizon": f"{H}h",
        "Accuracy_BASE": acc_base["accuracy"].mean(),
        "Accuracy_SENT": acc_sent["accuracy"].mean(),
        "Delta": acc_sent["accuracy"].mean() - acc_base["accuracy"].mean()
    })

# Crear DataFrames de resultados
df_results = pd.DataFrame(all_results)
df_direction = pd.DataFrame(all_direction_results)

print("\n" + "="*70)
print("RESULTADOS DE REGRESIÓN (Walk-Forward CV)")
print("="*70)
print(df_results.to_string(index=False))

print("\n" + "="*70)
print("RESULTADOS DE CLASIFICACIÓN DIRECCIONAL")
print("="*70)
print(df_direction.to_string(index=False))

# Guardar tablas en CSV y LaTeX
df_results.to_csv(f"{TABLES_DIR}/regression_results.csv", index=False)
df_direction.to_csv(f"{TABLES_DIR}/direction_results.csv", index=False)

# Formato LaTeX
latex_reg = df_results.pivot_table(
    index=['Horizon', 'Model'], 
    columns='Features', 
    values=['MAE', 'RMSE', 'R2']
).round(6)
latex_reg.to_latex(f"{TABLES_DIR}/regression_results.tex")

df_direction.to_latex(f"{TABLES_DIR}/direction_results.tex", index=False)
print(f"\n   ✅ Tablas guardadas en {TABLES_DIR}/")

# =============================================================================
# SECCIÓN 10: ANÁLISIS DE IMPORTANCIA DE FEATURES
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 10: IMPORTANCIA DE FEATURES")
print("="*70)

from sklearn.inspection import permutation_importance

# Usar horizonte de 24h para análisis principal
target = df_model_cleaned["ret_t+24h"]
split = int(len(df_model_cleaned) * (1 - CFG.TEST_SIZE))

Xtr, Xte = X_sent.iloc[:split], X_sent.iloc[split:]
ytr, yte = target.iloc[:split], target.iloc[split:]

print("   Entrenando Random Forest para análisis de importancia...")
rf = RandomForestRegressor(
    n_estimators=800, 
    max_depth=12, 
    random_state=SEED, 
    n_jobs=-1
)
rf.fit(Xtr, ytr)

pred = rf.predict(Xte)
holdout_mae = mean_absolute_error(yte, pred)
holdout_r2 = r2_score(yte, pred)
print(f"   Holdout MAE (ret 24h): {holdout_mae:.6f}")
print(f"   Holdout R²: {holdout_r2:.6f}")

print("   Calculando permutation importance...")
imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=SEED, n_jobs=-1)
imp_df = pd.DataFrame({
    "feature": X_sent.columns,
    "importance": imp.importances_mean,
    "importance_std": imp.importances_std
}).sort_values("importance", ascending=False)

# Clasificar features por grupo
imp_df["group"] = "Other"
imp_df.loc[imp_df["feature"].str.contains("vader|tb_|fomo|fud|prop_|volume_social|has_tweets"), "group"] = "Sentiment"
imp_df.loc[imp_df["feature"].str.contains("^vol_|^ma_|^mom_|rsi_|ret_1h|close|open|high|low|volume"), "group"] = "Technical"

# Guardar tabla de importancia
imp_df.to_csv(f"{TABLES_DIR}/feature_importance.csv", index=False)

# --- FIGURA 7: Top 20 features ---
fig, ax = plt.subplots(figsize=(10, 8))
top20 = imp_df.head(20).iloc[::-1]
colors = ['#e74c3c' if g == 'Sentiment' else '#3498db' for g in top20['group']]
ax.barh(range(len(top20)), top20["importance"], color=colors, xerr=top20["importance_std"])
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"])
ax.set_xlabel("Importancia (Permutation)")
ax.set_title("Top 20 Features por Importancia - Random Forest (ret 24h)")

# Leyenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Sentimiento'),
                   Patch(facecolor='#3498db', label='Técnico')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_feature_importance.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig7_feature_importance.pdf")
print("   ✅ Figura 7: Feature importance")
plt.show()

# --- FIGURA 8: Importancia por grupo ---
grp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
colors_grp = {'Sentiment': '#e74c3c', 'Technical': '#3498db', 'Other': '#95a5a6'}
ax.bar(grp.index, grp.values, color=[colors_grp.get(g, '#95a5a6') for g in grp.index])
ax.set_ylabel("Suma de Importancias")
ax.set_title("Importancia Total por Grupo de Features")
for i, (idx, val) in enumerate(grp.items()):
    ax.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_importance_by_group.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig8_importance_by_group.pdf")
print("   ✅ Figura 8: Importancia por grupo")
plt.show()

# =============================================================================
# SECCIÓN 11: COMPARACIÓN BASE vs SENT
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 11: COMPARACIÓN DETALLADA BASE vs SENT")
print("="*70)

# Entrenar ambos modelos
rf_base = RandomForestRegressor(n_estimators=600, max_depth=12, random_state=SEED, n_jobs=-1)
rf_sent = RandomForestRegressor(n_estimators=600, max_depth=12, random_state=SEED, n_jobs=-1)

X_base_clean = X_base.iloc[:split], X_base.iloc[split:]
X_sent_clean = X_sent.iloc[:split], X_sent.iloc[split:]

rf_base.fit(X_base.iloc[:split], ytr)
rf_sent.fit(X_sent.iloc[:split], ytr)

pred_base = rf_base.predict(X_base.iloc[split:])
pred_sent = rf_sent.predict(X_sent.iloc[split:])

abs_err_base = np.abs(yte.values - pred_base)
abs_err_sent = np.abs(yte.values - pred_sent)
delta_err = abs_err_sent - abs_err_base

# --- FIGURA 9: Distribución de mejora ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(delta_err, bins=60, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
ax.axvline(delta_err.mean(), color='green', linestyle='-', linewidth=2, label=f'Media: {delta_err.mean():.6f}')
ax.set_xlabel("Δ|error| (SENT - BASE)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del Cambio en Error Absoluto (negativo = mejora con sentimiento)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_error_distribution.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig9_error_distribution.pdf")
print("   ✅ Figura 9: Distribución de error")
plt.show()

mae_base = abs_err_base.mean()
mae_sent = abs_err_sent.mean()
improvement = (mae_base - mae_sent) / mae_base * 100

print(f"\n   MAE BASE:     {mae_base:.6f}")
print(f"   MAE SENT:     {mae_sent:.6f}")
print(f"   Δ MAE:        {mae_sent - mae_base:.6f}")
print(f"   Mejora:       {improvement:.2f}%")

# --- FIGURA 10: Predicciones vs Real ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot BASE
axes[0].scatter(yte, pred_base, alpha=0.3, s=10, c='blue')
axes[0].plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', linewidth=2)
axes[0].set_xlabel("Retorno Real")
axes[0].set_ylabel("Retorno Predicho")
axes[0].set_title(f"BASE: R² = {r2_score(yte, pred_base):.4f}")

# Scatter plot SENT
axes[1].scatter(yte, pred_sent, alpha=0.3, s=10, c='green')
axes[1].plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', linewidth=2)
axes[1].set_xlabel("Retorno Real")
axes[1].set_ylabel("Retorno Predicho")
axes[1].set_title(f"BASE+SENT: R² = {r2_score(yte, pred_sent):.4f}")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_predictions_scatter.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig10_predictions_scatter.pdf")
print("   ✅ Figura 10: Scatter predicciones")
plt.show()

# =============================================================================
# SECCIÓN 12: TEST ESTADÍSTICO DE HIPÓTESIS
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 12: TEST ESTADÍSTICO")
print("="*70)

from scipy import stats

# Test de Wilcoxon para errores pareados
stat, p_value = stats.wilcoxon(abs_err_base, abs_err_sent, alternative='greater')
print(f"\n   Test de Wilcoxon (H1: error_base > error_sent):")
print(f"   Estadístico: {stat:.2f}")
print(f"   p-value: {p_value:.6f}")

if p_value < 0.05:
    print("   → Resultado: RECHAZAMOS H0 al 5% de significancia")
    print("   → El sentimiento mejora significativamente las predicciones")
else:
    print("   → Resultado: NO rechazamos H0 al 5% de significancia")
    print("   → No hay evidencia suficiente de mejora significativa")

# Test de correlación Spearman: sentimiento vs retornos futuros
corr_spearman, p_spearman = stats.spearmanr(
    df_model_cleaned["vader_comp_mean_ewma"].dropna(),
    df_model_cleaned.loc[df_model_cleaned["vader_comp_mean_ewma"].notna(), "ret_t+24h"]
)
print(f"\n   Correlación Spearman (sentimiento vs ret_24h):")
print(f"   ρ = {corr_spearman:.4f}")
print(f"   p-value: {p_spearman:.6f}")

# Guardar resultados estadísticos
stats_results = {
    "wilcoxon_stat": stat,
    "wilcoxon_pvalue": p_value,
    "spearman_corr": corr_spearman,
    "spearman_pvalue": p_spearman,
    "mae_base": mae_base,
    "mae_sent": mae_sent,
    "improvement_pct": improvement
}

with open(f"{TABLES_DIR}/statistical_tests.json", "w") as f:
    json.dump(stats_results, f, indent=2)
print(f"\n   ✅ Resultados estadísticos guardados")

# =============================================================================
# SECCIÓN 13: EVENT STUDY
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 13: EVENT STUDY")
print("="*70)

def event_study(series_event, series_ret, q=0.99, pre=24, post=24, kind="high"):
    """Análisis de eventos alrededor de shocks de sentimiento"""
    thr = series_event.quantile(q) if kind == "high" else series_event.quantile(1-q)
    idx = series_event[series_event >= thr].index if kind == "high" else series_event[series_event <= thr].index
    
    mats = []
    for t in idx:
        try:
            if (t - pd.Timedelta(hours=pre) in series_ret.index) and \
               (t + pd.Timedelta(hours=post) in series_ret.index):
                w = series_ret.loc[t-pd.Timedelta(hours=pre): t+pd.Timedelta(hours=post)].values
                if len(w) == pre + post + 1:
                    mats.append(w)
        except:
            continue
    
    if len(mats) == 0:
        return None, thr, 0
    
    mats = np.vstack(mats)
    mean = mats.mean(axis=0)
    se = mats.std(axis=0) / np.sqrt(mats.shape[0])
    x = np.arange(-pre, post + 1)
    return (x, mean, se), thr, mats.shape[0]

# Event study: shocks FUD
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FUD alto
out, thr, n = event_study(df_model["fud_sum_ewma"].fillna(0), df_model["ret_1h"], q=0.99, pre=24, post=24, kind="high")
if out:
    x, m, se = out
    axes[0].plot(x, m, 'b-', linewidth=2)
    axes[0].fill_between(x, m-1.96*se, m+1.96*se, alpha=0.3, color='blue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title(f"Event Study: Shocks FUD (top 1%) | n={n}")
    axes[0].set_xlabel("Horas relativas al evento")
    axes[0].set_ylabel("Retorno promedio (1h)")

# Sentimiento negativo
out, thr, n = event_study(df_model["vader_comp_mean_ewma"], df_model["ret_1h"], q=0.99, pre=24, post=24, kind="low")
if out:
    x, m, se = out
    axes[1].plot(x, m, 'r-', linewidth=2)
    axes[1].fill_between(x, m-1.96*se, m+1.96*se, alpha=0.3, color='red')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title(f"Event Study: Sentimiento Extremo Negativo (bottom 1%) | n={n}")
    axes[1].set_xlabel("Horas relativas al evento")
    axes[1].set_ylabel("Retorno promedio (1h)")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig11_event_study.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig11_event_study.pdf")
print("   ✅ Figura 11: Event study")
plt.show()

# =============================================================================
# SECCIÓN 14: RESUMEN EJECUTIVO Y CONCLUSIONES
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 14: RESUMEN EJECUTIVO")
print("="*70)

summary = f"""
================================================================================
RESUMEN DE RESULTADOS - BTC SENTIMENT ANALYSIS
================================================================================

DATOS:
- Período analizado: {df_model.index.min().date()} a {df_model.index.max().date()}
- Total observaciones: {len(df_model_cleaned):,}
- Total tweets analizados: {len(df_tw):,}
- Cobertura temporal: {coverage*100:.1f}%

DISTRIBUCIÓN DE SENTIMIENTO:
- Positivo: {sent_dist.get('positive', 0)*100:.1f}%
- Neutral: {sent_dist.get('neutral', 0)*100:.1f}%
- Negativo: {sent_dist.get('negative', 0)*100:.1f}%

RESULTADOS DE MODELADO (Horizonte 24h):
- MAE Modelo BASE: {mae_base:.6f}
- MAE Modelo BASE+SENT: {mae_sent:.6f}
- Mejora porcentual: {improvement:.2f}%

TEST ESTADÍSTICO:
- Wilcoxon p-value: {p_value:.6f}
- Correlación Spearman (sent vs ret): {corr_spearman:.4f} (p={p_spearman:.6f})

CONCLUSIÓN:
{"El sentimiento social muestra capacidad predictiva significativa sobre retornos de BTC." if p_value < 0.05 else "No se encontró evidencia estadística suficiente de mejora significativa."}

ARCHIVOS GENERADOS:
- Figuras: {FIGURES_DIR}/ (11 figuras en PNG y PDF)
- Tablas: {TABLES_DIR}/ (CSV, LaTeX, JSON)

================================================================================
"""

print(summary)

# Guardar resumen
with open(f"{TABLES_DIR}/executive_summary.txt", "w") as f:
    f.write(summary)

# =============================================================================
# SECCIÓN 15: MODELO LSTM (OPCIONAL)
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 15: MODELO LSTM (Deep Learning)")
print("="*70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    print("   Preparando datos para LSTM...")
    
    # Preparar secuencias
    SEQUENCE_LENGTH = 24
    
    # Seleccionar features para LSTM
    lstm_features = ['close', 'volume', 'ret_1h', 'rsi_14', 'vol_24h',
                     'vader_comp_mean_ewma', 'volume_social', 'fomo_sum_ewma', 'fud_sum_ewma']
    lstm_features = [c for c in lstm_features if c in df_model_cleaned.columns]
    
    lstm_data = df_model_cleaned[lstm_features + ['ret_t+24h']].dropna()
    
    # Escalar datos
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(lstm_data[lstm_features])
    y_scaled = scaler_y.fit_transform(lstm_data[['ret_t+24h']])
    
    # Crear secuencias
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i+seq_length])
            ys.append(y[i+seq_length])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)
    
    # Split temporal
    split_idx = int(len(X_seq) * (1 - CFG.TEST_SIZE))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Construir modelo LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(lstm_features))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("   Entrenando LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluar
    y_pred_lstm = model.predict(X_test, verbose=0)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred_lstm)
    
    lstm_mae = mean_absolute_error(y_test_inv, y_pred_inv)
    lstm_r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"\n   LSTM Results:")
    print(f"   MAE: {lstm_mae:.6f}")
    print(f"   R²: {lstm_r2:.6f}")
    
    # --- FIGURA 12: Training history ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('LSTM Training Loss')
    axes[0].legend()
    
    axes[1].scatter(y_test_inv, y_pred_inv, alpha=0.3, s=10)
    axes[1].plot([y_test_inv.min(), y_test_inv.max()], 
                 [y_test_inv.min(), y_test_inv.max()], 'r--')
    axes[1].set_xlabel('Retorno Real')
    axes[1].set_ylabel('Retorno Predicho')
    axes[1].set_title(f'LSTM Predictions (R² = {lstm_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig12_lstm_results.png", dpi=300)
    plt.savefig(f"{FIGURES_DIR}/fig12_lstm_results.pdf")
    print("   ✅ Figura 12: LSTM results")
    plt.show()
    
    # Actualizar resumen con LSTM
    lstm_summary = f"""
LSTM RESULTS:
- MAE: {lstm_mae:.6f}
- R²: {lstm_r2:.6f}
- Sequence length: {SEQUENCE_LENGTH}
- Features: {len(lstm_features)}
"""
    with open(f"{TABLES_DIR}/lstm_results.txt", "w") as f:
        f.write(lstm_summary)
    
except ImportError:
    print("   ⚠️ TensorFlow no disponible. Saltando LSTM.")
except Exception as e:
    print(f"   ⚠️ Error en LSTM: {e}")

# =============================================================================
# SECCIÓN 16: TABLA FINAL COMPARATIVA (PARA LATEX)
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 16: TABLA COMPARATIVA FINAL")
print("="*70)

# Crear tabla comparativa final
final_comparison = []

for H in CFG.HORIZONS_HOURS:
    y_h = df_model_cleaned[f"ret_t+{H}h"]
    
    # Naive
    final_comparison.append({
        'Horizonte': f'{H}h',
        'Modelo': 'Naive (Zero)',
        'MAE': mean_absolute_error(y_h, np.zeros(len(y_h))),
        'RMSE': np.sqrt(mean_squared_error(y_h, np.zeros(len(y_h)))),
        'R²': 0.0
    })

# Agregar resultados de modelos entrenados
for row in all_results:
    if row['Model'] != 'Naive_Zero':
        final_comparison.append({
            'Horizonte': row['Horizon'],
            'Modelo': f"{row['Model']} ({row['Features']})",
            'MAE': row['MAE'],
            'RMSE': row['RMSE'],
            'R²': row['R2']
        })

df_final = pd.DataFrame(final_comparison)

# Formato para LaTeX
latex_table = df_final.pivot_table(
    index='Modelo',
    columns='Horizonte',
    values=['MAE', 'RMSE', 'R²'],
    aggfunc='first'
).round(6)

print("\nTabla Comparativa Final:")
print(df_final.to_string(index=False))

# Guardar
df_final.to_csv(f"{TABLES_DIR}/final_comparison.csv", index=False)
latex_table.to_latex(f"{TABLES_DIR}/final_comparison.tex")

print(f"\n✅ Tabla final guardada en {TABLES_DIR}/")

# =============================================================================
# FIN DEL NOTEBOOK
# =============================================================================

print("\n" + "="*70)
print("🎉 NOTEBOOK COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"""
Archivos generados:

📊 FIGURAS ({FIGURES_DIR}/):
   - fig1_btc_price.png/pdf
   - fig2_sentiment_distribution.png/pdf
   - fig3_price_vs_sentiment.png/pdf
   - fig4_correlation_matrix.png/pdf
   - fig5_rolling_correlation.png/pdf
   - fig6_social_coverage.png/pdf
   - fig7_feature_importance.png/pdf
   - fig8_importance_by_group.png/pdf
   - fig9_error_distribution.png/pdf
   - fig10_predictions_scatter.png/pdf
   - fig11_event_study.png/pdf
   - fig12_lstm_results.png/pdf (si TensorFlow disponible)

📋 TABLAS ({TABLES_DIR}/):
   - regression_results.csv/tex
   - direction_results.csv/tex
   - feature_importance.csv
   - statistical_tests.json
   - final_comparison.csv/tex
   - executive_summary.txt

Para usar en Overleaf:
1. Sube las figuras PDF a tu proyecto
2. Copia las tablas .tex directamente
3. Usa los valores del executive_summary.txt para el texto

""")