# -*- coding: utf-8 -*-
"""
================================================================================
BTC SENTIMENT ANALYSIS - PROFESSIONAL RESEARCH NOTEBOOK (v2 - CON PCA)
================================================================================
Proyecto: Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin
Autores: Sebastián Marinovic, Ricardo Lizana, Luis Gutiérrez
Universidad de Las Américas - Magíster en Data Science

ACTUALIZACIÓN v2: Incluye análisis PCA para reducción de dimensionalidad
de las variables de sentimiento.

Ejecutar en Google Colab con GPU para mejor rendimiento.
================================================================================
"""

# =============================================================================
# SECCIÓN 0: INSTALACIÓN Y CONFIGURACIÓN
# =============================================================================

# Descomentar en Google Colab:
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

# Configuración de estilo
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

FIGURES_DIR = "figures"
TABLES_DIR = "tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("BTC SENTIMENT ANALYSIS - RESEARCH NOTEBOOK (v2 con PCA)")
print("=" * 70)

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    """Configuración central del experimento"""
    BTC_OHLCV_CSV: str = "BTCUSDT_1h_2021-02-05_2021-08-21.csv"
    TWEETS_LOCAL_FALLBACK: str = "bitcoin_tweets1000000.csv"
    
    HORIZONS_HOURS: tuple = (1, 6, 24)
    ROLL_WINDOWS: tuple = (6, 12, 24, 48)
    EWMA_SPAN: int = 12
    TEST_SIZE: float = 0.2
    SENT_LAGS: tuple = (1, 3, 6, 12, 24)
    SENT_ROLL: tuple = (6, 12, 24, 48)
    MIN_TEXT_LEN: int = 3
    N_SPLITS_CV: int = 5
    RF_N_ESTIMATORS: int = 500
    RF_MAX_DEPTH: int = 12
    
    # PCA Config
    PCA_VARIANCE_THRESHOLD: float = 0.80  # Retener 80% de varianza
    PCA_N_COMPONENTS: int = 3  # O usar n_components fijo

CFG = Config()

print(f"\n📊 Configuración:")
print(f"   - Horizontes: {CFG.HORIZONS_HOURS} horas")
print(f"   - PCA: {CFG.PCA_N_COMPONENTS} componentes")

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

if tweets_csv_path is None:
    for fallback in [CFG.TWEETS_LOCAL_FALLBACK, f"/content/{CFG.TWEETS_LOCAL_FALLBACK}"]:
        if os.path.exists(fallback):
            tweets_csv_path = fallback
            print(f"✅ Tweets (local): {tweets_csv_path}")
            break

if tweets_csv_path is None:
    raise FileNotFoundError("No se encontró CSV de tweets.")

df_tw = pd.read_csv(tweets_csv_path, low_memory=False, encoding="latin1")
print(f"   Shape tweets: {df_tw.shape}")

# 2B) Cargar OHLCV
btc_paths = [CFG.BTC_OHLCV_CSV, f"/content/{CFG.BTC_OHLCV_CSV}"]
df_btc = None
for p in btc_paths:
    if os.path.exists(p):
        df_btc = pd.read_csv(p)
        print(f"✅ OHLCV: {p}")
        break

if df_btc is None:
    raise FileNotFoundError(f"No se encontró OHLCV: {CFG.BTC_OHLCV_CSV}")

print(f"   Shape OHLCV: {df_btc.shape}")

# =============================================================================
# SECCIÓN 3: PREPROCESAMIENTO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 3: PREPROCESAMIENTO")
print("="*70)

TIME_CANDS = ["date", "datetime", "created_at", "timestamp", "time", "Date"]
TEXT_CANDS = ["cleanText", "text", "tweet", "content", "Text", "body"]

time_col = next((c for c in TIME_CANDS if c in df_tw.columns), None)
text_col = next((c for c in TEXT_CANDS if c in df_tw.columns), None)

print(f"   Columna temporal: {time_col}")
print(f"   Columna texto: {text_col}")

# Convertir timestamps
df_tw[time_col] = pd.to_datetime(df_tw[time_col], errors="coerce", utc=True)
df_tw = df_tw.dropna(subset=[time_col]).sort_values(time_col)
df_tw["ts_hour"] = df_tw[time_col].dt.floor("H")

df_btc["date"] = pd.to_datetime(df_btc["date"], errors="coerce", utc=True)
df_btc = df_btc.dropna(subset=["date"]).sort_values("date")
df_btc = df_btc.rename(columns={"date": "ts_hour"})

# Limpieza de texto
def clean_text_basic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#[\w_]+", " ", s)
    s = re.sub(r"[^a-z0-9\s$%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df_tw["text_clean"] = df_tw[text_col].apply(clean_text_basic)
df_tw = df_tw[df_tw["text_clean"].str.len() >= CFG.MIN_TEXT_LEN]
df_tw = df_tw.drop_duplicates(subset=["ts_hour", "text_clean"])

print(f"   Tweets tras limpieza: {df_tw.shape[0]:,}")

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
    s = sia.polarity_scores(text)
    return s["neg"], s["neu"], s["pos"], s["compound"]

def textblob_scores(text: str):
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

# Clasificación de sentimiento
def classify_sentiment(score: float, thresh: float = 0.05) -> str:
    if score >= thresh:
        return "positive"
    elif score <= -thresh:
        return "negative"
    return "neutral"

df_tw["vader_label"] = df_tw["vader_comp"].apply(classify_sentiment)

# FOMO/FUD
FOMO_WORDS = ["moon", "rocket", "pump", "bullish", "buy", "hodl", "lambo", "rich", "profit", "ath"]
FUD_WORDS = ["crash", "dump", "bearish", "sell", "scam", "fear", "panic", "dead", "bubble", "ponzi"]

df_tw["fomo_count"] = df_tw["text_clean"].apply(
    lambda t: sum(1 for w in FOMO_WORDS if w in t.split())
)
df_tw["fud_count"] = df_tw["text_clean"].apply(
    lambda t: sum(1 for w in FUD_WORDS if w in t.split())
)

print(f"   Tweets positivos: {(df_tw['vader_label']=='positive').sum():,}")
print(f"   Tweets negativos: {(df_tw['vader_label']=='negative').sum():,}")
print(f"   Tweets neutrales: {(df_tw['vader_label']=='neutral').sum():,}")

# =============================================================================
# SECCIÓN 5: AGREGACIÓN HORARIA
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 5: AGREGACIÓN HORARIA")
print("="*70)

agg_dict = {
    "vader_neg": ["mean", "std", "max"],
    "vader_neu": ["mean"],
    "vader_pos": ["mean", "std", "max"],
    "vader_comp": ["mean", "std", "min", "max"],
    "tb_polarity": ["mean", "std"],
    "tb_subjectivity": ["mean"],
    "fomo_count": ["sum", "mean"],
    "fud_count": ["sum", "mean"],
    "text_clean": "count"
}

df_hourly = df_tw.groupby("ts_hour").agg(agg_dict)
df_hourly.columns = ["_".join(col).strip() for col in df_hourly.columns]
df_hourly = df_hourly.rename(columns={"text_clean_count": "volume_social"})
df_hourly = df_hourly.reset_index()

# Proporciones de sentimiento
prop = df_tw.groupby("ts_hour")["vader_label"].value_counts(normalize=True).unstack(fill_value=0)
prop.columns = [f"prop_{c}" for c in prop.columns]
prop = prop.reset_index()

df_hourly = pd.merge(df_hourly, prop, on="ts_hour", how="left")

# Has tweets flag
df_hourly["has_tweets"] = (df_hourly["volume_social"] > 0).astype(int)

print(f"   Horas con tweets: {len(df_hourly):,}")

# =============================================================================
# SECCIÓN 6: MERGE Y FEATURES TÉCNICOS
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 6: FEATURES TÉCNICOS")
print("="*70)

df = pd.merge(df_btc, df_hourly, on="ts_hour", how="left")
df = df.set_index("ts_hour").sort_index()

# Rellenar NaN en sentimiento
sent_cols = [c for c in df.columns if "vader" in c or "tb_" in c or "fomo" in c or "fud" in c or "prop_" in c]
df[sent_cols] = df[sent_cols].fillna(method="ffill", limit=6)
df["volume_social"] = df["volume_social"].fillna(0)
df["has_tweets"] = df["has_tweets"].fillna(0)

# Retornos
df["ret_1h"] = np.log(df["close"]).diff()

# Volatilidad
for w in CFG.ROLL_WINDOWS:
    df[f"vol_{w}h"] = df["ret_1h"].rolling(w).std()

# Medias móviles
for w in CFG.ROLL_WINDOWS:
    df[f"ma_{w}h"] = df["close"].rolling(w).mean()

# Momentum
for w in [6, 12, 24]:
    df[f"mom_{w}h"] = df["close"].pct_change(w)

# RSI
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = calc_rsi(df["close"], 14)

# EWMA de sentimiento
for col in ["vader_comp_mean", "tb_polarity_mean", "fomo_count_sum", "fud_count_sum"]:
    if col in df.columns:
        df[f"{col}_ewma"] = df[col].ewm(span=CFG.EWMA_SPAN).mean()

# Variables objetivo
for h in CFG.HORIZONS_HOURS:
    df[f"ret_t+{h}h"] = np.log(df["close"].shift(-h) / df["close"])
    df[f"dir_t+{h}h"] = (df[f"ret_t+{h}h"] > 0).astype(int)

print(f"   Shape tras features: {df.shape}")

# =============================================================================
# SECCIÓN 7: PCA DE VARIABLES DE SENTIMIENTO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 7: PCA DE VARIABLES DE SENTIMIENTO")
print("="*70)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Seleccionar variables de sentimiento para PCA
sentiment_cols_for_pca = [
    "vader_comp_mean", "vader_pos_mean", "vader_neg_mean",
    "tb_polarity_mean", "tb_subjectivity_mean",
    "fomo_count_sum", "fud_count_sum",
    "volume_social", "has_tweets"
]

# Filtrar columnas que existen
sentiment_cols_for_pca = [c for c in sentiment_cols_for_pca if c in df.columns]
print(f"   Variables para PCA: {len(sentiment_cols_for_pca)}")
print(f"   {sentiment_cols_for_pca}")

# Crear subset sin NaN para PCA
df_pca_subset = df[sentiment_cols_for_pca].dropna()
print(f"   Observaciones válidas para PCA: {len(df_pca_subset):,}")

# Escalado
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(df_pca_subset)

# PCA completo para análisis
pca_full = PCA()
pca_full.fit(X_scaled)

# Varianza explicada
explained_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = np.argmax(explained_var >= CFG.PCA_VARIANCE_THRESHOLD) + 1

print(f"\n   📊 Análisis de Varianza Explicada:")
for i, (var, cum_var) in enumerate(zip(pca_full.explained_variance_ratio_, explained_var)):
    print(f"      PC{i+1}: {var*100:.1f}% (acumulado: {cum_var*100:.1f}%)")
    if cum_var >= 0.95:
        break

print(f"\n   Componentes para {CFG.PCA_VARIANCE_THRESHOLD*100:.0f}% varianza: {n_components_80}")
print(f"   Usando {CFG.PCA_N_COMPONENTS} componentes (configurado)")

# --- FIGURA PCA: Varianza explicada ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(explained_var)+1), explained_var, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=0.80, color='r', linestyle='--', label='80% varianza')
ax.axhline(y=0.90, color='g', linestyle='--', label='90% varianza')
ax.axvline(x=n_components_80, color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Número de Componentes')
ax.set_ylabel('Varianza Explicada Acumulada')
ax.set_title('PCA: Selección de Componentes de Sentimiento')
ax.legend()
ax.set_ylim(0, 1.05)
ax.set_xticks(range(1, len(explained_var)+1))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_pca_variance.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig_pca_variance.pdf")
print(f"   ✅ Figura PCA: Varianza explicada")
plt.show()

# Aplicar PCA con n componentes configurados
pca = PCA(n_components=CFG.PCA_N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# Agregar componentes al dataframe
for i in range(CFG.PCA_N_COMPONENTS):
    df[f"sent_pca_{i+1}"] = np.nan
    df.loc[df_pca_subset.index, f"sent_pca_{i+1}"] = X_pca[:, i]

# Loadings (contribución de cada variable a cada PC)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(CFG.PCA_N_COMPONENTS)],
    index=sentiment_cols_for_pca
)

print(f"\n   📊 Loadings PCA (contribución de variables):")
print(loadings.round(3).to_string())

# Guardar loadings
loadings.to_csv(f"{TABLES_DIR}/pca_loadings.csv")

# --- FIGURA PCA: Loadings heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax)
ax.set_title('PCA Loadings: Contribución de Variables de Sentimiento')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_pca_loadings.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig_pca_loadings.pdf")
print(f"   ✅ Figura PCA: Loadings")
plt.show()

# Interpretación de componentes
print(f"\n   📊 Interpretación de Componentes Principales:")
for i in range(CFG.PCA_N_COMPONENTS):
    top_pos = loadings[f"PC{i+1}"].nlargest(2)
    top_neg = loadings[f"PC{i+1}"].nsmallest(2)
    print(f"      PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% var):")
    print(f"         (+) {top_pos.index[0]}: {top_pos.iloc[0]:.2f}, {top_pos.index[1]}: {top_pos.iloc[1]:.2f}")
    print(f"         (-) {top_neg.index[0]}: {top_neg.iloc[0]:.2f}, {top_neg.index[1]}: {top_neg.iloc[1]:.2f}")

# =============================================================================
# SECCIÓN 8: PREPARACIÓN PARA MODELADO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 8: PREPARACIÓN PARA MODELADO")
print("="*70)

# Crear lags de sentimiento
sent_base_cols = [
    "vader_comp_mean", "vader_comp_mean_ewma", "tb_polarity_mean",
    "fomo_count_sum", "fud_count_sum", "fomo_count_sum_ewma", "fud_count_sum_ewma",
    "prop_negative", "prop_neutral", "prop_positive",
    "volume_social", "has_tweets"
]
sent_base_cols = [c for c in sent_base_cols if c in df.columns]

# Lags
for col in sent_base_cols:
    for lag in CFG.SENT_LAGS:
        df[f"{col}_lag{lag}h"] = df[col].shift(lag)

# Rolling features
for col in ["vader_comp_mean_ewma", "tb_polarity_mean_ewma", "volume_social"]:
    if col in df.columns:
        for w in CFG.SENT_ROLL:
            df[f"{col}_roll{w}h_mean"] = df[col].rolling(w, min_periods=max(3, w//4)).mean()
            df[f"{col}_roll{w}h_std"] = df[col].rolling(w, min_periods=max(3, w//4)).std()

# Limpiar dataset
df_model = df.copy()
df_model = df_model.dropna(subset=["ret_1h", "rsi_14"])
df_model = df_model.dropna(subset=[f"ret_t+{h}h" for h in CFG.HORIZONS_HOURS])

print(f"   Dataset modelado: {df_model.shape}")

# =============================================================================
# SECCIÓN 9: MODELADO CON 4 ESCENARIOS
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 9: MODELADO Y EVALUACIÓN (4 ESCENARIOS)")
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

sent_raw_cols = [c for c in df_model.columns if (
    "vader_" in c or "tb_" in c or "fomo" in c or "fud" in c or 
    "prop_" in c or "volume_social" in c or "has_tweets" in c
) and "pca" not in c]

pca_cols = [f"sent_pca_{i+1}" for i in range(CFG.PCA_N_COMPONENTS)]

# Filtrar columnas existentes
base_cols = [c for c in base_cols if c in df_model.columns]
sent_raw_cols = [c for c in sent_raw_cols if c in df_model.columns]
pca_cols = [c for c in pca_cols if c in df_model.columns]

# Limpiar dataset para todos los escenarios
all_cols = list(set(base_cols + sent_raw_cols + pca_cols + 
                    [f"ret_t+{h}h" for h in CFG.HORIZONS_HOURS] +
                    [f"dir_t+{h}h" for h in CFG.HORIZONS_HOURS]))
all_cols = [c for c in all_cols if c in df_model.columns]
df_model_cleaned = df_model[all_cols].dropna()

print(f"   Observaciones finales: {len(df_model_cleaned):,}")
print(f"\n   📊 Escenarios de Features:")
print(f"      1. BASE:     {len(base_cols)} features (solo técnicos)")
print(f"      2. SENT_RAW: {len(base_cols) + len(sent_raw_cols)} features (técnicos + sentimiento crudo)")
print(f"      3. PCA_1:    {len(base_cols) + 1} features (técnicos + PC1)")
print(f"      4. PCA_3:    {len(base_cols) + len(pca_cols)} features (técnicos + PC1-PC3)")

# Crear datasets
X_base = df_model_cleaned[base_cols].copy()
X_sent_raw = df_model_cleaned[base_cols + sent_raw_cols].copy()
X_pca_1 = df_model_cleaned[base_cols + ["sent_pca_1"]].copy()
X_pca_3 = df_model_cleaned[base_cols + pca_cols].copy()

datasets = {
    "BASE": X_base,
    "SENT_RAW": X_sent_raw,
    "PCA_1": X_pca_1,
    "PCA_3": X_pca_3
}

# Función de evaluación
def eval_regression_walkforward(X, y, n_splits=5):
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

# Almacenar resultados
all_results = []

for H in CFG.HORIZONS_HOURS:
    print(f"\n   📊 Evaluando horizonte {H}h...")
    
    y_reg = df_model_cleaned[f"ret_t+{H}h"]
    
    for scenario_name, X_scenario in datasets.items():
        res = eval_regression_walkforward(X_scenario, y_reg, CFG.N_SPLITS_CV)
        
        for _, row in res.groupby("model")[["MAE", "RMSE", "R2"]].mean().iterrows():
            all_results.append({
                "Horizon": f"{H}h",
                "Model": _,
                "Features": scenario_name,
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"]
            })

# Crear DataFrame de resultados
df_results = pd.DataFrame(all_results)

print("\n" + "="*70)
print("RESULTADOS DE REGRESIÓN (Walk-Forward CV)")
print("="*70)

# Pivot para mejor visualización
pivot_results = df_results.pivot_table(
    index=['Horizon', 'Model'], 
    columns='Features', 
    values='MAE'
).round(6)

print("\nMAE por Escenario:")
print(pivot_results.to_string())

# Guardar resultados
df_results.to_csv(f"{TABLES_DIR}/regression_results_with_pca.csv", index=False)
pivot_results.to_csv(f"{TABLES_DIR}/mae_comparison_pca.csv")

# =============================================================================
# SECCIÓN 10: VISUALIZACIÓN DE RESULTADOS PCA
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 10: VISUALIZACIÓN DE RESULTADOS")
print("="*70)

# --- FIGURA: Comparación MAE por escenario ---
fig, ax = plt.subplots(figsize=(12, 6))

# Filtrar solo RandomForest para comparación clara
rf_results = df_results[df_results["Model"] == "RandomForest"]
pivot_rf = rf_results.pivot_table(index='Horizon', columns='Features', values='MAE')

# Reordenar columnas
col_order = ["BASE", "SENT_RAW", "PCA_1", "PCA_3"]
col_order = [c for c in col_order if c in pivot_rf.columns]
pivot_rf = pivot_rf[col_order]

x = np.arange(len(pivot_rf.index))
width = 0.2
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for i, col in enumerate(pivot_rf.columns):
    ax.bar(x + i*width, pivot_rf[col], width, label=col, color=colors[i])

ax.set_xlabel('Horizonte')
ax.set_ylabel('MAE')
ax.set_title('Comparación de MAE por Escenario de Features (Random Forest)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(pivot_rf.index)
ax.legend(title='Features')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_pca_comparison.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig_pca_comparison.pdf")
print("   ✅ Figura: Comparación PCA")
plt.show()

# --- FIGURA: Mejora porcentual vs BASE ---
fig, ax = plt.subplots(figsize=(10, 6))

improvement_data = []
for horizon in pivot_rf.index:
    base_mae = pivot_rf.loc[horizon, "BASE"]
    for scenario in ["SENT_RAW", "PCA_1", "PCA_3"]:
        if scenario in pivot_rf.columns:
            scenario_mae = pivot_rf.loc[horizon, scenario]
            improvement = (base_mae - scenario_mae) / base_mae * 100
            improvement_data.append({
                "Horizon": horizon,
                "Scenario": scenario,
                "Improvement": improvement
            })

df_improvement = pd.DataFrame(improvement_data)
pivot_imp = df_improvement.pivot_table(index='Horizon', columns='Scenario', values='Improvement')

# Plot
pivot_imp.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71', '#9b59b6'])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Horizonte')
ax.set_ylabel('Mejora vs BASE (%)')
ax.set_title('Mejora Porcentual en MAE vs Modelo BASE (positivo = mejora)')
ax.legend(title='Escenario')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_pca_improvement.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/fig_pca_improvement.pdf")
print("   ✅ Figura: Mejora porcentual")
plt.show()

# =============================================================================
# SECCIÓN 11: ANÁLISIS ESTADÍSTICO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 11: ANÁLISIS ESTADÍSTICO")
print("="*70)

from scipy import stats

# Correlación de componentes PCA con retornos
print("\n   📊 Correlación Spearman: Componentes PCA vs Retornos")
for h in CFG.HORIZONS_HOURS:
    print(f"\n   Horizonte {h}h:")
    target = df_model_cleaned[f"ret_t+{h}h"]
    
    for pc in pca_cols:
        if pc in df_model_cleaned.columns:
            corr, pval = stats.spearmanr(df_model_cleaned[pc].dropna(), target.dropna())
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"      {pc}: ρ = {corr:.4f} (p = {pval:.4e}) {sig}")

# Comparar VADER crudo vs PC1
print("\n   📊 Comparación: VADER vs PC1")
for h in [24]:
    target = df_model_cleaned[f"ret_t+{h}h"]
    
    corr_vader, p_vader = stats.spearmanr(df_model_cleaned["vader_comp_mean"], target)
    corr_pc1, p_pc1 = stats.spearmanr(df_model_cleaned["sent_pca_1"], target)
    
    print(f"      VADER compound: ρ = {corr_vader:.4f} (p = {p_vader:.4e})")
    print(f"      PCA Component 1: ρ = {corr_pc1:.4f} (p = {p_pc1:.4e})")

# =============================================================================
# SECCIÓN 12: RESUMEN EJECUTIVO
# =============================================================================

print("\n" + "="*70)
print("SECCIÓN 12: RESUMEN EJECUTIVO")
print("="*70)

# Calcular mejores resultados
best_by_horizon = df_results.loc[df_results.groupby(['Horizon', 'Model'])['MAE'].idxmin()]

summary = f"""
================================================================================
                    RESUMEN EJECUTIVO - ANÁLISIS CON PCA
================================================================================

DATOS:
- Tweets analizados: {len(df_tw):,}
- Período: {df_model.index.min().date()} a {df_model.index.max().date()}
- Observaciones para modelado: {len(df_model_cleaned):,}

PCA DE SENTIMIENTO:
- Variables incluidas: {len(sentiment_cols_for_pca)}
- Componentes extraídos: {CFG.PCA_N_COMPONENTS}
- Varianza explicada PC1: {pca.explained_variance_ratio_[0]*100:.1f}%
- Varianza explicada PC1-PC3: {sum(pca.explained_variance_ratio_)*100:.1f}%

COMPARACIÓN DE ESCENARIOS (MAE, horizonte 24h, Random Forest):
"""

h24_rf = df_results[(df_results["Horizon"] == "24h") & (df_results["Model"] == "RandomForest")]
for _, row in h24_rf.iterrows():
    summary += f"   {row['Features']:10s}: MAE = {row['MAE']:.6f}\n"

# Mejor escenario
best_scenario = h24_rf.loc[h24_rf["MAE"].idxmin()]
summary += f"""
MEJOR ESCENARIO (24h): {best_scenario['Features']}
- MAE = {best_scenario['MAE']:.6f}

CONCLUSIÓN:
"""

# Determinar conclusión
base_mae = h24_rf[h24_rf["Features"] == "BASE"]["MAE"].values[0]
pca3_mae = h24_rf[h24_rf["Features"] == "PCA_3"]["MAE"].values[0] if "PCA_3" in h24_rf["Features"].values else base_mae
sent_raw_mae = h24_rf[h24_rf["Features"] == "SENT_RAW"]["MAE"].values[0]

if pca3_mae < base_mae:
    improvement = (base_mae - pca3_mae) / base_mae * 100
    summary += f"PCA mejora el MAE en {improvement:.2f}% respecto a BASE\n"
elif sent_raw_mae < base_mae:
    improvement = (base_mae - sent_raw_mae) / base_mae * 100
    summary += f"Sentimiento crudo mejora el MAE en {improvement:.2f}% respecto a BASE\n"
else:
    summary += "El sentimiento (crudo o PCA) NO mejora significativamente sobre BASE\n"

summary += """
================================================================================
"""

print(summary)

# Guardar resumen
with open(f"{TABLES_DIR}/executive_summary_pca.txt", "w") as f:
    f.write(summary)

print(f"✅ Resumen guardado en {TABLES_DIR}/executive_summary_pca.txt")
print(f"✅ Figuras guardadas en {FIGURES_DIR}/")
print(f"✅ Tablas guardadas en {TABLES_DIR}/")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
