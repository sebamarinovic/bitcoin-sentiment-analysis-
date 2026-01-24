# Bitcoin Sentiment Analysis: FinBERT + PCA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Descripción

Este proyecto investiga la capacidad predictiva del sentimiento expresado en Twitter sobre el precio de Bitcoin, comparando métodos tradicionales de análisis de sentimiento (VADER, TextBlob) con modelos de NLP basados en Transformers (FinBERT, Twitter-RoBERTa) y técnicas de reducción de dimensionalidad (PCA).

**Autores:** Sebastián Marinovic, Ricardo Lizana, Luis Gutiérrez  
**Institución:** Universidad de Las Américas - Magíster en Data Science

## 🎯 Resultados Principales

| Modelo | Escenario | MAE | Mejora vs BASE |
|--------|-----------|-----|----------------|
| **LSTM** | **PCA_FinBERT** | **0.0316** | **-18.6%** ✅ |
| LSTM | PCA_ALL | 0.0381 | -1.8% |
| Random Forest | PCA_ALL | 0.0586 | -3.9% |
| Random Forest | BASE | 0.0610 | -- |

### Correlación Sentimiento-Retornos (24h)

| Método | Correlación (ρ) | p-value | Significativo |
|--------|-----------------|---------|---------------|
| VADER | 0.027 | 0.302 | ❌ No |
| **FinBERT** | **0.113** | **1.53e-05** | ✅ **Sí (4x más fuerte)** |

## 📁 Estructura del Proyecto

```
bitcoin-sentiment-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                    # Instrucciones para obtener datos
│
├── notebooks/
│   └── BTC_Sentiment_Analysis.ipynb
│
├── scripts/
│   ├── btc_sentiment_finbert_pca_v3.py    # Script principal completo
│   └── btc_sentiment_final_with_pca.py    # Versión solo VADER + PCA
│
├── results/
│   ├── figures/
│   │   ├── fig_pca_variance_comparison.png
│   │   ├── fig_pca_loadings_combined.png
│   │   ├── fig_scenario_comparison.png
│   │   ├── fig_lstm_learning_curves.png
│   │   └── fig_lstm_predictions.png
│   └── tables/
│       ├── mae_comparison.csv
│       ├── pca_loadings_combined.csv
│       └── executive_summary_v3.txt
│
└── paper/
    ├── main.tex
    └── figures/
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/your-username/bitcoin-sentiment-analysis.git
cd bitcoin-sentiment-analysis
```

### 2. Crear ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar datos

Los datos deben descargarse de las siguientes fuentes:

**Tweets:**
- Dataset: [Bitcoin Sentiment Analysis Twitter Data](https://www.kaggle.com/datasets/gautamchettiar/bitcoin-sentiment-analysis-twitter-data)

**Precios OHLCV:**
- Fuente: Binance (BTC/USDT, frecuencia horaria)
- Período: Febrero - Agosto 2021

## 💻 Uso

### Opción 1: Google Colab (Recomendado)

```python
# Instalar dependencias
!pip install transformers torch kagglehub textblob nltk scikit-learn scipy --quiet

# Subir script y datos, luego ejecutar:
!python btc_sentiment_finbert_pca_v3.py
```

### Opción 2: Ejecución local

```bash
python scripts/btc_sentiment_finbert_pca_v3.py
```

**Nota:** Se recomienda GPU para la inferencia de FinBERT (~2 horas en GPU T4).

## 📊 Metodología

### 1. Datos

| Variable | Valor |
|----------|-------|
| Tweets totales | 1,000,025 |
| Tweets tras limpieza | 904,427 |
| Período | Feb-Ago 2021 |
| Observaciones modelado | 1,449 |

### 2. Métodos de Sentimiento

| Método | Tipo | Características |
|--------|------|-----------------|
| VADER | Léxico | Rápido, general |
| TextBlob | Léxico | Polaridad + Subjetividad |
| FinBERT | Transformer | Especializado finanzas |
| RoBERTa | Transformer | Especializado Twitter |

### 3. Escenarios de Features

| Escenario | Descripción | Features |
|-----------|-------------|----------|
| BASE | Solo técnicos | 5 |
| VADER | Técnicos + VADER | 12 |
| FinBERT | Técnicos + FinBERT | 15 |
| PCA_VADER | Técnicos + PCA(VADER) | 8 |
| PCA_FinBERT | Técnicos + PCA(FinBERT) | 8 |
| PCA_ALL | Técnicos + PCA(todos) | 8 |

### 4. Modelos

- **Naive Zero**: Baseline (predice retorno 0)
- **Ridge Regression**: Regularización L2
- **Random Forest**: 200 árboles, max_depth=10
- **LSTM**: 2 capas, 64 unidades, dropout 0.2

### 5. Validación

- Walk-Forward Cross-Validation (5 splits)
- Split temporal 80/20 para LSTM
- Métrica principal: MAE

## 📈 Resultados

### Varianza Explicada por PCA

| Config | PC1 | PC2 | PC3 | Total |
|--------|-----|-----|-----|-------|
| PCA_VADER | 45.1% | 20.1% | 18.4% | 83.6% |
| PCA_FinBERT | 50.4% | 19.4% | 9.4% | 79.2% |
| PCA_ALL | 47.7% | 16.8% | 12.5% | 76.9% |

### Interpretación de Componentes

- **PC1**: Índice de sentimiento positivo (combina todos los métodos)
- **PC2**: Volumen social y negatividad (actividad + pánico)
- **PC3**: Señales FOMO/FUD (específico crypto, correlación negativa)

## 🔬 Conclusiones

1. **FinBERT > VADER**: Correlación 4x más fuerte con retornos
2. **PCA mejora LSTM**: Reducción de 18.6% en MAE
3. **PCA como regularizador**: Elimina ruido y multicolinealidad
4. **Eficiencia de mercado**: R² negativo sugiere predictibilidad limitada

## 📝 Citar

```bibtex
@mastersthesis{marinovic2026btcsentiment,
  author = {Marinovic, Sebastián and Lizana, Ricardo and Gutiérrez, Luis},
  title = {Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin},
  school = {Universidad de Las Américas},
  year = {2026},
  type = {Tesis de Magíster en Data Science}
}
```

## 📚 Referencias Principales

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
- Barbieri, F. et al. (2020). TweetEval: Unified Benchmark for Tweet Classification
- Hutto, C.J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model
- Kristoufek, L. (2013). Bitcoin meets Google Trends and Wikipedia

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE)

## 🤝 Contacto

Universidad de Las Américas - Magíster en Data Science
