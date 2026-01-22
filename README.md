# 📈 Bitcoin Sentiment Analysis: Effect of Social Media on BTC Price

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **Tesis de Magíster en Data Science** — Universidad de Las Américas (UDLA), Chile (Enero 2026).  
> Este repositorio contiene el **paper** y el **script reproducible** que genera **figuras/tablas** para el estudio.

---

## 👥 Autores

- **Sebastián Marinovic Leiva**
- **Ricardo Iván Lizana Aseña**
- **Luis Andrés Gutiérrez González**

---

## 📌 Resumen

Este estudio evalúa si el **sentimiento extraído desde Twitter** aporta **capacidad predictiva** sobre variaciones del precio de **Bitcoin** en horizontes de **1, 6 y 24 horas**, y si mejora el desempeño de modelos frente a un set base de indicadores técnicos.  
Se analizaron **905,863 tweets** (Feb–Ago 2021) con **VADER** y **TextBlob**, integrando señales de sentimiento a modelos **Ridge**, **Random Forest** y **LSTM**, con validación **Walk-Forward CV (5 folds)**.

**Resultado clave:** aunque existe **correlación estadísticamente significativa** entre sentimiento y retornos (Spearman ρ ≈ 0.088, p < 0.01), **no se observa mejora predictiva** al incorporar variables de sentimiento (Wilcoxon p ≈ 0.683).  

---

## 🔬 Metodología (alto nivel)

- **Fuente social:** Tweets relacionados con Bitcoin (dataset Kaggle).  
- **Preprocesamiento:** limpieza de texto, agregación horaria, features de sentimiento (lags/rolling, EWMA), indicadores **FOMO/FUD** por conteo de keywords.
- **Fuente financiera:** OHLCV horario de BTCUSDT (Binance).
- **Targets:** retornos logarítmicos futuros a **H ∈ {1, 6, 24}** horas.
- **Evaluación:** Walk-Forward Cross-Validation (TimeSeriesSplit, 5 folds), métricas MAE/RMSE/R² + accuracy direccional.

---

## 📊 Principales resultados

| Hallazgo | Evidencia | Interpretación |
|---|---:|---|
| Correlación Spearman | ρ ≈ 0.088 (p < 0.01) | Existe relación, pero **débil** |
| Wilcoxon (error BASE > error SENT) | p ≈ 0.683 | **No** hay mejora significativa |
| Desempeño predictivo incremental | No mejora consistente | El sentimiento no aporta señal adicional útil en el período analizado |

> Conclusión: Para **Feb–Ago 2021**, el sentimiento de Twitter **no agrega poder predictivo incremental** sobre variables técnicas tradicionales.

---

## 📁 Estructura del repositorio

```text
bitcoin-sentiment-analysis-/
├── figures/                                      # Visualizaciones generadas / o almacenadas
├── BTCUSDT_1h_2021-02-05_2021-08-21.csv          # OHLCV BTCUSDT (1h)
├── Btc sentiment research final.py               # Script reproducible (pipeline + figs + tablas)
├── Efecto_del_Sentimiento_en_Redes_Sociales_sobre_el_Precio_del_Bitcoin.pdf  # Paper
├── requirements.txt
├── README.md
└── LICENSE
```
