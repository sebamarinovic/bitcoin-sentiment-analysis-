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

| Componente | Descripción |
|------------|-------------|
| **Fuente social** | Tweets relacionados con Bitcoin (dataset Kaggle, 1M tweets) |
| **Preprocesamiento** | Limpieza de texto, agregación horaria, features de sentimiento (lags/rolling, EWMA), indicadores **FOMO/FUD** por conteo de keywords |
| **Fuente financiera** | OHLCV horario de BTCUSDT (Binance) |
| **Targets** | Retornos logarítmicos futuros a **H ∈ {1, 6, 24}** horas |
| **Modelos** | Naive (Zero), Ridge Regression, Random Forest (500 árboles), LSTM |
| **Evaluación** | Walk-Forward Cross-Validation (TimeSeriesSplit, 5 folds), métricas MAE/RMSE/R² + accuracy direccional |

---

## 📊 Principales resultados

| Hallazgo | Evidencia | Interpretación |
|----------|----------:|----------------|
| Correlación Spearman | ρ ≈ 0.088 (p < 0.01) | Existe relación, pero **débil** |
| Wilcoxon (error BASE > error SENT) | p ≈ 0.683 | **No** hay mejora significativa |
| Desempeño predictivo incremental | No mejora consistente | El sentimiento no aporta señal adicional útil en el período analizado |

> **Conclusión:** Para **Feb–Ago 2021**, el sentimiento de Twitter **no agrega poder predictivo incremental** sobre variables técnicas tradicionales.

---

## 📈 Visualizaciones

### Precio de Bitcoin (Feb-Ago 2021)
![Precio BTC](figures/fig1.png)

### Distribución de Sentimiento
![Sentimiento](figures/fig2.png)

### Matriz de Correlación
![Correlación](figures/fig4.png)

### Feature Importance (Random Forest)
![Importance](figures/fig7.png)

---

## 📁 Estructura del repositorio

```text
bitcoin-sentiment-analysis-/
├── figures/                                      # Visualizaciones generadas
│   ├── fig1.png                                  # Precio BTC
│   ├── fig2.png                                  # Distribución sentimiento
│   ├── fig3.png                                  # Precio vs Sentimiento
│   ├── fig4.png                                  # Matriz de correlación
│   ├── fig5.png                                  # Correlación rolling
│   ├── fig6.png                                  # Cobertura social (heatmap)
│   ├── fig7.png                                  # Feature importance
│   ├── fig8.png                                  # Importancia por grupo
│   ├── fig9.png                                  # Distribución de error
│   ├── fig10.png                                 # Scatter predicciones
│   ├── fig11.png                                 # Event study
│   └── fig12.png                                 # LSTM results
├── BTCUSDT_1h_2021-02-05_2021-08-21.csv          # OHLCV BTCUSDT (1h)
├── Btc_sentiment_research_final.py               # Script reproducible (pipeline + figs + tablas)
├── Efecto_del_Sentimiento_en_Redes_Sociales_sobre_el_Precio_del_Bitcoin.pdf  # Paper
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/sebamarinovic/bitcoin-sentiment-analysis-.git
cd bitcoin-sentiment-analysis-
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el script

```bash
python Btc_sentiment_research_final.py
```

El script descargará automáticamente los tweets desde Kaggle (requiere cuenta) y generará todas las figuras y tablas en las carpetas `figures/` y `tables/`.

---

## 📦 Dependencias principales

```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
tensorflow >= 2.12.0
nltk >= 3.8.0
textblob >= 0.17.0
vaderSentiment >= 3.3.2
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.10.0
statsmodels >= 0.14.0
kagglehub >= 0.2.0
```

---

## 📚 Referencias

1. Kristoufek, L. (2013). Bitcoin meets Google Trends and Wikipedia. *Scientific Reports*, 3, 3415.
2. Garcia, D., & Schweitzer, F. (2015). Social signals and algorithmic trading of Bitcoin. *Royal Society Open Science*.
3. Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis. *ICWSM 2014*.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

- **Sebastián Marinovic** - sebamarinovic.leiva@gmail.com
- **GitHub:** [@sebamarinovic](https://github.com/sebamarinovic)

---

<p align="center">
  <b>Universidad de Las Américas - Magíster en Data Science - Enero 2026</b>
</p>
