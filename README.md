# 📊 Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin

Este repositorio contiene el código, datos y material asociado al estudio académico:

**“Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin”**

El proyecto analiza si el sentimiento expresado en Twitter puede aportar capacidad predictiva sobre el precio futuro de Bitcoin, utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y modelos de Machine Learning y Deep Learning.

---

## 🎯 Objetivo

Evaluar si la incorporación de variables de sentimiento social mejora la predicción del precio de Bitcoin en horizontes de corto plazo (24 horas), en comparación con modelos basados únicamente en datos históricos de precios.

---

## 🧠 Metodología

- **Datos**
  - Tweets relacionados con Bitcoin (Twitter / Kaggle)
  - Precio de Bitcoin en formato OHLC
- **NLP**
  - Análisis de sentimiento con enfoques léxicos (VADER)
  - Extensión futura con Transformers (BERT / FinBERT)
- **Modelos**
  - Regresión Lineal
  - Random Forest
  - LSTM (series temporales)
- **Evaluación**
  - MAE, MSE, R²

---

## 📁 Estructura del repositorio

- `paper/`: artículo académico en PDF  
- `data/`: datasets (raw y procesados)  
- `notebooks/`: notebooks Jupyter del flujo completo  
- `figures/`: visualizaciones y gráficos  
- `src/`: scripts reutilizables de procesamiento y modelado  

---

## ⚠️ Consideraciones

- El dataset de Twitter puede contener ruido y actividad automatizada (bots).
- No se realizó detección explícita de bots; se reconoce como limitación del estudio.
- El análisis se realiza a nivel agregado para mitigar este efecto.

---

## 🔮 Trabajo futuro

- Detección explícita de bots (Botometer, anomaly detection)
- Incorporación de FinBERT
- Inclusión de datos de Reddit
- Evaluación de distintos horizontes temporales

---

## 👥 Autores

- Sebastián Marinovic Leiva  
- Ricardo Iván Lizana Aseña  
- Luis Andrés Gutiérrez González  

Magíster en Data Science – Universidad de Las Américas

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia MIT.
