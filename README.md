# 📊 Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin

Este repositorio contiene el material asociado al estudio académico:

**“Efecto del Sentimiento en Redes Sociales sobre el Precio del Bitcoin”**

El objetivo del proyecto es analizar si el sentimiento expresado en redes sociales, específicamente Twitter, puede aportar capacidad predictiva sobre el precio futuro de Bitcoin, utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y modelos de Machine Learning y Deep Learning.

---

## 📁 Contenido del repositorio

El repositorio incluye los siguientes archivos principales:

- **`BTC_Sentiment_Improved_Notebook_v3.ipynb`**  
  Notebook principal del proyecto. Contiene:
  - Análisis exploratorio de los datos
  - Procesamiento y análisis de sentimiento
  - Construcción de variables
  - Implementación de modelos predictivos
  - Evaluación de resultados

- **`BTCUSDT_1h_2021-02-05_2021-08-21.csv`**  
  Dataset con precios históricos de Bitcoin en intervalo horario (1h), en formato OHLCV:
  - Open, High, Low, Close, Volume

- **`Efecto_del_Sentimiento_en_Redes_Sociales_sobre_el_Precio_del_Bitcoin.pdf`**  
  Artículo académico del estudio, donde se describe:
  - Marco teórico
  - Metodología
  - Resultados exploratorios
  - Discusión, limitaciones y trabajo futuro

- **`README.md`**  
  Documento descriptivo del proyecto.

---

## 🎯 Objetivo del estudio

Evaluar si el sentimiento promedio extraído desde Twitter puede anticipar variaciones en el precio de Bitcoin en ventanas de corto plazo (24 horas), y determinar si la incorporación de esta información mejora el desempeño de modelos predictivos en comparación con enfoques basados únicamente en datos históricos de precios.

---

## 🧠 Metodología

### 🔹 Datos
- **Redes sociales:** Tweets relacionados con Bitcoin, procesados para extraer métricas de sentimiento.
- **Precio:** Serie temporal del precio de Bitcoin en formato OHLC.

### 🔹 Procesamiento de lenguaje natural (NLP)
- Limpieza y normalización de texto.
- Análisis de sentimiento mediante enfoques léxicos (baseline).
- Agregación temporal del sentimiento para su integración con la serie de precios.

### 🔹 Modelos implementados
- Regresión Lineal
- Random Forest Regressor
- Redes neuronales recurrentes **LSTM**, orientadas al modelamiento de series temporales

### 🔹 Evaluación
- Métricas utilizadas: **MAE, MSE y R²**
- Comparación entre modelos con y sin variables de sentimiento

---

## 📊 Principales hallazgos (resumen)

- El sentimiento social por sí solo no actúa como un predictor robusto del precio.
- Sin embargo, cuando se incorpora como variable adicional, puede aportar señal complementaria en ciertos períodos.
- Los modelos LSTM muestran un mejor desempeño para capturar dependencias temporales complejas en comparación con modelos tradicionales.

---

## ⚠️ Limitaciones

- Posible presencia de ruido y actividad automatizada (bots) en los datos de Twitter.
- Sesgo temporal del período analizado.
- Limitaciones inherentes a los métodos léxicos de análisis de sentimiento (sarcasmo, jerga).
- Desfase temporal entre redes sociales y reacción del mercado.

Estas limitaciones se reconocen explícitamente en el artículo académico.

---

## 🔮 Trabajo futuro

- Incorporar modelos de lenguaje especializados en finanzas (FinBERT).
- Implementar detección explícita de bots.
- Integrar otras plataformas sociales como Reddit.
- Evaluar distintos horizontes temporales de predicción.

---

## 👥 Autores

- **Sebastián Marinovic Leiva**  
- **Ricardo Iván Lizana Aseña**  
- **Luis Andrés Gutiérrez González**  

Magíster en Data Science  
Universidad de Las Américas

---

## 📜 Licencia

Este repositorio se publica con fines académicos y educativos.
