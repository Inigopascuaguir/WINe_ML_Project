# 🍷 Proyecto de Machine Learning: Predicción de la Calidad del Vino Tinto

## 📖 Descripción del Proyecto

Este proyecto tiene como objetivo predecir la calidad del vino tinto utilizando un conjunto de datos obtenido de [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data). La variable objetivo es `quality`, que representa la calidad del vino en una escala del 0 al 10. Se han desarrollado y evaluado varios algoritmos de Machine Learning, tanto supervisados como no supervisados, para abordar este problema.

Además, se ha creado una aplicación interactiva utilizando **Streamlit** para visualizar los resultados y permitir a los usuarios interactuar con el modelo.

---

## 📊 Dataset

El dataset utilizado contiene 11 características físico-químicas del vino tinto:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`

**Variable objetivo**: `quality` (valor numérico que representa la calidad del vino).

🔗 Descargar dataset: [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data)

---

## 🗂️ Estructura del Proyecto

```plaintext
wine-quality-prediction/
├── data/                    # 📁 Carpeta con el dataset original
├── notebooks/               # 📓 Jupyter Notebooks de análisis/modelado
├── models/                  # 🧠 Modelos entrenados guardados
├── src/                     # 🐍 Código fuente (scripts de Python)
├── app/                     # 🖥️ Aplicación Streamlit
├── README.md                # 📄 Este archivo
```

## 🛠️ Metodología

### 1. 🧹 Preprocesamiento de Datos
- **Limpieza de datos**: Manejo de valores nulos y duplicados.
- **Normalización y escalado** de características.
- **Balanceo de clases** con SMOTE (Synthetic Minority Over-sampling Technique).

### 2. 🤖 Modelado Supervisado
**Modelos implementados**:
- Random Forest
- Gradient Boosting
- SVM (Support Vector Machine)
- KNN (K-Nearest Neighbors)
- Regresión Logística

**Métricas evaluadas**:
- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusión

### 3. 🧩 Modelado No Supervisado
- **KMeans**: Agrupación de vinos en clusters basados en características físico-químicas.
- Evaluación con **Silhouette Score**.

### 4. ⚙️ Optimización
- Ajuste de hiperparámetros con **GridSearchCV** para mejorar el rendimiento.

---

## 📱 Aplicación Streamlit
Aplicación interactiva que permite:
- Visualizar el dataset en tiempo real.
- Explorar métricas de rendimiento de los modelos.
- Predecir la calidad del vino mediante entrada manual de parámetros.
