import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../data/raw/winequality.csv')

# Boxplots para todas las variables numéricas
plt.figure(figsize=(15, 20))
for i, column in enumerate(df.drop('quality', axis=1).columns):
    plt.subplot(4, 3, i+1)
    sns.boxplot(data=df, y=column, color='#8B0000')
    plt.title(f'Boxplot de {column}')
plt.tight_layout()
plt.show()

# Matriz de correlación entre las variables
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()

# Scatterplots de Variables Clave vs Calidad
key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(data=df, x=feature, y='quality', alpha=0.5, color = '#8B0000')
    plt.title(f'{feature} vs Quality')
plt.tight_layout()
plt.show()

# Análisis de Outliers Multivariante
# Pairplot de Variables Clave (key features)
sns.pairplot(df, vars=key_features, hue='quality');

# Distribución de la Calidad
# Histograma de Calidad
plt.figure(figsize=(8, 5))
sns.histplot(df['quality'], bins=6, kde=True, color="#8B0000")
plt.title('Distribución de la Calidad del Vino')
plt.show()

df['quality_category'] = pd.cut(df['quality'],
                               bins=[0, 4, 6, 10],
                               labels=['Baja', 'Media', 'Alta'])

# Eliminamos outliers (solo en columnas numéricas)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
mask = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
df_final = df[~mask.any(axis=1)]

df_final.to_csv('df_final.csv', index=False)