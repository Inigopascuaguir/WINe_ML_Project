import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, confusion_matrix
from imblearn.over_sampling import SMOTE

df_final = pd.read_csv('../data/processed/df_final.csv')
df_final.head(10)

#Codificar el target y dividir en train/test dataset
le = LabelEncoder()
X = df_final.drop(['quality', 'quality_category'], axis=1)
y = le.fit_transform(df_final['quality_category'])

#Entrenamos el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicamos SMOTE para corregir el desbalance, solo en train
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=le.classes_)
    }

# Métricas
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print("Classification Report:")
    print(result['report'])
    print("-" * 60)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_final.drop(['quality', 'quality_category'], axis=1))

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Evaluar
df_final['cluster'] = clusters
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.2f}")
print(pd.crosstab(df_final['quality_category'], df_final['cluster']))

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

search_space = [
    {
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'scaler': [StandardScaler()],
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    },
    {
        'scaler': [StandardScaler()],
        'classifier': [GradientBoostingClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1]
    }
]

clf = GridSearchCV(
    estimator=pipe,
    param_grid=search_space,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

clf.fit(X_res, y_res)

# Resultados
best_model = clf.best_estimator_
best_params = clf.best_params_
best_score = clf.best_score_

# Imprimir métricas
print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score:.2f}")

# Crear un DataFrame con los resultados
metrics_table = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [result['accuracy'] for result in results.values()]
})

# Añadir el resultado del GridSearchCV como una nueva fila
new_row = pd.DataFrame({
    "Model": ["GridSearchCV Best Model"],
    "Accuracy": [best_score]
})

# Concatenar la nueva fila al DataFrame existente
metrics_table = pd.concat([metrics_table, new_row], ignore_index=True)

# Imprimir la tabla
metrics_table

# Crear un DataFrame con las métricas de los modelos supervisados
supervised_metrics = []
for name, result in results.items():
    supervised_metrics.append({
        'Model': name,
        'Accuracy': result['accuracy'],
        'Precision': result['report'].split()[5],
        'Recall': result['report'].split()[6],
        'F1-Score': result['report'].split()[7]
    })

df_supervised = pd.DataFrame(supervised_metrics)

# Crear un DataFrame con las métricas del modelo no supervisado
unsupervised_metrics = pd.DataFrame({
    'Model': ['KMeans'],
    'Silhouette Score': [silhouette_score(X_scaled, clusters)]
})

# Crear un DataFrame con las métricas del GridSearchCV
gridsearch_metrics = pd.DataFrame({
    'Model': ['GridSearchCV'],
    'Best Score': [best_score]
})

# Combinar todos los DataFrames
df_metrics_allfeatures = pd.concat([df_supervised, unsupervised_metrics, gridsearch_metrics], ignore_index=True)

# Mostrar la tabla
df_metrics_allfeatures

df_metrics_allfeatures.to_csv('df_metrics_allfeatures.csv', index=False)

'''import pickle

# Guardar modelos supervisados
for name, model in models.items():
    with open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Guardar el mejor modelo del GridSearchCV
with open('best_model_gridsearchcv.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Guardar el modelo KMeans
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)'''