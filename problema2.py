import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Importar Bibliotecas y Cargar Datos
# Importar librerías Python.
# Cargar los datos en un dataframe.
df = pd.read_csv("Loan_Data.csv")

# Paso 2: Preparación de Datos
# Determinar la cantidad total de registros y de variables (atributos).
total_registros = df.shape[0]
total_atributos = df.shape[1]

print(f"Total de registros: {total_registros}")
print(f"Total de atributos: {total_atributos}")

# Determinar los tipos de datos por cada variable (atributo).
tipos_de_datos = df.dtypes
print(tipos_de_datos)

# Eliminar la columna ‘Loan_ID’.
df = df.drop(columns=['Loan_ID'])

# Si se da el caso de datos faltantes en las variables numéricas sustituirlos por la media respectiva.
variables_numericas = df.select_dtypes(include=['float64', 'int64']).columns
for variable in variables_numericas:
    df[variable].fillna(df[variable].mean(), inplace=True)

# Si se da el caso de datos faltantes en las variables categóricas sustituirlos por la moda respectiva.
variables_categoricas = df.select_dtypes(include=['object']).columns
for variable in variables_categoricas:
    df[variable].fillna(df[variable].mode()[0], inplace=True)

# Convertir a valores numéricos los datos de las variables categóricas, es decir, numerizar las variables categóricas.
df = pd.get_dummies(df, drop_first=True)

# Paso 3: Separar los conjuntos de variables predictoras (características) y de la variable objetivo (etiqueta).
X = df.drop(columns=['Loan_Status_Y'])
y = df['Loan_Status_Y']

# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba.
# Utilice una proporción 70% entrenamiento - 30% prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 5: Crear el modelo de árbol de decisión preliminar
# Utilizar los siguientes parámetros y argumentos para crear el modelo preliminar:
# criterion="entropy"
# max_depth=10
# random_state=42
# En caso de que los valores de la variable objetivo estén desbalanceados, utilizar class_weight='balanced'
modelo = DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight='balanced', random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Paso 6: Evaluar el modelo de predicción preliminar
# Predecir los resultados en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular e interpretar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de confusion')
plt.show()

# Calcular e interpretar la métrica Exactitud (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")

# Calcular e interpretar la métrica Precisión (Precision)
precision = precision_score(y_test, y_pred)
print(f"Precisión (Precision): {precision}")

# Calcular e interpretar la métrica Sensibilidad (Recall)
recall = recall_score(y_test, y_pred)
print(f"Sensibilidad (Recall): {recall}")

# Paso 7: Optimizar el modelo de árbol de decisión
# Determinar la profundidad máxima óptima.

param_grid = {'max_depth': np.arange(1, 21)}
tree = DecisionTreeClassifier(criterion="entropy", class_weight='balanced', random_state=42)
tree_cv = GridSearchCV(tree, param_grid, cv=5)
tree_cv.fit(X_train, y_train)

print(f"Profundidad máxima óptima: {tree_cv.best_params_['max_depth']}")

# Crear el árbol de decisión con base al valor óptimo para el hiperparámetro profundidad máxima.
modelo_optimo = DecisionTreeClassifier(criterion="entropy", max_depth=tree_cv.best_params_['max_depth'], class_weight='balanced', random_state=42)
modelo_optimo.fit(X_train, y_train)

# Evaluar el modelo optimizado
y_pred_optimo = modelo_optimo.predict(X_test)

# Calcular e interpretar la matriz de confusión del modelo optimizado
cm_optimo = confusion_matrix(y_test, y_pred_optimo)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimo, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de confusion, modelo optimizado')
plt.show()

# Calcular e interpretar la métrica Exactitud (Accuracy) del modelo optimizado
accuracy_optimo = accuracy_score(y_test, y_pred_optimo)
print(f"Precisión del modelo optimizado: {accuracy_optimo}")

# Calcular e interpretar la métrica Precisión (Precision) del modelo optimizado
precision_optimo = precision_score(y_test, y_pred_optimo)
print(f"Precisión (Precision) del modelo optimizado: {precision_optimo}")

# Calcular e interpretar la métrica Sensibilidad (Recall) del modelo optimizado
recall_optimo = recall_score(y_test, y_pred_optimo)
print(f"Sensibilidad (Recall) del modelo optimizado: {recall_optimo}")

# # Crear un nuevo dataframe con las características del nuevo cliente
# prediccion = modelo_optimo.predict(crear_nuevo_cliente)
# resultado = 'Aprobado' if prediccion[0] == 1 else 'No Aprobado'
# print(f"El préstamo para el nuevo cliente está: {resultado}")


#¿Qué fue lo más complicado?
#Como en todo analisis de datos, el proceso de preparacion de datos supone desafios en cuanto a identificar y tratar datos faltantes, sobre todo escoger una estrategia adecuadar para reemplazar. Sumado a implementar LOF para detectar los "outliers", ya que requiere dominar la comprension e interpretacion de los parametros y resultados.
#¿Cómo se resolvió?
#Similar a la solemne 1, se realizó el análisis y preparación de datos, seguido de detección y tratamiento de datos. Además, se implementó el método LOF y validación cruzada para un árbol de decisión, usando métodos como grid search y cross-validation para evaluar el rendimiento del modelo y probar su optimización.
#¿Qué se aprendió? 
#Al igual que la solemne 1, se reafirma la idea de la importancia de la preparación y la calidad de los datos, sobre todo aplicado a machine learning, dado el impacto de los outliers en el rendimiento de un modelo. En esencia, se aprendió a valorar y dar importancia al proceso de optimización.