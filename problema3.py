# Paso 1: Importar Bibliotecas y Cargar Datos
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('heart_disease_data.csv')  # Asegúrate de tener el archivo 'heart.csv' en tu directorio de trabajo

# Paso 2: Preparación de Datos (Preprocesamiento)
# Identificar variables numéricas y categóricas
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Pipeline para características numéricas
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para características categóricas
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ]
)

# Paso 3: Separar los conjuntos de variables predictoras y de la variable objetivo
X = data.drop('target', axis=1)
y = data['target']

# Aplicar preprocesamiento
X_preprocessed = preprocessor.fit_transform(X)

# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# Paso 5: Construir el modelo óptimo de red neuronal utilizando validación cruzada
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# Validación cruzada
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(model, X_train, y_train, cv=kfold)

print("Modelo: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Paso 6: Generar el gráfico de la mejor performance de la red neuronal
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

# Paso 7: Evaluar el modelo con las métricas pertinentes
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print("Exactitud en datos de prueba: %.2f%%" % (accuracy_score(y_test, y_pred)*100))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Paso 8: Realizar predicciones con nuevos datos
#
# Preprocesar los nuevos datos
#new_data_preprocessed = preprocessor.transform(new_data)
# Predecir
# new_prediction = model.predict(new_data_preprocessed)
# print("Predicción para el nuevo paciente: ", "Enfermedad cardíaca" if new_prediction[0] == 1 else "Sin enfermedad cardíaca")