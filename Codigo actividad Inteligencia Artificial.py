import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset creado directamente en el código
datos = pd.DataFrame({
    'hora': [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'pasajeros': [850, 920, 800, 300, 250, 400, 450, 300, 500, 700, 850, 980, 750, 400],
    'lluvia': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    'congestion': [
        'Alta', 'Alta', 'Alta', 'Baja', 'Baja',
        'Media', 'Media', 'Baja', 'Media',
        'Media', 'Alta', 'Alta', 'Media', 'Baja'
    ]
})

# Variables de entrada
X = datos[['hora', 'pasajeros', 'lluvia']]

# Variable a predecir
y = datos['congestion']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Crear modelo
modelo = DecisionTreeClassifier(random_state=42)

# Entrenar modelo
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print("Precisión del modelo:", round(precision * 100, 2), "%")

# Nueva predicción
nueva_muestra = [[18, 950, 1]]

resultado = modelo.predict(nueva_muestra)

print("Predicción para hora 18, 950 pasajeros y lluvia:")
print(resultado[0])

