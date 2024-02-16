from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


# Cargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784', version=1, data_home="./datasets", cache=True, parser="auto")

# Filtrar los dígitos 0 y 8
indices = (mnist.target == '0') | (mnist.target == '8')
X_filtered = mnist.data[indices]
y_filtered = mnist.target[indices]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión logística
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', report)

#Usando SVD
svd_model = TruncatedSVD(n_components=2)
svd_model.fit(X_filtered)
X_approximated = svd_model.transform(X_filtered)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_approximated, y_filtered, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión logística
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Resultados SVD')
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', report)

#Usando PCA
PCA_model = PCA(n_components=2)
PCA_model.fit(X_filtered)
X_approximated = PCA_model.transform(X_filtered)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_approximated, y_filtered, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión logística
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Resultados PCA')
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', report)
