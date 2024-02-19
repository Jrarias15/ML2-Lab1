import numpy as np

from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from unsupervised_package.SVD import SVDApproximation
from unsupervised_package.PCA import PCA
from unsupervised_package.TSNE import TSNE

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
svd_model = SVDApproximation(k=2)
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
X_cp = PCA_model.transform(X_filtered)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_cp, y_filtered, test_size=0.2, random_state=42)

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

#Usando TSNE
n_componentes = 2


#Reduce the number of samples to reduce the processing time
subset_size = 5000
subset_indices = np.random.choice(len(X_filtered), size=subset_size, replace=False)
X, y = X_filtered.iloc[subset_indices], y_filtered.iloc[subset_indices]

X1=np.array(X)

#Normalize data. I've dcided not to use it, there was not significant improvement.
#scaler = StandardScaler()
#X_normalized = scaler.fit_transform(X1)

tsne_model = TSNE(n_components=n_componentes, max_iter=1000)
X_transformed = tsne_model.fit_transform(X1)


# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_filtered, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión logística
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Resultados TSNE')
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', report)

