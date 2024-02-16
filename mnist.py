import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Cargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784', version=1, data_home="./datasets", cache=True)

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

#Clase para definir el servidor http. Solo recibe solicitudes POST.
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")

        #Obtener datos de la peticion y limpiar los datos
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)
        
    

        # Realizar transformacion para dejar igual que los ejemplos que usa MNIST
        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(1, -1)  # Cambiar la forma a una fila con múltiples columnas
        print(arr)

        # Realizar y obtener la prediccion
        prediction_values = model.predict(arr)
        predicted_label = str(prediction_values[0])
        print("Prediccion final: " + predicted_label)

        
        #Regresar respuesta a la peticion HTTP
        self.send_response(200)
        #Evitar problemas con CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(predicted_label.encode())
        

#Iniciar el servidor en el puerto 8000 y escuchar por siempre
#Si se queda colgado, en el admon de tareas buscar la tarea de python y finalizar tarea
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()
