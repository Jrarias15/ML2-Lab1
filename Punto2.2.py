
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta = "C:/Users/jei_s/OneDrive/Documents/Especializacion/ML2/fotos"  # Cambia esta ruta a la ubicación real de tu carpeta

# Obtener la lista de nombres de archivos en la carpeta
nombres_archivos = os.listdir(ruta_carpeta)

# Lista para almacenar las matrices de las imágenes
matrices_imagenes = []

# Recorrer cada archivo en la carpeta
for nombre_archivo in nombres_archivos:
    # Ruta completa de la imagen
    ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)

    #Convierte las imagenes a escala de grises y la redimesiona a 256 x 256
    imagen = Image.open(ruta_imagen).convert('L').resize((256, 256))

    # Cargar la imagen como una matriz de numpy
    imagen = np.array(imagen) 
    
    # Agregar la matriz al listado
    matrices_imagenes.append(imagen)
    

# Calcular el promedio elemento por elemento de todas las matrices
promedio_matriz = np.mean(matrices_imagenes, axis=0).astype(np.uint8)


# Convertir la matriz promedio en una imagen
imagen_promedio = Image.fromarray(promedio_matriz)


# Mostrar la imagen promedio
plt.imshow(np.asarray(imagen_promedio), cmap='gray')
plt.title('Imagen Promedio')
plt.axis('off')
plt.show()
# Guardar la imagen procesada
imagen_promedio.save('C:/Users/jei_s/OneDrive/Documents/Especializacion/ML2/Imagen_promedio.jpg')
# Cargar tu imagen
ruta_mi_rostro = "C:/Users/jei_s/OneDrive/Documents/Especializacion/ML2/fotos/jeison_arias.jpg" 
mi_rostro = np.array(Image.open(ruta_mi_rostro))

# Calcular la diferencia (distancia) entre mi rostro y el promedio
diferencia = np.linalg.norm(mi_rostro - promedio_matriz)

# Mostrar la distancia
print(f"La distancia entre mi rostro y el promedio es: {diferencia}")