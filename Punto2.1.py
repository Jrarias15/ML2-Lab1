
from PIL import Image
import numpy as np
from io import BytesIO

import requests
import matplotlib.pyplot as plt

# Cargar la imagen desde la biblioteca content de Colab
imagen_colab = Image.open('C:/Users/jei_s/Downloads/foto.jpg')

# Convertir la imagen a escala de grises
imagen_grayscale = imagen_colab.convert('L')

# Redimensionar a 256x256
imagen_redimensionada = imagen_grayscale.resize((256, 256))

# Mostrar la imagen original
plt.imshow(np.asarray(imagen_colab), cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Mostrar la imagen en escala de grises y redimensionada
plt.imshow(np.asarray(imagen_redimensionada), cmap='gray')
plt.title('Imagen Procesada')
plt.axis('off')
plt.show()

# Guardar la imagen procesada
imagen_redimensionada.save('C:/Users/jei_s/OneDrive/Documents/Especializacion/ML2/jeison_arias.jpg')