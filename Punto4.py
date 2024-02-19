from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from unsupervised_package.SVD import SVDApproximation

ruta_rostro = "C:/Users/jei_s/OneDrive/Documents/Especializacion/ML2/fotos/jeison_arias.jpg" 
rostro = np.array(Image.open(ruta_rostro))
imagen_rostro = Image.fromarray(rostro)

imagenes = []
description = []
for i in range (8):    
    svd_model = SVDApproximation(k=(4*i))
    svd_model.fit(imagen_rostro)
    A_approximated = svd_model.transform(imagen_rostro)

    #Convertir el resultado en imagen
    imagen_comp_principales = Image.fromarray(A_approximated)
    imagenes.append(imagen_comp_principales)
    description.append(f'K= {4*i}')

fig, axs = plt.subplots(2, 4, figsize=(8, 4))
description
for i, ax in enumerate(axs.flatten()):
    ax.imshow(np.asarray(imagenes[i]), cmap='gray')  # 'gray' para imágenes en escala de grises
    ax.set_title(description[i])
    ax.axis('off')  # Desactivar ejes

plt.show()


    
