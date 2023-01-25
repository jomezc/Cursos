#!/usr/bin/env python
# coding: utf-8
#####################################
# 20 Comparación de imágenes** ######
#####################################

# 1. Comparar imágenes utilizando el error cuadrático medio (MSE)
# 2. Comparar imágenes usando similitud estructural
# La diferencia entre las imágenes es bastante importante y tiene muchos casos de uso.
# Uno de ellos, sencillo de entender, es la detección de movimiento. Puede usar fácilmente cambios en las imágenes
# para detectar cuándo ha habido movimiento.


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')

get_ipython().system('unzip -qq images.zip')'''


# #### **Error cuadrático medio (MSE)**
#
# El MSE entre las dos imágenes es la suma de la diferencia al cuadrado entre las dos imágenes. Esto se puede
# implementar fácilmente con numpy.
# Cuanto menor sea el MSE más parecidas son las imágenes.


def mse(image1, image2):
    # Las imágenes deben tener la misma dimensión
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


# #### **Vamos a obtener 3 imágenes**
#
# 1. Fuegos artificiales1
# 2. Fuegos artificiales1 con brillo mejorado
# 3. Fuegos artificiales2


fireworks1 = cv2.imread('images/fireworks.jpeg')
fireworks2 = cv2.imread('images/fireworks2.jpeg')

# aumentamos el brillo de una imagen para su comparación
M = np.ones(fireworks1.shape, dtype = "uint8") * 100 
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)


def compare(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print('MSE = {:.2f}'.format(mse(image1, image2)))
    #  la función de structural_similarity es una función que sesga las métricas, lo que significa que es un algoritmo
    #  complicado que nos da básicamente estas similitudes estructurales basadas en relaciones de vecindad entre
    #  matrices para decir la diferencia. 1.0 misma imagen, caunto mas baja mas diferencias
    print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


# Cuando son iguales
compare(fireworks1, fireworks1)


compare(fireworks1, fireworks2)

compare(fireworks1, fireworks1b)

compare(fireworks2, fireworks1b)

