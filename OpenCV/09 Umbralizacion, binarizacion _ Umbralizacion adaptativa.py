#!/usr/bin/env python
# coding: utf-8

########################################################################
# 09 Umbralización, binarización y umbralización adaptativa ######
########################################################################
# ####**En esta lección aprenderemos:**
# 1. Imágenes binarizadas, estamos conviertiendo a binario los colores, los píxeles de una imagen a 0 o 1, mediante un
#    algoritmo de sesión binaria.
# 2. Métodos de Umbral
# 3. Umbral adaptativo
# 4. Umbral local de SkImage
# In[1]:


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ### **Métodos de umbral**
#  explicación de que la binarización Convirtió pasa a blanco o negro una escala de grises de una imagen en mediante un
#  umbral todo por encima de un cierto umbral se vuelve blanco y por debajo negro mediant eun algortimo , exisitiendo la
#  operación binaria contraria ( en vez de blanco negro y viceversa).
#  El truncamiento es que todo lo que está por encima d eun umbral se convierte en ese valor máximo del umbral
#  TOZERO es que todo lo que es menor que el umbral se vuelve 0 y TOZERO_INV lo contrario
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.57.55%20am.png)
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.58.09%20am.png)
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html


'''
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/scan.jpeg')
'''

# Cargar nuestra imagen en escala de grises
image = cv2.imread('./images/scan.jpg',0)
imshow("Original", image)

# Los valores por debajo de 127 van a 0 o negro, por encima va a 255 (blanco)
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('1 Threshold Binary @ 127', thresh1)

# Los valores por debajo de 127 van a 255 y los valores por encima de 127 van a 0 (inverso de arriba)
ret,thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
imshow('2 Threshold Binary Inverse @ 127', thresh2)

# Los valores por encima de 127 se truncan (se mantienen) en 127 (el argumento 255 no se usa)
ret,thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
imshow('3 THRESH TRUNC @ 127', thresh3)

# Los valores por debajo de 127 van a 0, por encima de 127 no cambian
ret,thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
imshow('4 THRESH TOZERO @ 127', thresh4)

# Inverso de lo anterior, por debajo de 127 no cambia, por encima de 127 pasa a 0
ret,thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
imshow('5 THRESH TOZERO INV @ 127', thresh5)


# #### **parámetros cv2.adaptiveThreshold**
# si queremos calcular automáticamente el umbral, usamos el umbral adaptativo, son pequeños algoritmos que en realidad
# ejecutan algunos cálculos en la imagen y tratan de averiguar el valor umbral óptimo.
# ``**cv2.adaptiveThreshold**(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst``
#
# - **src** – Imagen de origen de un solo canal de 8 bits.
# - **dst** – Imagen de destino del mismo tamaño y del mismo tipo que src .
# - **maxValue**: valor distinto de cero asignado a los píxeles para los que se cumple la condición. Vea los detalles
#                 a continuación.
# - **adaptiveMethod**: algoritmo de umbral adaptable para usar, ADAPTIVE_THRESH_MEAN_C o ADAPTIVE_THRESH_GAUSSIAN_C.
#                       Vea los detalles a continuación.(mejor el segundo)
# - **thresholdType**: tipo de umbral que debe ser THRESH_BINARY o THRESH_BINARY_INV.
# - **blockSize**: tamaño de una vecindad de píxeles que se utiliza para calcular un valor de umbral para el píxel: 3,
#                   5, 7, etc.
# - **C** – Constante restada de la media o media ponderada. Normalmente, es positivo, pero también puede ser cero o
#           negativo.

image = cv2.imread('./images/scan.jpg',0)
imshow("Original", image)

# MANUAL
# Los valores por debajo de 127 van a 0 (negro, todo lo anterior va a 255 (blanco)
# 127 es el umbral y 255 el máximo
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('Threshold Binary', thresh1)

# Uso de umbral adaptativo # 3 y 5 por defecto en la documentación
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 
imshow("Adaptive Mean Thresholding", thresh) 

# otra forma que se explica en la documentación, no es muy intuitivo por el umbral que establece pero funciona bien
_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Otsu's Thresholding", th2) 

# Umbralización de Otsu después del filtrado gaussiano
# Es una buena práctica desenfocar las imágenes ya que elimina el ruido
# imagen = cv2.GaussianBlur(imagen, (3, 3), 0)
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Guassian Otsu's Thresholding", th3) 

# ### **Umbral local de SkImage** USABLE EN LA VIDA REAL MUY BUENO
# umbral_local(imagen, tamaño_bloque, desplazamiento=10)
# La función Threshold_local calcula umbrales en regiones con un tamaño característico ``block_size`` que rodea cada
# píxel (es decir, vecindarios locales). Cada valor de umbral es la media ponderada del vecindario local menos un valor
# de ``compensación``
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

from skimage.filters import threshold_local

image = cv2.imread('./images/scan.jpg')

# Obtenemos el componente Valor del espacio de color HSV, lo necesita esta función
# luego aplicamos un umbral adaptativo a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Apply the threshold operation 
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)


# ### **Por qué es importante desenfocar
# ## **respuesta - ruido *
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/otsu.jpg)
# https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
