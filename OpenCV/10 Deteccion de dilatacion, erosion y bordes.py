#!/usr/bin/env python
# coding: utf-8
#####################################################
# 10 Detección de dilatación, erosión y bordes ######
#####################################################
# # **Detección de dilatación, erosión y bordes**
# ####**En esta lección aprenderemos:**
# - **Dilatación**: agrega píxeles a los límites de los objetos en una imagen
# - **Erosión**: elimina píxeles en los límites de los objetos en una imagen
# - **Apertura** - Erosión seguida de dilatación
# - **Cierre** - Dilatación seguida de erosión
# 5. Detección de borde astuto

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

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-11-15%20at%205.19.08%20pm.png)

image = cv2.imread('images/opencv_inv.png', 0)
imshow('Original', image)

# Definamos el tamaño de nuestro kernel, es una matriz que usamos para la convolución 2D quw se realiza en estas
# funciones
kernel = np.ones((5, 5), np.uint8)

# Ahora erosionamos
erosion = cv2.erode(image, kernel, iterations = 1)
imshow('Erosion', erosion)

# Dilatar aqui
dilation = cv2.dilate(image, kernel, iterations = 1)
imshow('Dilation', dilation)

# Apertura - Bueno para eliminar el ruido
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
imshow('Opening',opening)

# Cierre - Bueno para eliminar el ruido
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
imshow('Closing',closing)


# ## **Detección de bordes astutos**

# Los bordes de una imagen digital se pueden definir como transiciones entre dos regiones de
# niveles de gris significativamente distintos. Suministran una valiosa información sobre las
# fronteras de los objetos y puede ser utilizada para segmentar la imagen, reconocer objetos, etc.
# La mayoría de las técnicas para detectar bordes emplean operadores locales basados en distintas
# aproximaciones discretas de la primera y segunda derivada (el cambio a oscuro y el retorno a claro) de los niveles
# de grises de la imagen.

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-11-15%20at%205.24.15%20pm.png)
# - El primer argumento es nuestra imagen de entrada.
# - El segundo y tercer argumento son nuestro minVal y maxVal respectivamente.
# - El cuarto argumento (opcional) es opening_size. Es el tamaño del núcleo Sobel utilizado para encontrar gradientes
#   de imagen. Por defecto es 3.
#
# La detección de bordes necesita un umbral para indicar qué diferencia/cambio debe contarse como borde

image = cv2.imread('images/londonxmas.jpeg',0)

# Canny Edge Detection utiliza valores de gradiente como umbrales
# El primer gradiente de umbral
canny = cv2.Canny(image, 50, 120)
imshow('Canny 1', canny)

# Los umbrales de borde ancho esperan muchos bordes
canny = cv2.Canny(image, 10, 200)
imshow('Canny Wide', canny)

# Umbral estrecho, espere menos bordes
canny = cv2.Canny(image, 200, 240)
imshow('Canny Narrow', canny)

canny = cv2.Canny(image, 60, 110)
imshow('Canny 4', canny)

# Luego, debemos proporcionar dos valores: umbral1 y umbral2. Cualquier valor de gradiente mayor que el umbral2
# se considera una ventaja. Cualquier valor por debajo del umbral 1 se considera que no es un borde.
# Los valores entre el umbral 1 y el umbral 2 se clasifican como bordes o no bordes en función de cómo Las intensidades
# están “conectadas”. En este caso, cualquier valor de degradado por debajo de 60 se considera sin bordes
# mientras que cualquier valor por encima de 120 se considera borde.

# #### **Astucia automática** ( sacado de stackoverflow)
def autoCanny(image):
  # Encuentra umbrales óptimos basados en la mediana de la intensidad de píxeles de la imagen
  blurred_img = cv2.blur(image, ksize=(5,5))
  med_val = np.median(image) 
  lower = int(max(0, 0.66 * med_val))
  upper = int(min(255, 1.33 * med_val))
  edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
  return edges

auto_canny = autoCanny(image)
imshow("auto canny", auto_canny)
