#!/usr/bin/env python
# coding: utf-8

###################################################################
# # 06 Escalado, cambio de tamaño, interpolaciones y recorte** ####
###################################################################
# **En esta lección aprenderemos:**
# 1. Cómo redimensionar y escalar imágenes
# 2. Pirámides de imágenes
# 3. Recortar

# ### **Cambio de tamaño**
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Resizing.png)
# 
# Cambiar el tamaño es una función simple que ejecutamos usando la función cv2.resize, sus argumentos son:

# cv2.resize(imagen, dsize(tamaño de la imagen de salida), escala x, escala y, interpolación)
# - si dsize es Ninguno, la imagen de salida se calcula en función de la escala usando la escala x e y

# la interpolación es básicamente un algoritmo para encontrar un valor entre dos puntos. si tuviéramos unos puntos por
# una ruta de gps la interpolación adivinará puntos intermedios entre los originales del camino, aportando información
# adicional, e suna forma de agregar más datos a los existentes para conectar los puntos existentes ( en el ejemplo)
# si estamos agrandando una imagen, estamos tratando de adivinar los puntos que se tomarán en una nueva dimensión.
# algorítmicamente adivina la mejor suposición
# #### **Lista de métodos de interpolación, las diferentes fórmulas que suelen aplicarse:**
# - cv2.INTER_AREA- Bueno para reducir o reducir el muestreo
# - cv2.INTER_NEAREST - Más rápido
# - cv2.INTER_LINEAR- Bueno para hacer zoom o muestreo ascendente (predeterminado)
# - cv2.INTER_CUBIC- Mejor
# - cv2.INTER_LANCZOS4 - El Mejor


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ### **Tipos de métodos de reescalado en OpenCV**
#
# - **INTER_NEAREST** – una interpolación de vecino más cercano
# - **INTER_LINEAR** – una interpolación bilineal (usada por defecto)
# - **INTER_AREA** – remuestreo usando relación de área de píxeles. Puede ser un método preferido para la destrucción
#                    de imágenes, ya que brinda resultados sin muaré. Pero cuando se amplía la imagen, es similar al
#                    método INTER_NEAREST.
# - **INTER_CUBIC**: una interpolación bicúbica sobre una vecindad de 4×4 píxeles
# - **INTER_LANCZOS4**: una interpolación de Lanczos sobre un vecindario de 8×8 píxeles
#
# Vea más sobre su desempeño - https://chadrick-kwag.net/cv2-resize-interpolation-methods/

# carga nuestra imagen de entrada
image = cv2.imread('images/oxfordlibrary.jpeg')
imshow("Scaling - Linear Interpolation", image)

# Si no se especifica ninguna interpolación, cv.INTER_LINEAR se usa por defecto
# Hagamos nuestra imagen 3/4 de su tamaño original
# vamos a usar los efectos del argumento y la forma para reducir la imagen en un 75% (0.75 de ancho y alto)
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
imshow("0.75x Scaling - Linear Interpolation", image_scaled)

# Dupliquemos el tamaño de nuestra imagen
img_scaled2 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
imshow("2x Scaling - Inter Cubic", img_scaled2)

# Dupliquemos el tamaño de nuestra imagen usando la interpolación inter_nearest
img_scaled3 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
imshow("2x Scaling - Inter Nearest", img_scaled3)

# Sesguemos el cambio de tamaño estableciendo dimensiones exactas
img_scaled4 = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
imshow("Scaling - Inter Area", img_scaled4)


# ## **Imagen de pirámides**
# Es una operación mucho más rápida, siendo una forma rápida de ampliar una imagen escalada
# Básicamente, duplique recuzca la mitad del tamaño
image = cv2.imread('images/oxfordlibrary.jpeg')

smaller = cv2.pyrDown(image)  # reduce la mitad
larger = cv2.pyrUp(smaller)  # dobla la imagen

imshow("Original", image)
imshow('Smaller', smaller)
imshow('Larger', larger)

even_smaller = cv2.pyrDown(smaller)
imshow('Even Smaller', even_smaller)


# # **Recorte**
# es una técnica muy útil especialmente con detectores de objetos o OCR donde tienes que recortar segmentos de la imagen
image = cv2.imread('images/oxfordlibrary.jpeg')

# Obtenga las dimensiones de nuestra imagen
height, width = image.shape[:2]

# Obtengamos las coordenadas del píxel inicial (arriba a la izquierda del rectángulo de recorte)
# usando 0.25 para obtener la posición x,y que está 1/4 por debajo de la parte superior izquierda (0,0)

start_row, start_col = int(height * .25), int(width * .25)

# Obtengamos las coordenadas del píxel final (abajo a la derecha)
end_row, end_col = int(height * .75), int(width * .75)

# Simplemente use la indexación para recortar el rectángulo que deseamos
# hace lo que se supone, es decir recorta la imagen
cropped = image[start_row:end_row, start_col:end_col]

imshow("Original Image", image)

# La función cv2.rectangle dibuja un rectángulo sobre nuestra imagen (operación in situ)
copy = image.copy()
cv2.rectangle(copy, (start_col,start_row), (end_col,end_row), (0,255,255), 10)

imshow("Area we are cropping", copy)

imshow("Cropped Image", cropped) 
