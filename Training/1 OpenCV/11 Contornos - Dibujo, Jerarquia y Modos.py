#!/usr/bin/env python
# coding: utf-8

# # **Contornos**
#
# ####**En esta lección aprenderemos:**
# 1. Usando findContours
# 2. Dibujo de contornos
# 3. Jerarquía de Contornos
# 4. Modos de contorno (simple vs aproximado)

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


# ## **¿Qué son los contornos?**
# Los contornos son líneas o curvas continuas que limitan o cubren el límite total de un objeto.

# Carguemos una imagen simple de placa de matrícula
image = cv2.imread('images/LP.jpg')
imshow('Input Image', image)


# #### **Aplicando cv2.findContours()**
# ```cv2.findContours(imagen, modo de recuperación, método de aproximación)```
#
# **Modos de recuperación**
# - **RETR_LIST** - Recupera todos los contornos, pero no crea ninguna relación padre-hijo. Padres e hijos son iguales bajo esta regla, y son solo contornos. es decir, todos pertenecen al mismo nivel de jerarquía.
# - **RETR_EXTERNAL** - devuelve solo banderas externas extremas. Todos los contornos secundarios se dejan atrás.
# - **RETR_CCOMP** - Esta bandera recupera todos los contornos y los organiza en una jerarquía de 2 niveles. es decir, los contornos externos del objeto (es decir, su límite) se colocan en la jerarquía-1. Y los contornos de los agujeros dentro del objeto (si los hay) se colocan en la jerarquía-2. Si hay algún objeto dentro de él, su contorno se coloca nuevamente en la jerarquía-1 solamente. Y su agujero en la jerarquía-2 y así sucesivamente.
# - **RETR_TREE** - Recupera todos los contornos y crea una lista de jerarquía familiar completa.
#
# **Opciones de método de aproximación**
# - cv2.CHAIN_APPROX_NONE – Almacena todos los puntos a lo largo de la línea (¡ineficiente!)
# - cv2.CHAIN_APPROX_SIMPLE – Almacena los puntos finales de cada línea


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Encontrar contornos
# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))

contours[0]

# #### **¿Qué sucede si no establecemos un umbral? Cosas malas..**


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow('After Grayscaling', gray)

# Encontrar contornos
contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
#cv2.drawContours(imagen, contornos, -1, (0,255,0), grosor = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))


# # **NOTA: Para que findContours funcione, el fondo debe ser negro y de primer plano (es decir, el texto o los objetos)**
# #### De lo contrario, deberá invertir la imagen utilizando **cv2..bitwise_not(input_image)**
# #### **Podemos usar Canny Edges en lugar de Thresholding**

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bordes astutos
edged = cv2.Canny(gray, 30, 200)
imshow('Canny Edges', edged)

# Encontrar contornos
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))


## ## **Recuerda estos pasos para contornear**
#
# 1. Escala de grises
# 2. Detección de umbral o Canny Edge para binarizar la imagen
#
# **Nota:** Se recomienda desenfocar antes del Paso 2 para eliminar contornos ruidosos

# # **Modos de recuperación**
#
# Documento oficial: https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
#
# **Jerarquía**
#
# Esta matriz almacena 4 valores para cada contorno:
# - El primer término es el índice del siguiente contorno
# - El segundo término es el índice del contorno anterior
# - El tercer término es el índice del contorno padre
# - Cuarto término es el índice del contorno hijo

# ### **RETR_LIST**
# Recupera todos los contornos, pero no crea ninguna relación padre-hijo. Padres e hijos son iguales bajo esta regla, y
# son solo contornos. es decir, todos pertenecen al mismo nivel de jerarquía.

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)


# ### **RETR_EXTERNO**

# Devuelve solo banderas exteriores extremas. Todos los contornos secundarios se dejan atrás.


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image, size = 10)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)


# ### **RETR_CCOMP**
#
# Recupera todos los contornos y los organiza en una jerarquía de 2 niveles. es decir, los contornos externos del objeto
# (es decir, su límite) se colocan en la jerarquía-1. Y los contornos de los agujeros dentro del objeto (si los hay)
# se colocan en la jerarquía-2. Si hay algún objeto dentro de él, su contorno se coloca nuevamente en la jerarquía-1
# solamente. Y su agujero en la jerarquía-2 y así sucesivamente.


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours
contours, hierarchy = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)


# ### **RETR_ÁRBOL**
# Recupera todos los contornos y crea una lista de jerarquía familiar completa.


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)


# ## **Modos de contorno**
#
# #### **CHAIN_APPROX_NONE**


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
  print(len(c))


#
# #### **CADENA_APPROX_SIMPLE**


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la entrada
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
  print(len(c))


