#!/usr/bin/env python
# coding: utf-8

# # **Operaciones aritméticas y bit a bit**
#
# #### **En esta lección aprenderemos:**
# 1. Operaciones aritméticas, aquellas que nos permiten sumar o restar la intensidad o los valores de la imagen
# 2. Operaciones bit a bit

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

'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ## **Operaciones aritméticas**
# Son operaciones sencillas que nos permiten sumar o restar directamente a la intensidad del color.
# Calcula la operación por elemento de dos matrices. El efecto general es aumentar o disminuir el brillo.


# Agregar coma cero en cv2.imread carga nuestra imagen como una imagen en escala de grises
image = cv2.imread('images/liberty.jpeg', 0)  # 0 es como escala de grises
imshow("Grayscaled Image",  image)
print(image)


# Crea una matriz de unos con el tamaño de la imagen, luego multiplícala por un escalador de 100
# Esto da una matriz con las mismas dimensiones de nuestra imagen con todos los valores siendo 100
M = np.ones(image.shape, dtype = "uint8") * 100 

print(M)

# #### **Brillo creciente**
# Usamos esto para agregar esta matriz M, a nuestra imagen, la función respeta los valores de 0 a 255 dejando el máximo
# Note el aumento en el brillo
added = cv2.add(image, M)
imshow("Increasing Brightness", added)

# Ahora si lo acabamos de agregar, pero al no usar la funcíón el valor sobrepasa el 255 con lo que se resetea a
# 0 sumandole la diferencia por ejemplo si es 288 pues 33 con lo que no se ve como se espera
added2 = image + M 
imshow("Simple Numpy Adding Results in Clipping", added2)


# #### **Reducción del brillo**

# Así mismo también podemos restar
# Note la disminución en el brillo
subtracted = cv2.subtract(image, M)
imshow("Subtracted", subtracted)

subtracted = image - M  # aquí pasa lop mismo que antes pero al reves los valores se quedan negativos y al no permitirse
# van de 255 hacia abajo
imshow("Subtracted 2", subtracted)


# ## **Operaciones bit a bit y enmascaramiento**
# Para demostrar estas operaciones, creemos algunas imágenes simples
# Si se pregunta por qué solo dos dimensiones, bueno, esta es una imagen en escala de grises,
# Hacer un cuadrado

# Hacer un cuadrado
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
imshow("square", square)

# Haciendo una elipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
imshow("ellipse", ellipse)


# ### **Experimentando con algunas operaciones bit a bit como AND, OR, XOR y NOT**
# Muestra solo donde se cruzan, es decir donde ambos tienen info distinta de 0 ( 0 negro 255 blanco
# soilo 1 si los dos 1)
And = cv2.bitwise_and(square, ellipse)
imshow("AND", And)

# Muestra la información de ambos (1 si alguno de los 2 1) dónde está el cuadrado o la elipse
bitwiseOr = cv2.bitwise_or(square, ellipse)
imshow("bitwiseOr", bitwiseOr)

# Muestra dónde existen por sí mismos (1 si solo uno de ellos 1)
bitwiseXor = cv2.bitwise_xor(square, ellipse)
imshow("bitwiseXor", bitwiseXor)

# Muestra todo lo que no es parte del cuadrado ( lo contrario)
bitwiseNot_sq = cv2.bitwise_not(square)
imshow("bitwiseNot_sq", bitwiseNot_sq)

# Observe que la última operación invierte la imagen totalmente




