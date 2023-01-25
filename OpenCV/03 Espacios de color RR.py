#!/usr/bin/env python
# coding: utf-8

###########################
# 03 Espacios de color ####
###########################
'''# # **Color Spaces**
In this lesson we'll learn to:
1. View the individual channels of an RGB Image
2. Manipulate a color space
3. Introduce HSV Color Spaces
'''

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


# Carga nuestra imagen de entrada
image = cv2.imread('./images/castara.jpeg')

# Use cv2.split para obtener cada espacio de color por separado
# separa los tuneles en los componenetes de arbol azúl, verde y rojo, conviertiéndose en imágenes bidimensionales
B, G, R = cv2.split(image)
print(B.shape)  # (1280, 960)
print(G.shape)  # (1280, 960)
print(R.shape)  # (1280, 960)

'''Cada espacio de color que esté encendido se verá como una escala de grises ya que carece de los otros canales de 
color, esto es porque tiene sólo una dimensión, son sólo las intensidades en grado de componente de color azul
'''
imshow("Blue Channel Only", B)

import numpy as np

'''Vamos a crear el arbol de la imagen de la dimensión del árbol vamos a hacer todos los otros componentes de color
a cero menos el que queremos visualizar, mediante la siguiente matriz'''
# Vamos a crear una matriz de ceros con dimensiones de la imagen h x w
zeros = np.zeros(image.shape[:2], dtype = "uint8")

imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))

#####
# por otro lado, recargamos la imagen Original
image = cv2.imread('./images/castara.jpeg')

# La función 'dividir' de OpenCV divide la imagen en cada índice de color
B, G, R = cv2.split(image)

# Rehagamos una copia de la imagen original, observando que se muestra la misma imagen
merged = cv2.merge([B, G, R])
imshow("Merged", merged)


# Ampliemos el color azul, se ve extraño
merged = cv2.merge([B+100, G, R])
imshow("Blue Boost", merged)


# ## **The HSV Color Space**
'''#  en vez de usar una combinación de los colores RGB, sua un mapa de color llamado tono (HUE) del azul al amarillo con
la intensidad, que es el brillo pudiendo ver hacia abajo los colores más oscuros, y la saturación, que te dice los
conflictos alimentados en apelación, volíendose más rico y profundo a medida que avanza.
básicamente usando este esquema hay una fomra diferente de representar los colores de diferentes espacios de color'''
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f2/HSV_color_solid_cone.png)
# - Matiz HUE: 0 - 179
# - Saturación: 0 - 255
# - Valor (Intensidad): 0 - 255

# Recargamos la imagen
image = cv2.imread('./images/castara.jpeg')

# convertimos a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)

# #### Esto se ve extraño... ¿por qué?
# Porque nuestra función de trazado fue diseñada solo para imágenes RGB, no para HSV

plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
plt.show()


# ### **Veamos cada tipo de canal en la representación del espacio de color HSV**
# Volviendo a ver la representación RGB mediante el uso de indexación
#  HUE es en realidad el color naranja, para que puedas ver la arena y los árboles ( intensidad)
imshow("Hue", hsv_image[:, :, 0])  #
imshow("Saturation", hsv_image[:, :, 1])  # cuanto mas brillante en la saturación
imshow("Value", hsv_image[:, :, 2])  # intensidad de brillo
