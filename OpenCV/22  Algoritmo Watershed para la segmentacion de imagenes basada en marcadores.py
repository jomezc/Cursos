#!/usr/bin/env python
# coding: utf-8
######################################################################################
# 22 Algoritmo Watershed para la segmentación de imágenes basada en marcadores ######
######################################################################################
# 1. Cómo utilizar el algoritmo Watershed para la segmentación de imágenes basada en marcadores



# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
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

'''get_ipython().system('wget https://docs.opencv.org/3.4/water_coins.jpg')'''

# **Teoría de Algoritmos de Cuencas Hidrográficas**
# Cualquier imagen en escala de grises puede ser vista como una superficie topográfica donde la alta intensidad denota
# picos y colinas mientras que la baja intensidad denota valles.

# Este algoritmo utiliza esa analogía y comienza a llenar esos puntos bajos (valles) con una etiqueta de color diferente
# (aka nuestra agua).
#
# A medida que el agua sube, dependiendo de los picos (gradientes) cercanos, el agua de diferentes valles, obviamente
# con diferentes colores comenzará a fusionarse. Para evitar eso, construyes barreras en los lugares donde el agua se
# fusiona. Continúa el trabajo de llenar agua y construir barreras hasta que todos los picos estén bajo el agua.
#
# Las barreras que has creado te dan el resultado de la segmentación. Esta es la "filosofía" detrás de la cuenca.
# Puedes visitar la página web [CMM webpage](http://cmm.ensmp.fr/~beucher/wtshed.html) sobre la cuenca hidrográfica
# para entenderla con la ayuda de algunas animaciones.
# Su enfoque, sin embargo, le da un resultado oversegmented debido al ruido o cualquier otra irregularidad en la imagen.

# MI EXPLICACION
# En resumen la transformación de cuencas hidrográficas se basa en la idea de que, sobre una imagen en escala de grises
# y tomando los cambios de tonalidad de dicha escala, podemos, simulando el negro como el mínimo y el blanco como el
# máximo, indundar desde sus mínimos la imagen con agua evitando la fusión del agua en zonas distinas con los tonos
# blancos, creando así una división o segmentación en la imagen. Debido a problemas que conlleva el ruido y cambios de
# tono en imágenes reales, se establecen marcadores antes de la "inundación" para que se realiza la segmentación de
# forma correcta a raíz de lo deseado de la imagen


#  OpenCV implementó un algoritmo de cuenca basado en marcadores donde se especifica cuáles son todos los
# puntos que se van a fusionar y cuáles no. Da diferentes etiquetas para los objetos que conocemos.
# Etiquetamos la región del primer plano u objeto con un color (o intensidad),
# etiquetamos del fondo o no objeto con otro color y finalmente la región de
# que desconocemos, la etiquetamos con 0.
# Ese es nuestro marcador. A continuación, aplicar el algoritmo el marcador se actualizará con las etiquetas que le
# dimos, y los límites de los objetos tendrán un valor de -1.

#


# Cargar imagen
img = cv2.imread('images/water_coins.jpg')
imshow("Original image", img)

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbral usando OTSU (visto en 09)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

imshow("Thresholded", thresh)


## ## **Eliminar las máscaras de retoque**


# eliminación de ruido
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

# área de fondo, estamos creando los marcadores sobre las monedas dilatantdo  (visto en 10, es decir agregando
# píxeles a los límites de los objetos, el fondo en este caso en una imagen) 3 veces la imagen
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Encontrar el área de primer plano, mediante la función cv2.distanceTransform y la binarización de la imagen resultante
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Encontrar región desconocida restando el fondo al primer plano
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

imshow("SureFG", sure_fg)
imshow("SureBG", sure_bg)
imshow("unknown", unknown)


# Etiquetado de marcadores
# connectedComponents determina la conectividad de regiones tipo blob en una imagen binaria.
ret, markers = cv2.connectedComponents(sure_fg)

# Añadir uno a todas las etiquetas para que el fondo no sea 0, sino 1
markers = markers+1

# Ahora, marca la región de unknown con cero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

imshow("img", img)



