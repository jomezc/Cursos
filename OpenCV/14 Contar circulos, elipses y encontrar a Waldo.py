#!/usr/bin/env python
# coding: utf-8

########################################################
# 14 Contar círculos, elipses y encontrar a Waldo*######
########################################################

# ####**En esta lección aprenderemos:**
# 1. Mini proyecto sobre el conteo de manchas circulares
# 2. Mini proyecto sobre el uso de la coincidencia de plantillas para encontrar a Waldo


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 12):
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

# ## **Contar manchas circulares**
# la siguiente imagen ayuda mucho a enterder los parametros!
# ![](https://i.stack.imgur.com/zYL2C.jpg)
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/simpleblob.png)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen
image = cv2.imread("images/blobs.jpg", 0)
imshow('Original Image',image)

# Inicialice el detector usando los parámetros predeterminados
detector = cv2.SimpleBlobDetector_create()
 
# Detectar manchas
keypoints = detector.detect(image)
 
# Dibujar manchas en nuestra imagen como círculos rojos
blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Mostrar imagen con puntos clave de blob
imshow("Blobs using default parameters", blobs)

# Establecer nuestros parámetros de filtrado
# Inicializa la configuración de parámetros usando cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Establecer parámetros de filtrado de área (tamaño del círculo)
params.filterByArea = True
params.minArea = 100

# Establecer parámetros de filtrado de circularidad (más o menos circular, es decir un triangulo tiene muy
# poca, un cuadrado más, polígono tiene mucha más... hasta llegar al círculo, 0.9 muy circular
params.filterByCircularity = True 
params.minCircularity = 0.9

# Establecer parámetros de filtrado de convexidad, si está completo el círculo, imaginándolo como una tarta las
# porciones que tiene
params.filterByConvexity = False
params.minConvexity = 0.2
    
# Establecer parámetros de filtrado de inercia, si es un circulo perfecto o más una elipse, es decir si es redondo
# o más 'chafado'
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Crear un detector con los parámetros
detector = cv2.SimpleBlobDetector_create(params)
    
# Detectar manchas
keypoints = detector.detect(image)

# Dibujar manchas en nuestra imagen como círculos rojos
blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Mostrar manchas
imshow("Filtering Circular Blobs Only", blobs)


# # **Buscando a Waldo usando la coincidencia de plantillas**
#
# #### **Notas sobre la coincidencia de plantillas**
#
# Hay una variedad de métodos para realizar la comparación de plantillas, pero en este caso estamos usando el
# coeficiente de correlación que se especifica mediante el indicador **cv2.TM_CCOEFF.**
#
# Entonces, ¿qué está haciendo exactamente la función cv2.matchTemplate?
# Esencialmente, esta función toma una "ventana deslizante" de nuestra imagen de consulta de waldo y la desliza a
# través de nuestra imagen de rompecabezas de izquierda a derecha y de arriba a abajo, un píxel a la vez. Luego, para
# cada una de estas ubicaciones, calculamos el coeficiente de correlación para determinar cuán "buena" o "mala" es la
# coincidencia.
#
# Las regiones con una correlación suficientemente alta pueden considerarse "coincidencias" para nuestra plantilla de
# waldo.A partir de ahí, todo lo que necesitamos es una llamada a cv2.minMaxLoc en la Línea 22 para encontrar dónde
#  están nuestras "buenas" coincidencias. ¡Eso es realmente todo lo que hay que hacer para hacer coincidir plantillas!

# http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html


template = cv2.imread('./images/waldo.jpg')
imshow('Template', template)


# Cargue la imagen de entrada y conviértala a escala de grises
image = cv2.imread('./images/WaldoBeach.jpg')
imshow('Where is Waldo?', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cargar imagen de plantilla
template = cv2.imread('./images/waldo.jpg',0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Crear cuadro delimitador
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

imshow('Where is Waldo?', image)





