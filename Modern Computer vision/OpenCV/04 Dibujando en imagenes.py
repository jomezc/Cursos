#!/usr/bin/env python
# coding: utf-8

# ## **Dibujar imágenes y formas usando OpenCV**
# Primero, importemos OpenCV y numpy y definamos nuestra función imshow


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


# Empecemos por hacer un lienzo cuadrado en blanco

# Cree una imagen negra usando numpy para crear una matriz de negro
# array tridimensional, tamaño 512x512 de 3 canales de tipo entero de 0 a 255, todo a 0 significa negro
image = np.zeros((512,512,3), np.uint8)

# ¿Podemos hacer esto en blanco y negro? escala de grises
image_gray = np.zeros((512,512), np.uint8)

# El negro sería lo mismo que una imagen en escala de grises o en color (lo mismo para el blanco)
# el 1º ocupa 3 veces más memoria por las 3 dimensiones
imshow("Black Canvas - RGB Color", image)
imshow("Black Canvas - Grayscale", image_gray)


# ### **Dibujemos una línea sobre nuestro cuadrado negro**
# ```cv2.line(imagen, coordenadas iniciales, coordenadas finales, color, espesor)```

# Tenga en cuenta que esta es una operación en el lugar, lo que significa que cambia la imagen de entrada
# A diferencia de muchas otras funciones de OpenCV que devuelven una nueva imagen sin afectar la entrada
# Recuerda que nuestra imagen era el lienzo negro
cv2.line(image, (0,0), (511,511), (255,127,0), 5)

imshow("Black Canvas With Diagonal Line", image)


# ### **Drawing Rectangles**
# ```cv2.rectangle(imagen, vértice inicial (sup izq), vértice opuesto (inf der), color, espesor)```
# Vuelva a crear nuestro lienzo negro porque ahora tiene una línea
image = np.zeros((512,512,3), np.uint8)

# Espesor - si es positivo. Espesor negativo significa que está lleno
cv2.rectangle(image, (100,100), (300,250), (127,50,127), 10)
imshow("Black Canvas With Pink Rectangle", image)


# ### **Dibujemos algunos círculos**
# ```cv2.circle(imagen, centro, radio, color, relleno)```
# de nuevo la imagen negra ...
image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,150,50), -1) 
imshow("Black Canvas With Green Circle", image)


# ### **Polygons**
# ```cv2.polylines(imagen, puntos, ¿Cerrado?, color, grosor)```
# si Cerrado = Verdadero, unimos el primer y último punto.
# De nuevo reseteamos la imagen negra ...
image = np.zeros((512, 512, 3), np.uint8)

# Definamos cuatro puntos mediante un array, una matriz con subpuntos dentro
pts = np.array([[10,50], [400,50], [90,200], [50,500]], np.int32)
pts.shape   # (4,2)
# **Nota** cv2.polylines requiere que nuestros datos tengan la siguiente forma:
# Ahora remodelemos nuestros puntos en la forma requerida por las polilíneas ( en realidad solo cambia el formato)

print(pts)
'''[[ 10  50]
 [400  50]
 [ 90 200]
 [ 50 500]]'''
# estás agregando un 1, en una dimensión adicional en medio por como funciona polylines internamente, como decodifica
# los puntos
pts = pts.reshape((-1, 1, 2))
pts.shape  # (4, 1 ,2)
'''[[[ 10  50]]
 [[400  50]]
 [[ 90 200]]
 [[ 50 500]]]'''
print(pts)

cv2.polylines(image, [pts], True, (0,0,255), 3)
imshow("Black Canvas with Red Polygon", image)

# ### **Y ahora para agregar texto con cv2.putText**
# cv2.putText(imagen, 'Texto para mostrar', punto de inicio inferior izquierdo, Fuente, Tamaño de fuente, Color, Grosor)

# **Fuentes disponibles**
# - FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
# - FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX 
# - FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
# - FONT_HERSHEY_SCRIPT_SIMPLEX
# - FONT_HERSHEY_SCRIPT_COMPLEX

image = np.zeros((1000,1000,3), np.uint8)
ourString =  'Hello World!'
cv2.putText(image, ourString, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 4)
imshow("Messing with some text", image)
