#!/usr/bin/env python
# coding: utf-8
#################################################
# 18 Transformaciones de perspectiva ######
#################################################
# a transformación de perspectiva es una forma en que podemos traducir una señal de imágenes de imagen para convertirla
# en una diferente.
# ####**En esta lección aprenderemos:**
# 1. 1. Usar getPerspectiveTransform de OpenCV
# 2. Usar findContours para obtener esquinas y automatizar la transformación de perspectiva


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
    
'''# Descargar y descomprimir nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

image = cv2.imread('images/scan.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarizamos la imagen
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, por ejemplo edged.copy(), ya que findContours altera la imagen
# sacmos los contornos externos ( RETR_EXTERNAL)
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibuja todos los contornos, ten en cuenta que esto sobrescribe la imagen de entrada (operación inplace)
# Usa '-1' como 3er parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 54


### **Approxiamamos nuestro contorno anterior a sólo 4 puntos usando approxPolyDP** ( visto en 12)
# Ordenar los contornos de mayor a menor por área (no necesario pero acelera el calculo) además como hay ruido en la
# imagen, la razón por la que estamos ordenando por área en primer lugar sabemos que el área mas grande es el que
# queremos, Es el control más grande, porque los otros van a ser muy pequeños.
#
# Son como píxeles son sólo grupos de píxeles


sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# bucle sobre los contornos
for cnt in sorted_contours:
    #  Aproximación de cada contorno calculando el perímetro y multipicandole el accuracy recomendando (OCV)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
 
    if len(approx) == 4:  # paramos cuando hayamos encontrado esos 4 puntos
        break

# Nuestras coordenadas x, y de las cuatro esquinas
print("Nuestros 4 puntos de esquina son:")
print(approx)
''''
Nuestros 4 puntos de esquina son:
[[[326  15]]

 [[ 83 617]]

 [[531 779]]

 [[697 211]]]'''


### **Usamos getPerspectiveTransform y warpPerspective para crear nuestra vista de arriba abajo**
#
# Nota: Hemos igualado manualmente el orden de los puntos

# El orden obtenido aquí es arriba a la izquierda, abajo a la izquierda, abajo a la derecha, arriba a la derecha

# acabamos de convertir el tipo de datos aquí porque lo necesitamos como float32
inputPts = np.float32(approx)

# estamos especificando que punto de salida queremos, donde queremos que esté de izq a derecha arriba a abajo
outputPts = np.float32([[0,0],
                       [0,800],
                       [500,800],
                       [500,0]])

# Obtenemos nuestra matriz de transformación, M
M = cv2.getPerspectiveTransform(inputPts,outputPts)

# Aplica la matriz de transformación M usando Warp Perspective
dst = cv2.warpPerspective(image, M, (500,800))

imshow("Perspective", dst)

# ### **Ejercicio**
# 1. Ordenar los puntos en ```approx`` ordenando desde arriba a la izquierda en el sentido de las agujas del reloj
# (es decir, arriba a la izquierda, arriba a la derecha, abajo a la izquierda, abajo a la derecha)
# 2. 2. Obtener la relación de aspecto inicial del contorno y ajustar el Warp final para que salga en esa relación de
# aspecto y orientación.

