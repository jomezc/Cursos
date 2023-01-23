#!/usr/bin/env python
# coding: utf-8

###############################################################################
# 12 Momentos, Clasificación, Aproximación y Correspondencia de Contorno ######
###############################################################################

# 1. Ordenar contornos por área
# 2. Ordenar de izquierda a derecha (Excelente para OCR)
# 3. Contornos aproximados
# 4. Casco convexo

# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 16):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ### **Encontrar contornos como de costumbre** ( visto en 11)

# Carga la imagen
image = cv2.imread('images/bunchofshapes.jpg')
imshow('Original Image', image)

# Escala de grises nuestra imagen
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Encuentra bordes Canny
edged = cv2.Canny(gray, 50, 200)
imshow('Canny Edges', edged)

# Encuentre contornos e imprima cuántos se encontraron
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

# Dibuja todos los contornos sobre una imagen en blanco
cv2.drawContours(image, contours, -1, (0,255,0), 3)
imshow('All Contours', image)

'''¿Y si quisiéramos ordenar por área de estos contornos?
¿Cómo obtenemos el área de cada uno de estos objetos? para hacer eso, en realidad vamos a usar la función 
cv2.ContourArea'''
# ## **Clasificación por área usando cv2.ContourArea y cv2.Moments**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/moments.png)


import cv2
import numpy as np

# Función que usaremos para mostrar el área del contorno

def get_contour_areas(contours):
    """devuelve las áreas de todos los contornos como una lista
    Entonces, estamos recorriendo los contornos que antes hemos sacado y estamos obteniendo el área de cada contorno
    y añadiendolo a una lista"""
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Carga nuestra imagen
image = cv2.imread('images/bunchofshapes.jpg')

# Imprimamos las áreas de los contornos antes de ordenar
print("Contor Areas before sorting...")  # [20587.5, 22901.5, 66579.5, 90222.0]
print(get_contour_areas(contours))

# Ordenar contornos grandes a pequeños por área
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

print("Contor Areas after sorting...")   # [90222.0, 66579.5, 22901.5, 20587.5]
print(get_contour_areas(sorted_contours))

# Iterar sobre nuestros contornos y dibujar uno a la vez
for (i, c) in enumerate(sorted_contours):
    M = cv2.moments(c)  # estamos sacando el punto central del contorno
    # Esa M contiene un diccionario con las claves m00 m10 y los valores que son los puntos o pixeles... son una forma
    # de calcular los puntos xy
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # vamos a poner en el punto central de cada área de cada contorno un texto
    cv2.putText(image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    # dibujamos los contornos
    cv2.drawContours(image, [c], -1, (255,0,0), 3)

# lo que hemos realizado es clasificar de forma numerica por el tamaño del área de las figuras de la imagen
# ( son 2cuadrados,1 circulo, 1 triangulo) que hemos calculado a raíz de los contornos, es decir, hemos dibujado
# el controno y un número que clasifica de más grande a pequeño las áreas de las figuras del ejemplo
imshow('Contours by area', image)


# #### **Definir algunas funciones que usaremos**


# Funciones que usaremos para ordenar por posición
def x_cord_contour(contours):
    """Devuelve la coorednada X para el centroide del controno ( una función Usando los momentos para sacar
    la coordenada x  )"""
    if cv2.contourArea(contours) > 10:  # rechaza los contornos más pequeños
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass
    
def label_contour_center(image, c):
    """Coloca un círculo rojo en los centros de los contornos. Usando los momentos para sacar las coordenadas x e y"""
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Dibuja el número de contorno en la imagen
    cv2.circle(image,(cx,cy), 10, (0,0,255), -1)
    return image


# #### **Usamos momentos para calcular el centro y luego usamos la coordenada X para ordenar de izquierda a derecha**
# Carga nuestra imagen
image = cv2.imread('images/bunchofshapes.jpg')
orginal_image = image.copy()

# Computar Centro de Masa o centroides y dibujarlos en nuestra imagen
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c)
 
# Mostrando los centros de contorno
imshow("Sorting Left to Right", image)

# Ordenar de izquierda a derecha usando nuestra función x_cord_contour, por dentro usa esa función creada para ordenar
contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)

# Etiquetado de contornos de izquierda a derecha
for (i,c)  in enumerate(contours_left_to_right):
    cv2.drawContours(orginal_image, [c], -1, (0,0,255), 3)  
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orginal_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    (x, y, w, h) = cv2.boundingRect(c)  

imshow('Sorting Left to Right', orginal_image)


# ***cv2.approxPolyDP(contorno, precisión de aproximación, cerrado)***
# Entonces, esta es una función que puede tomar un contorno y aproximarlo
# - **contorno** – es el contorno individual que deseamos aproximar
# - **Precisión de la aproximación**: un parámetro importante determina la precisión de la aproximación. Los valores
# pequeños dan aproximaciones precisas, los valores grandes dan una aproximación más genérica
# una buena good regla empírica od es menos del 5% del perímetro del contorno
# # - **Cerrado**: un valor booleano que indica si el contorno aproximado debe estar abierto o cerrado

import numpy as np
import cv2

# Cargar imagen y guardar una copia
image = cv2.imread('images/house.jpg')
orig_image = image.copy()
imshow('Original Image', orig_image)
 
# Escala de grises y binarizar
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Encuentra contornos
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

copy = image.copy()

# Iterar a través de cada contorno
# queremos que se dibujen rectángulos de los contornos encontrados en la imagen:
for c in contours:
    #  boundingRect()
    # está sacando los puntos para poder dibujar esos rectángulos sobre los contornos encontrados
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
    # y luego dibujamos los contornos
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

imshow('Drawing of Contours', image)
imshow('Bounding Rectangles', orig_image)

# ******
# Iterar a través de cada contorno y calcular el contorno aproximado
# una forma de limpiar sus contornos y aproximarlos, como en la imagen que es un dibujo a mano irregular y lo mejora
for c in contours:  # toma un contorno de una imagen
    # Calcule la precisión como un porcentaje del perímetro del contorno
    #  Toma una precisión y precisión como un porcentaje del parámetro de contorno.
    # Así que quitas el 3 por ciento aquí del parámetro cuántico
    accuracy = 0.03 * cv2.arcLength(c, True)
    # ahora calcula  calcular el contorno aproximado con ese porcentaje de precisión
    approx = cv2.approxPolyDP(c, accuracy, True)
    # lo dibujas
    cv2.drawContours(copy, [approx], 0, (0, 255, 0), 2)

imshow('Approx Poly DP', copy)


# ## **Casco convexo**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/convex.png)
#
# Convex Hull se verá similar a la aproximación de contorno, pero no lo es (Ambos pueden proporcionar los mismos
# resultados en algunos casos).
#
# La función cv2.convexHull() verifica una curva en busca de defectos de convexidad y la corrige. En términos generales,
# las curvas convexas son las curvas que siempre están abultadas, o al menos planas. Y si está abombado por dentro, se
# llama defectos de convexidad. Por ejemplo, compruebe la siguiente imagen de la mano. La línea roja muestra el casco
# convexo de la mano. Las marcas de flecha de dos lados muestran los defectos de convexidad, que son las desviaciones
# máximas locales del casco de los contornos.


import numpy as np
import cv2

image = cv2.imread('images/hand.jpg')
orginal_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imshow('Original Image', image)

# Umbral de la imagen
ret, thresh = cv2.threshold(gray, 176, 255, 0)

# Encuentra contornos
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, 0, (0, 255, 0), 2)
imshow('Contours of Hand', image)


# Ordene los contornos por área y luego elimine el contorno de marco más grande
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]  # [:n]desde el primero hasta longitud -1 ( ultimo)

# Iterar a través de los contornos y dibujar el casco convexo
for c in contours:  # recorremos los contornos.
    # La integridad convexa toma contornos de entradas y salidas.
    hull = cv2.convexHull(c)
    cv2.drawContours(orginal_image, [hull], 0, (0, 255, 0), 2)
    
imshow('Convex Hull', orginal_image)


# # **Contornos coincidentes**
# básicamente cogen una plantilla de contorno como una referencia del contorno que queremos encontrar en una imagen
# #### **cv2.matchShapes(plantilla de contorno, contorno, método, parámetro de método)**
#
# **Salida**: valor de coincidencia (los valores más bajos significan una coincidencia más cercana)

# - Plantilla de contorno: este es nuestro contorno de referencia que estamos tratando de encontrar en la nueva imagen
# - Contorno: el contorno individual con el que estamos comprobando
# - Método - Tipo de coincidencia de contorno (1, 2, 3)
# - Parámetro de método: déjelo solo como 0.0 (no utilizado completamente en python OpenCV)

import cv2
import numpy as np

# Cargue la plantilla de forma o la imagen de referencia
template = cv2.imread('images/4star.jpg',0)
imshow('Template', template)

# Cargue la imagen de destino con las formas que estamos tratando de hacer coincidir
target = cv2.imread('images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

# Primero establezca el umbral de ambas imágenes antes de usar cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# Encuentra contornos en la plantilla
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Necesitamos ordenar los contornos por área para poder eliminar los más grandes
# contorno que es el contorno de la imagen
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Extraemos el segundo contorno más grande que será nuestro contorno de plantilla
template_contour = contours[1]  # antes ordenábamos al revés y eliminábamos el último, ahora nos quedamos con el 2º

# Extraer contornos de la segunda imagen de destino
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # Iterar a través de cada contorno en la imagen de destino y
    # usar cv2.matchShapes para comparar formas de contorno
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    print(match)
    # Si el valor de la coincidencia es inferior a 0,15,
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = [] 
                
cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
imshow('Output', target)





