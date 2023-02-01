###################################################
# 13 Detección de líneas, círculos y manchas ######
###################################################

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

# ## **Detección de línea: uso de líneas Hough**
#
# La transformada de Hough toma un mapa de bordes binarios como entrada e intenta ubicar los bordes colocados como
# líneas rectas. La idea de la transformada de Hough es que cada punto de borde en el mapa de bordes se transforma en
# todas las líneas posibles que podrían pasar por ese punto.
#
# Es decir, es una transformada utilizada para detectar líneas rectas.
# Para aplicar la Transformación, primero es deseable un preprocesamiento de detección de bordes.

# En general, una línea se puede detectar al encontrar el número de intersecciones entre curvas. Cuantas más curvas se
# intersecan, significa que la línea representada por esa intersección tiene más puntos. En general, podemos definir un
# umbral del número mínimo de intersecciones necesarias para detectar una línea.
# Esto es lo que hace la transformada de línea de Hough. Realiza un seguimiento de la intersección entre las curvas de
# cada punto de la imagen. Si el número de intersecciones está por encima de algún umbral, lo declara como una línea
# con los parámetros( θ ,rθ)del punto de intersección.

# `líneas = cv2.HoughLines(imagen binarizada/con umbral, 𝜌 precisión, 𝜃 precisión, umbral)`
# bordes : Salida del detector de bordes.
# líneas : Un vector para almacenar las coordenadas del inicio y final de la línea.
# rho : El parámetro de resolución \rho en píxeles.
# theta : La resolución del parámetro \theta en radianes.
# umbral : El número mínimo de puntos de intersección para detectar una línea.
# - El umbral aquí es el voto mínimo para que se considere una línea
#

image = cv2.imread('images/soduku.jpg')
imshow('Original', image)

# Escala de grises y Canny Edges extraídos
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)  # binarizamos la imagen

# Ejecute HoughLines con una precisión de rho de 1 píxel
# precisión theta de np.pi / 180 que es 1 grado (es un valor predefinido)
# Nuestro umbral de línea está establecido en 240 (número de puntos en línea que vamos a buscar)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)
# nos da básicamente un array de líneas y ángulos, así que tenemos que extraer todas las líneas por separado de lines.
print(lines)
'''[[[   4.           2.8797932]]
 [[   2.           2.8797932]]...'''
# // Dibujar las lineas
# Iteramos a través de cada línea y la convertimos al formato
# requerido por cv2.lines (es decir, requiere puntos finales)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

imshow('Hough Lines', image)


# ## **Líneas de Hough probabilísticas** # el anterior funciona muy bien, pero puede ser costoso
# Una transformada de Hough se considera probabilística si utiliza un muestreo aleatorio de los puntos de borde.
# Estos algoritmos se pueden dividir en función de cómo asignan el espacio de la imagen al espacio de los parámetros.
# cv2.HoughLinesP(imagen binarizada, precisión 𝜌, precisión 𝜃, umbral, longitud mínima de línea, espacio máximo entre
#                 líneas)


# Escala de grises y Canny Edges extraídos
image = cv2.imread('images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

# Nuevamente usamos las mismas precisiones rho y theta
# Sin embargo, especificamos un voto mínimo (pts a lo largo de la línea) de 100
# y longitud de línea mínima de 3 píxeles y espacio máximo entre líneas de 25 píxeles
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 3, 25)
print(lines.shape)  # (63, 1, 4)

for x in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

imshow('Probabilistic Hough Lines', image)  # este no funciona tan bien como el otro


# ## **Detección de círculos - Hough Cirlces**
# **cv2.HoughCircles**(imagen, método, dp, MinDist, param1, param2, minRadius, MaxRadius)
# - Método: actualmente solo está disponible cv2.HOUGH_GRADIENT
# - dp - Relación inversa de la resolución del acumulador a la resolución de la imagen. Por ejemplo, si dp=1, el
#         acumulador tiene la misma resolución que la imagen de entrada. Si dp=2, el acumulador tiene la mitad de ancho
#         y alto. Para HOUGH_GRADIENT_ALT el valor recomendado es dp=1.5, a menos que se necesiten detectar algunos
#         círculos muy pequeños.
# - MinDist - la distancia mínima entre el centro de los círculos detectados
# - param1 - Valor de gradiente utilizado en la detección de bordes
# - param2 - Umbral del acumulador para el método HOUGH_GRADIENT (menor permite detectar más círculos
#           (falsos positivos))
# - minRadius - limita el círculo más pequeño a este tamaño (a través del radio)
# - MaxRadius - establece de manera similar el límite para los círculos más grandes


image = cv2.imread('images/Circles_Packed_In_Square_11.jpeg')
imshow('Circles', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# medianBlur toma una mediana de todos los píxeles debado del kernel_size (5) y reemplaza el elemento  con ese valor
# central, efectivo para reducir el ruido, desenfoca la imagen
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)

cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # dibujar el círculo exterior
    cv2.circle(image,(i[0], i[1]), i[2], (0, 0, 255), 5)
    
    # dibujar el centro del circulo
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 8)

imshow('Detected circles', image)


# ## **Detección de (blob)manchas**
#
'''Un blob se define como un área interesante o básicamente un grupo o segmento de píxeles que son interesantes
en una imagen, un significado interesante que probablemente tienen algún tipo de uniformidad de consistencia a través
diferentes manchas o blobs para que realmente pueda detectar diferentes blobs en las imágenes.

Así que es básicamente una forma de  revisar una imagen y encontrar áreas interesantes, es decir *Podemos detectar 
manchas o círculos en una imagen*'''

# La función **cv2.drawKeypoints** toma los siguientes argumentos:
# **cv2.drawKeypoints**(imagen de entrada, puntos clave, matriz_de_salida_en_blanco, color, banderas)
# banderas:
# - cv2.DRAW_MATCHES_FLAGS_DEFAULT
# - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# - cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# - cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

# Leer imagen
image = cv2.imread("images/Sunflowers.jpg")
imshow("Original", image)

# Configure el detector con los parámetros predeterminados.
detector = cv2.SimpleBlobDetector_create()
 
# Detectar puntos clave.
keypoints = detector.detect(image)
 
# Dibujar manchas detectadas como círculos rojos.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS asegura el tamaño del círculo corresponde al tamaño de blob
blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
 
# Mostrar puntos clave
imshow("Blobs", blobs)



