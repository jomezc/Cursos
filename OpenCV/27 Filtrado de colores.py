###############################
# 27 Filtrado de colores ######
###############################
# 1. Cómo utilizar el espacio de color HSV para filtrar por color
#
# #### **Recordar el Espacio de Color HSV** ( visto en 02)
# ![](https://answers.opencv.org/upfiles/15186766673210035.png)
#
# - Tono: 0 - 179
# - Saturación 0 - 255
# - Valor (Intensidad): 0 - 255
# es mucho más facil extraer un color en HSV que en RGB


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


# Vamos a intentar quitar el camion y la tierra de la imagen dejando solo el cielo

image = cv2.imread('images/truck.jpg')

# Entonces, para hacer eso necesitamos definir un rango superior e inferior.
# definir el rango de color AZUL en HSV, en la imagen de arriba se ve que el azul va del tono 90 al 135

lower = np.array([90,0,0])  # tono , saturación , valor
upper = np.array([135,255,255])

# Convertir la imagen de RBG/BGR a HSV para poder filtrar fácilmente
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Usar inRange para capturar solo los valores entre inferior y superior, es decir crear una máscara, un umbral binario
# en la imagen, el blanco sería un SI, entra en la máscara y el negro un NO
mask = cv2.inRange(hsv_img, lower, upper)

# Realizar Bitwise AND en la máscara y nuestro fotograma original, obteniendo con esa simple operación el filtro
res = cv2.bitwise_and(image, image, mask=mask)

imshow('Original', image)  
imshow('mask', mask)
imshow('Filtered Color Only', res)


# Otra imagen
# #### **Filtrar el rojo**
image = cv2.imread("./images/Hillary.jpg")
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# es más complicado porque el rojo va desde el 0 al 10 y del 170 al 180 ( por estar dividido por el cero) por lo que
# para poder filtrarlo creamos 2 máscaras en vez de una

# máscara inferior (0-10)
lower_red = np.array([0,0,0])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# máscara superior (170-180)
lower_red = np.array([170,0,0])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# unir máscaras, sumándolas
mask = mask0+mask1

# Realizar Bitwise AND en la máscara y nuestro marco original
res = cv2.bitwise_and(image, image, mask=mask)

imshow('Original', image)  
imshow('mask', mask)
imshow('Filtered Color Only', res)
