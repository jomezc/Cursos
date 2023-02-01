# *****************************************
# ***** 46 Pintar imágenes para restauralas
# *****************************************
# **En esta lección tomaremos una foto vieja dañada, y la restauraremos usando la función inpaint()**

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


# Cargamos nuestra foto dañada
image = cv2.imread('images/abraham.jpg')
imshow('Original Damaged Photo', image)

# Cargamos la foto en la que hemos marcado las zonas dañadas, con un programa de utilidad de fotos normal
# dibujando las lineas
marked_damages = cv2.imread('images/mask.jpg', 0)
imshow('Marked Damages', marked_damages)

# Hagamos una máscara de nuestra imagen marcada cambiando todos los colores
# que no sean blancos, a negro, para usar esas marcas dibujadas en blanco
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
imshow('Threshold Binary', thresh1)


# Vamos a dilatar (hacer más gruesas) las marcas que hemos hecho
# ya que el umbral lo ha estrechado ligeramente
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations = 1)
imshow('Dilated Mask', mask)
cv2.imwrite("images/abraham_mask.png", mask)

restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

imshow('Restored', restored)




