# *************************************************************************************
# ***** 77 Anadir y eliminar ruido y corregir contraste con ecualizacion de histograma
# *************************************************************************************
# 1. Cómo añadir ruido blanco o efectos de grano de película a las imágenes
# 2. Cómo implementar la ecualización de histograma

### **¿Qué es el ruido?**
#
# ![](https://2.bp.blogspot.com/-b-hwrNlSs4Y/V6IKh7NamaI/AAAAAAAAOB4/rJ7oPYVKZgg2Py9eA7pR62Lbn1yNJjnvwCLcB/s1600/ISO-Noise.jpg)
#
# Los sensores de las cámaras digitales pueden hacer fotos en entornos con poca luz aumentando la sensibilidad del
# sensor de la cámara (CCD). Sin embargo, este aumento de la sensibilidad (aumento ISO) tiene un precio. El precio es
# el ruido. El ruido surge porque la mayor sensibilidad del sensor lo hace susceptible al ruido aleatorio. Esto se
# debe a que en las escenas con poca luz no hay mucha variación entre la escena y el ruido aleatorio de los fotones.
#
# https://blog.michaeldanielho.com/2016/08/understanding-cameras-exposure-setting.html

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import random
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


# ## **Añadir grano de película o ruido a las imágenes**

def addWhiteNoise(image):
    # Establece el rango para una probabilidad aleatoria
    # Una probabilidad grande significará más ruido
    prob = random.uniform(0.05, 0.1)

    # Generar una matriz aleatoria con la forma de nuestra imagen de entrada
    rnd = np.random.rand(image.shape[0], image.shape[1])

    # Si los valores aleatorios de nuestra matriz rnd son menores que nuestra probabilidad aleatoria
    # Cambiamos aleatoriamente ese píxel de nuestra imagen de entrada a un valor dentro del rango especificado
    image[rnd < prob] = np.random.randint(50,230)
    return image


# Cargar nuestra imagen
image = cv2.imread('images/londonxmas.jpeg')
imshow("Input Image", image)

# Aplicar nuestra función de ruido blanco a nuestra imagen de entrada
noise_1 = addWhiteNoise(image)
imshow("Noise Added", noise_1)

# cv2.fastNlMeansDenoisingColored(input, None, h, hForColorComponents, templateWindowSize, searchWindowSize)
# - Entrada ( input )
# - matriz de salida ( se pone a None)
# - h, parámetro que decide la intensidad del filtro. Un valor h más alto elimina mejor el ruido, pero también los
#   detalles de la imagen. - la fuerza del filtro 'h' (5-12 es un buen rango)
# - Lo siguiente es hForColorComponents, el mismo valor que h de nuevo normalmente, pero solo para imágenes a color
# - tamaño de la ventana de plantilla templateWindowSize (sólo números impares) rec. 7
# - busqueda de tamaño de ventana searchWindowSize (sólo números impares) rec. 21

# ojo reducir el ruido elimina detalles
dst = cv2.fastNlMeansDenoisingColored(noise_1, None, 11, 6, 7, 21)


imshow("Noise Removed", dst)


# **hay 4 variaciones **
# - cv2.fastNlMeansDenoising() - trabaja con una sola imagen en escala de grises
# - cv2.fastNlMeansDenoisingColored() - trabaja con una imagen en color.
# - cv2.fastNlMeansDenoisingMulti() - trabaja con secuencia de imágenes capturadas en corto periodo de tiempo
#                                   (imágenes en escala de grises)
# - cv2.fastNlMeansDenoisingColoredMulti() - igual que el anterior, pero para imágenes en color.


# ### **Usando la Cualificación del Histograma**
# ![](https://docs.opencv.org/master/histogram_equalization.png)
#
# Esto 'ajusta' el rango dinámico de una imagen ( capacidad de captar en una imagen la mayor cantidad posible de tonos
# de exposición, es decir, la cantidad de señales que es capaz de captar o reproducir, en términos de luminosidad).
# Esto provoca que se extienda más uniformemente según la distribución de intensidad, y mejorando así el contraste.
#
# El contraste significa diferencia. Una diferencia que en fotografía suele hacer referencia a la luminosidad y
# cromaticidad de una imagen. Es decir, nos ayuda a comprender cómo la luz y el color influyen en una fotografía.
# Luz y color.

# #### **Primero, echemos un vistazo al histograma de nuestra imagen de entrada**

# Cargar nuestra imagen
img = cv2.imread('images/soaps.jpeg')
imshow("Original", img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear nuestra distribución del histograma
hist, bins = np.histogram(gray_image.flatten(),256,[0,256])

# Obtener la suma acumulada
cdf = hist.cumsum()

# Obtener una distribución acumulativa normalizada
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Trazar nuestro CDF superpuesto a nuestro histograma
plt.plot(cdf_normalized, color = 'b')
plt.hist(gray_image.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
imshow("gray_image", gray_image)


# #### **Ahora, apliquemos la ecualización del histograma**

img = cv2.imread('images/soaps.jpeg')

# Convertir a escala de grises
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear nuestra distribución del histograma, es lo que transforma la imagen
gray_image = cv2.equalizeHist(gray_image)
imshow("equalizeHist", gray_image)

# Esta parte sirve para crear el histograma no para la imagrn
# Obtener una distribución acumulativa normalizada
hist, bins = np.histogram(gray_image.flatten(),256,[0,256])

# Obtener la suma acumulada
cdf = hist.cumsum()

# Obtener una distribución acumulativa normalizada
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Trazar nuestro CDF superpuesto a nuestro histograma
plt.plot(cdf_normalized, color = 'b')
plt.hist(gray_image.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


# ### **Excerise:**
# 1. Igualar todos los canales RGB (BGR) de esta imagen y luego fusionarlos para obtener una imagen en color igualada.

import cv2 
 
img = cv2.imread('images/soaps.jpeg')
 
imshow("Original", img)
 
# Igualar nuestro histograma
# El formato de color por defecto es BGR
 
red_channel = img[:, :, 2]
red = cv2.equalizeHist(red_channel)
 
green_channel = img[:, :, 1]
green = cv2.equalizeHist(green_channel)
 
blue_channel = img[:, :, 0]
blue = cv2.equalizeHist(blue_channel)
 
# crear una imagen vacía con la misma forma que la imagen de origen
red_img = np.zeros(img.shape)
red_img[:,:,2] = red
red_img = np.array(red_img, dtype=np.uint8)
imshow("Red", red_img)
 
green_img = np.zeros(img.shape)
green_img[:,:,1] = green
green_img = np.array(green_img, dtype=np.uint8)
imshow("Green", green_img)
 
blue_img = np.zeros(img.shape)
blue_img[:,:,0] = blue
blue_img = np.array(blue_img, dtype=np.uint8)
imshow("Blue", blue_img)
 
merged = cv2.merge([blue, green, red])
imshow("Merged", merged)



