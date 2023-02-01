#####################################################
# 10 Detección de dilatación, erosion y bordes apertura cierre ######
#####################################################

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

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-11-15%20at%205.19.08%20pm.png)

image = cv2.imread('images/opencv_inv.png', 0)
imshow('Original', image)

# Definamos el tamaño de nuestro kernel, es una matriz que usamos para la convolución 2D quw se realiza en estas
# funciones
kernel = np.ones((5, 5), np.uint8)

# Ahora erosionamos, quitando pixeles a los limites de los objetos
erosion = cv2.erode(image, kernel, iterations = 1)
imshow('Erosion', erosion)

# Dilatar aqui, es decir agregando píxeles a los límites de los objetos, el fondo en este caso en una imagen
dilation = cv2.dilate(image, kernel, iterations = 1)
imshow('Dilation', dilation)

# Apertura - Bueno para eliminar el ruido, La operación de apertura es una operación de erosión seguida de dilatación.
# se usa para eliminar el ruido interno presente dentro de una imagen.
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
imshow('Opening',opening)

# Cierre - Bueno para eliminar el ruido, aplica dilatación seguida de erosión.
# Al igual que el operador Apertura, también utiliza un elemento estructurante, pero se utiliza para eliminar pequeños
# agujeros en lugar de pertusiones.

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
imshow('Closing',closing)


# ## **Detección de bordes astutos**

# Los bordes de una imagen digital se pueden definir como transiciones entre dos regiones de
# niveles de gris significativamente distintos. Suministran una valiosa información sobre las
# fronteras de los objetos y puede ser utilizada para segmentar la imagen, reconocer objetos, etc.
# La mayoría de las técnicas para detectar bordes emplean operadores locales basados en distintas
# aproximaciones discretas de la primera y segunda derivada (el cambio a oscuro y el retorno a claro) de los niveles
# de grises de la imagen.

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-11-15%20at%205.24.15%20pm.png)

# La detección de bordes necesita un umbral para indicar qué diferencia/cambio debe contarse como borde

image = cv2.imread('images/londonxmas.jpeg',0)
'''
Detector de bordes Canny con OpenCV
La función Canny() en OpenCV se utiliza para detectar los bordes de una imagen
canny = cv2.Canny(imagen, umbral_minimo, umbral_maximo)
Donde:
- canny: es la imagen resultante. Aparecerán los bordes detectados tras el proceso.
- imagen: es la imagen original.
- umbral_minimo: es el umbral mínimo en la umbralización por histéresis
- umbral_maximo: es el umbral máximo en la umbralización por histéresis

hay mas parámetros: 
- opening_size: Tamaño de apertura del filtro Sobel. Es el tamaño del núcleo Sobel utilizado para encontrar gradientes
                de imagen. Por defecto es 3.
- L2Gradient: Parámetro booleano utilizado para mayor precisión en el cálculo de Edge Gradient.
             el umbral mínimo y el máximo dependerá de cada situación.
Docu 
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html'''

# Canny Edge Detection utiliza valores de gradiente como umbrales
# El primer gradiente de umbral
canny = cv2.Canny(image, 50, 120)
imshow('Canny 1', canny)

# Los umbrales de borde ancho esperan muchos bordes
canny = cv2.Canny(image, 10, 200)
imshow('Canny Wide', canny)

# Umbral estrecho, espere menos bordes
canny = cv2.Canny(image, 200, 240)
imshow('Canny Narrow', canny)

canny = cv2.Canny(image, 60, 110)
imshow('Canny 4', canny)

# Luego, debemos proporcionar dos valores: umbral1 y umbral2. Cualquier valor de gradiente mayor que el umbral2
# se considera una ventaja. Cualquier valor por debajo del umbral 1 se considera que no es un borde.
# Los valores entre el umbral 1 y el umbral 2 se clasifican como bordes o no bordes en función de cómo Las intensidades
# están “conectadas”. En este caso, cualquier valor de degradado por debajo de 60 se considera sin bordes
# mientras que cualquier valor por encima de 120 se considera borde.

# #### **Astucia automática** ( sacado de stackoverflow)
def autoCanny(image):
  # Encuentra umbrales óptimos basados en la mediana de la intensidad de píxeles de la imagen
  blurred_img = cv2.blur(image, ksize=(5,5))
  med_val = np.median(image) 
  lower = int(max(0, 0.66 * med_val))
  upper = int(min(255, 1.33 * med_val))
  edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
  return edges

auto_canny = autoCanny(image)
imshow("auto canny", auto_canny)
