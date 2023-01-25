#!/usr/bin/env python
# coding: utf-8
######################################
# 02 Imágenes en escalada grises ####
######################################
'''# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
**Imágenes en escala de grises**
En esta lección aprenderemos a:
1. Convierte una imagen en color a escala de grises
2. Ver el cambio de dimensiones entre las imágenes en escala de grises y en color

# ### **Descargando imágenes**
Si usa Google Colab, tendremos que **cargar nuestra imagen**.
Colab es un entorno de Jupyther Notebook que se ejecuta en la **nube** usando los servidores de Google. Como tal,
cualquier archivo que deseemos utilizar debe cargarse en sus servidores.'''




import cv2
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]  # desempaquetamos la relación de aspecto mediante el ancho y el alto
    aspect_ratio = w/h  # calculamos la relación de aspecto
    '''# para asegurar que se cumpla la relación de aspecto, multiplicamos el tamaño por la relación calculada como 1º 
    parametro y le pasamos el tamaño a mostrar segundo para poder cambiar el tamaño de la imagen de salida'''
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Download and unzip our images (colab)
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


# Load our input image
image = cv2.imread('./images/castara.jpeg')
imshow("Castara, Tobago", image)


# In[4]:


image.shape[:2]  # (1200, 1920) (height, width)
def imshow(title = "", image = None, size = 10):
 
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

imshow("Castara, Tobago", image)


# Usamos cvtColor, para convertir a escala de grises
# Toma 2 argumentos, el primero es la imagen de entrada
# El segundo es el código de conversión del espacio de color
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # estamos convirtiendo la imagen a escala de grises
imshow("Converted to Grayscale", gray_image)


# ### **Dimensiones de la imagen en escala de grises**
# Recuerde que las imágenes en color RGB tienen 3 dimensiones, una para cada color primario. La escala de grises solo
# tiene 1, que es la intensidad del gris. 0 es negro y 255 blanco lo demás la escala de gris
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/gray.png)

print(image.shape)  # (1280, 960, 3)
print(gray_image.shape)  # (1280, 960)

