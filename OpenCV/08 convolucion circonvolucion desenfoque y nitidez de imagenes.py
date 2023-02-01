############################################################
# 08 Convoluciones, desenfoque y nitidez de imágenes** #####
############################################################
#  Operaciones de convolución: una convolución es una operación matemática realizada en dos funciones que producen
#  una función escalonada que generalmente es una versión modificada de una de las funciones originales. No es más que
#  una multiplicación de funciones (primer elemento X resto y la suma de los resultados)

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

# ### **Desenfoque usando circunvoluciones ( la inversa de la convolución) **

import cv2
import numpy as np

image = cv2.imread('images/flowers.jpeg')
imshow('Original Image', image)

# estamos creando un "kernel" o matriz de árboles y la dividimos entre 9 para que podamos escalarla nuevamente por un
# factor del 11%, haciéndolo de esta forma para mantener consistente el brillo
# Creando nuestro kernel 3 x 3
kernel_3x3 = np.ones((3, 3), np.float32) / 9
print(kernel_3x3)
'''[[0.11111111 0.11111111 0.11111111]
 [0.11111111 0.11111111 0.11111111]
 [0.11111111 0.11111111 0.11111111]]'''

# Usamos el cv2.fitler2D para combinar el kernel con una imagen, realiza esa convolución o multiplicación de funciones
blurred = cv2.filter2D(image, -1, kernel_3x3)
imshow('3x3 Kernel Blurring', blurred)

# Creando nuestro kernel 7 x 7
kernel_7x7 = np.ones((7, 7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
imshow('7x7 Kernel Blurring', blurred2)


# ### **Otros métodos de desenfoque de uso común en OpenCV**
# - Desenfoque regular
# - Desenfoque gaussiano
# - Desenfoque medio

import cv2
import numpy as np

image = cv2.imread('images/flowers.jpeg')

'''
método cv2.blur()
El método se utiliza para desenfocar una imagen utilizando el filtro de cuadro normalizado. La función suaviza 
una imagen.
Sintaxis: cv2.blur(src, ksize[, dst[, ancla[, borderType]]])
Parámetros:
- src: Es la imagen de la que se desea difuminar.
- ksize: una tupla que representa el tamaño del kernel de desenfoque, es decir  son las dimensiones del núcleo 
        de la caja. En este ejemplo sería un kernel de caja de 13 por 13 que estaría involucrado con la imagen para 
        dar como resultado una imagen borrosa. si el tamaño del kernel es más pequeño que el desenfoque, se reduce, 
        si el tamaño del kernel es más grande se obtiene un desenfoque más sustancial.
- dst: Es la imagen de salida del mismo tamaño y tipo que src.
- ancla: es una variable de tipo entero que representa el punto de anclaje y su valor predeterminado es (-1, -1)
         ,lo que significa que el ancla está en el centro del kernel.
- borderType: representa qué tipo de borde se agregará. Está definido por indicadores como cv2.BORDER_CONSTANT 
             , cv2.BORDER_REFLECT , etc.
- Valor devuelto: Devuelve una imagen.
        '''
# blur(Promedio) realizado convolucionando la imagen con un filtro de cuadro normalizado.
# Esto toma los píxeles del paŕametro y reemplaza los píxeles de la imagen con el promedio
# El tamaño de la caja debe ser impar y positivo
blur = cv2.blur(image, (5,5))
imshow('Averaging', blur)

# En lugar de filtro de caja, kernel gaussiano
Gaussian = cv2.GaussianBlur(image, (5,5), 0)
imshow('Gaussian Blurring', Gaussian)

# Toma la mediana de todos los píxeles debajo del área del kernel y central
# elemento se reemplaza con este valor medio
median = cv2.medianBlur(image, 5)
imshow('Median Blurring', median)


# ### **Filtro bilateral**
# función de ELIMININACION de ruido
# #### ```dst = cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])```
# - **src** Imagen de origen de 8 bits o punto flotante, 1 canal o 3 canales.
# - **dst** Imagen de destino del mismo tamaño y tipo que src .
# - **d** Diámetro de cada vecindario de píxeles que se utiliza durante el filtrado. Si no es positivo, se calcula a
#         partir de sigmaSpace.
# - **sigmaColor** Filtra sigma en el espacio de color. Un valor mayor del parámetro significa que los colores más
#                  lejanos dentro de la vecindad de píxeles (consulte sigmaSpace) se mezclarán, lo que dará como
#                  resultado áreas más grandes de color semi-igual.
# - **sigmaSpace** Filtra sigma en el espacio de coordenadas. Un valor mayor del parámetro significa que los píxeles más
#                  lejanos se influirán entre sí siempre que sus colores estén lo suficientemente cerca (ver sigmaColor)
#                  .Cuando d>0, especifica el tamaño de la vecindad independientemente de sigmaSpace. De lo contrario,
#                  d es proporcional a sigmaSpace.
# - Modo de borde **borderType** utilizado para extrapolar píxeles fuera de la imagen


# Bilateral es muy efectivo en la eliminación de ruido mientras mantiene los bordes nítidos
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
imshow('Bilateral Blurring', bilateral)


# ## **Eliminación de ruido de imagen: eliminación de ruido de medios no locales**
# MÁS RÁPIDO
# **Hay 4 variaciones de medios no locales de eliminación de ruido:**
#
# - cv2.fastNlMeansDenoising() - funciona con una sola imagen en escala de grises
# - cv2.fastNlMeansDenoisingColored() - funciona con una imagen en color.
# - cv2.fastNlMeansDenoisingMulti() - funciona con secuencias de imágenes capturadas en un corto período de tiempo
#                                      (imágenes en escala de grises)
# - cv2.fastNlMeansDenoisingColoredMulti() - igual que arriba, pero para imágenes en color.
#
# fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7,
#                               int searchWindowSize=21 )¶```
#
# #### Parámetros para fastNlMeansDenoisingColored:
#
# - **src** – Entrada de imagen de 3 canales de 8 bits.
# - **dst** – Imagen de salida con el mismo tamaño y tipo que src.
# - **h** – Parámetro que regula la intensidad del filtro para el componente de luminancia. Un valor h más grande
#           elimina perfectamente el ruido pero también elimina los detalles de la imagen, un valor h más pequeño
#           conserva los detalles pero también conserva algo de ruido
# - **hColor** – Lo mismo que h pero para componentes de color. Para la mayoría de las imágenes, el valor igual a 10
#                será suficiente para eliminar el ruido de color y no distorsionar los colores.
# templateWindowSize: tamaño en píxeles del parche de plantilla que se utiliza para calcular los pesos. Debería ser
#                     extraño. Valor recomendado 7 píxeles
# - **searchWindowSize**: tamaño en píxeles de la ventana que se utiliza para calcular el promedio ponderado de un
#                         píxel determinado. Debería ser extraño. Afecta el rendimiento de forma lineal: mayor tamaño
#                         de ventana de búsqueda, mayor tiempo de eliminación de ruido. Valor recomendado 21 píxeles


image = cv2.imread('images/hilton.jpeg')
imshow('Original', image)

dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
imshow('fastNlMeansDenoisingColored', dst)


# ### **Afilado de imágenes**
# técnica similar al desenfoque, significa que éstamos mejorando los bordes, se ve un efecto HDR, todo se parece un poco
#  mas visible
# Cargando nuestra imagen
image = cv2.imread('images/hilton.jpeg')
imshow('Original', image)

# Crea nuestro núcleo de modelado, recuerda que debe sumar uno
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

# aplicar el núcleo de nitidez a la imagen
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
imshow('Sharpened Image', sharpened)



