########################################################################
# 15 Encontrar esquinas ######
########################################################################
# 1. Usar cornerHarris para encontrar esquinas
# 2. Use buenas funciones para rastrear

# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Define our imshow function
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



# ## **¿Qué es una esquina?**

# Una esquina es un punto cuya vecindad local se encuentra en dos direcciones de borde dominantes y diferentes. En otras
# palabras, una esquina puede interpretarse como la unión de dos bordes, donde un borde es un cambio repentino en el
# brillo de la imagen. Las esquinas son las características importantes de la imagen y, por lo general, se denominan
# puntos de interés que no varían con la traslación, la rotación y la iluminación.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/edge.png)

# ### **Harris Corner Detection** es un algoritmo desarrollado en 1988 para la detección de esquinas que funciona
# bastante bien incluso con estos parámetros predeterminados.

# **Papel** - http://www.bmva.org/bmvc/1988/avc-88-023.pdf

# **cv2.cornerHarris**(imagen de entrada, tamaño de bloque, tamañok, k)
# - Imagen de entrada - debe ser en escala de grises y tipo float32.
# - blockSize - el tamaño del vecindario considerado para la detección de esquinas
# - ksize - parámetro de apertura de la derivada de Sobel utilizada.
# - k - parámetro libre del detector de harris en la ecuación
# - **Salida**: matriz de ubicaciones de esquina (x, y)


# Cargar imagen y escala de grises
image = cv2.imread('images/chess.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# La función cornerHarris requiere que el tipo de datos de la matriz sea float32
gray = np.float32(gray)

harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

# Usamos la dilatación de los puntos de las esquinas para agrandarlos\
kernel = np.ones((7, 7), np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)

# Umbral para un valor óptimo, puede variar según la imagen.
image[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]

imshow('Harris Corners', image)

# **cv2.goodFeaturesToTrack**(imagen de entrada, maxCorners, qualityLevel, minDistance)

# - Imagen de entrada: imagen de un solo canal de 8 bits o punto flotante de 32 bits.
# - maxCorners – Número máximo de esquinas a devolver. Si hay más esquinas de las que se encuentran, se devuelve la más
# fuerte de ellas.
# - qualityLevel – Parámetro que caracteriza la calidad mínima aceptada de las esquinas de la imagen. El valor del
# parámetro se multiplica por la mejor medida de calidad de esquina (valor propio más pequeño). Las esquinas con la
# medida de calidad inferior al producto son rechazadas. Por ejemplo, si la mejor esquina tiene la medida de calidad =
# 1500 y el nivel de calidad = 0,01, todas las esquinas con la medida de calidad inferior a 15 se rechazan.
# - minDistance: distancia euclidiana mínima posible entre las esquinas devueltas.


img = cv2.imread('images/chess.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Especificamos las 50 mejores esquinas
corners = cv2.goodFeaturesToTrack(gray, 150, 0.0005, 10)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)

imshow("Corners Found", img)
