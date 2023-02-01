########################################################
# 11 contornos, encontrar dibujar jerarquía modos ######
########################################################
# ####**En esta lección aprenderemos:**
# 1. Usando findContours
# 2. Dibujo de contornos
# 3. Jerarquía de Contornos
# 4. Modos de contorno (simple vs aproximado)

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

# ## **¿Qué son los contornos?**
# Los contornos son líneas o curvas continuas ¿bordes? que limitan o cubren el límite total de un objeto en una imagen.

# Carguemos una imagen simple de placa de matrícula
image = cv2.imread('images/LP.jpg')
imshow('Input Image', image)


# #### **Aplicando cv2.findContours()**
# cv2.findContours(imagen, modo de recuperación, método de aproximación)
#
# **Modos de recuperación**
# - **RETR_LIST** - Recupera todos los contornos, pero no crea ninguna relación padre-hijo. Padres e hijos son iguales
#                   bajo esta regla, y son solo contornos. es decir, todos pertenecen al mismo nivel de jerarquía.
# - **RETR_EXTERNAL** - devuelve unicamente banderas EXTERNAS extremas. Todos los contornos secundarios se dejan atrás.
# - **RETR_CCOMP** - Esta bandera recupera todos los contornos y los organiza en una jerarquía de 2 niveles. es decir,
#                    los contornos externos del objeto (es decir, su límite) se colocan en la jerarquía-1. Y los
#                    contornos de los agujeros dentro del objeto (si los hay) se colocan en la jerarquía-2. Si hay algún
#                    objeto dentro de él, su contorno se coloca
#                    nuevamente en la jerarquía-1 solamente. Y su agujero en la jerarquía-2 y así sucesivamente.
# - **RETR_TREE** - Recupera todos los contornos y crea una lista de jerarquía familiar completa.
#
# **Opciones de método de aproximación**
# - cv2.CHAIN_APPROX_NONE – Almacena todos los puntos a lo largo de la línea (¡ineficiente!)
# - cv2.CHAIN_APPROX_SIMPLE – Almacena los puntos finales de cada línea


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# aplicamos el umbral para modificar una imagen a una representación binaria ( visto en 09 y en 07 info bit)
# se realiza porque las funciones de contorno funcionan mejor con el umbral de las imágenes y los binarios
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Encontrar contornos
# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
# le estamos metiendo la imagen binarizada, la opción de recuperar lso contronos sin jerarquia y el almacenamiento
# completo
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# + findo jerarquías en documentación OpenCV

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

# el número de contornos encontrados
print("Number of Contours found = " + str(len(contours)))

# visualizamos el primer elemento del contorno vemos que son las lista de los puntos, es una lista de píxeles que son
# el perímetro del objeto
print(contours[0])
'''[[[564 112]]
 [[563 113]]
 [[562 113]]
 [[561 113]]
 [[560 113]]...'''

# #### **¿Qué sucede si no establecemos un umbral? Cosas malas..**
'''para el trabajo de contornos finos, el fondo debe ser negro y el primer plano debe ser básicamente
blanco a cualquier otra cosa. De lo contrario, no obtendrá los contornos que desea si desea hacer.'''
image = cv2.imread('images/LP.jpg')
# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow('After Grayscaling', gray)
# Encontrar contornos
contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo

# cv2.drawContours(imagen, contornos, -1, (0,255,0), grosor = 2)
imshow('Contours overlaid on original image', image) # no encuentra

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 1


# # **NOTA: Para que findContours funcione, el fondo debe ser negro y de primer plano (es decir, el texto o los objetos)
# #### De lo contrario, deberá invertir la imagen utilizando **cv2..bitwise_not(input_image)**
# #### **Podemos usar Canny Edges en lugar de Thresholding**

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bordes astutos
edged = cv2.Canny(gray, 30, 200)
imshow('Canny Edges', edged)
# Encontrar contornos
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
#  encontramos 77 contornos, y eso es porque los bordes de Kanae crean mucho más ruido. Entonces podrías tener muchos
#  más contornos.

## ## **Recuerda estos pasos para contornear**
# 1. Escala de grises
# **Nota:** Se recomienda desenfocar antes del Paso 2 para eliminar contornos ruidosos
# 2. Detección de umbral o Canny Edge (bordes) para binarizar la imagen



# # **Modos de recuperación**
# Documento oficial: https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
#
# **Jerarquía**
# Esta matriz almacena 4 valores para cada contorno:
# - El primer término es el índice del siguiente contorno
# - El segundo término es el índice del contorno anterior
# - El tercer término es el índice del contorno padre
# - Cuarto término es el índice del contorno hijo



# ### **RETR_LIST**
# Recupera todos los contornos, pero no crea ninguna relación padre-hijo. Padres e hijos son iguales bajo esta regla, y
# son solo contornos. es decir, todos pertenecen al mismo nivel de jerarquía.

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)  # aqui hemos recuperado todos los contornos

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 38
print(hierarchy)
'''
[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 3  1 -1 -1]
  [ 4  2 -1 -1]
  [ 5  3 -1 -1]
  [ 6  4 -1 -1]
  [ 7  5 -1 -1]
'''


# ### **RETR_EXTERNO**
# Devuelve solo banderas exteriores extremas. Todos los contornos secundarios se dejan atrás.

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2) # como solo se recuperan las externas los contrnos de dentro ( por ejemplo en una O
# el de dentro no se recuperan

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image, size = 10)

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 16
print(hierarchy)
'''[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 3  1 -1 -1]
  [ 4  2 -1 -1]
  [ 5  3 -1 -1]
'''

# ### **RETR_CCOMP**
# Recupera todos los contornos y los organiza en una jerarquía de 2 niveles. es decir, los contornos externos del objeto
# (es decir, su límite) se colocan en la jerarquía-1. Y los contornos de los agujeros dentro del objeto (si los hay)
# se colocan en la jerarquía-2. Si hay algún objeto dentro de él, su contorno se coloca nuevamente en la jerarquía-1
# solamente. Y su agujero en la jerarquía-2 y así sucesivamente.


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours
contours, hierarchy = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 38
print(hierarchy)
'''[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 3  1 -1 -1]
  [ 4  2 -1 -1]
  [ 5  3 -1 -1]
  lo que ha cambiado respecto al anterior es la jerarquía'''

# ### **RETR_ÁRBOL**
# Recupera todos los contornos y crea una lista de jerarquía familiar completa.
image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)


# ## **Modos de contorno**
# #### **CHAIN_APPROX_NONE** que básicamente nos da todos los puntos


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la imagen de entrada (operación en el lugar)
# Use '-1' como tercer parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
  print(len(c))


#
# #### **CADENA_APPROX_SIMPLE**
'''Simple solo almacena los puntos finales de la luz, por lo que no almacena todas las coordenadas, sino el 
el punto final, por lo que es mucho menos espacio
ocupa'''

image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, p. edged.copy(), ya que findContours altera la imagen
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Dibuje todos los contornos, tenga en cuenta que esto sobrescribe la entrada
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
  print(len(c))


