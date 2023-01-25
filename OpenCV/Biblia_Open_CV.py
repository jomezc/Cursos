########################################################################
# 01 Primeros pasos carga visualización ahorro y dimensiones RR ######
########################################################################

#!/usr/bin/env python
# coding: utf-8

'''
primera lección de OpenCV. Aquí aprenderemos a:
1. Importar el modelo OpenCV en Python
2. Cargar imágenes
3. Mostrar imágenes
4. Guardar imágenes
5. Obtención de las dimensiones de la imagen
'''

# Así es como importamos OpenCV, no podemos usar las funciones de OpenCV sin antes hacer esto
import cv2


# Veamos qué versión estamos ejecutando
print(cv2.__version__)  # 4.7.0


# ### **Descargando imagenes**
''' comandos ipynb
# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


# ### **Cargamos las imagenes**
# Cargue una imagen usando 'imread' especificando la ruta a la imagen
image = cv2.imread('./images/castara.jpeg')


# ### **Mostramos las imagenes**
from matplotlib import pyplot as plt

# Mostramos la imagen con matpoit matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
'''usamos una funcion de open cv para cambiar el color porque Open CV en su dimensión de colores utiliza el orden Blue 
Green red, BGR y matploit usa red green blue, RGB. necesitamos esos espacios de color porque necesitamos esos 3 colores 
primarios para crear cualquier color que queramos'''
plt.show()


# Vamos a crear una función simple para hacer que mostrar nuestras imágenes sea más simple y fácil.
def imshow(title="", image = None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # cambio de color
    plt.title(title)  # le damos un título a la imagen
    plt.show()  # mostramos la imagen


# Vamos a probarlo
imshow("Displaying Our First Image", image)


# ### **Salvamos la imagen**
# Simplemente use 'imwrite' especificando el nombre del archivo y la imagen que se guardará
cv2.imwrite('output.jpg', image)

# O guárdelo como PNG (gráficos de red portátiles), que es un formato de imagen de mapa de bits sin pérdida
cv2.imwrite('output.png', image)


# ### **mostramos las dimensiones de la imagen**
# Recuerda las imágenes son arrays::
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/array.png?token=ADLZD2HNEL33JAKTYRM3B5C7WMIV4)
# Necesitamos usar numpy para realizar esta operación. No te preocupes, numpy se convertirá en uno de tus mejores amigos
# si estás aprendiendo ciencia de datos y visión artificial.

# Import numpy, librería numérica de arrays
import numpy as np

print(image.shape)  # (1280, 960, 3), es una estructura tridimensional, de ancho, alto y color

# # Para acceder a una dimensión, simplemente indícela usando 0, 1 o 2.
image.shape[0]


# Puedes ver que la primera dimensión es la altura y tiene 960 píxeles
# La segunda dimensión es el ancho, que es de 1280 píxeles.
# Podemos imprimirlos muy bien así:
print('Height of Image: {} pixels'.format(int(image.shape[0])))  # Height of Image: 1280 pixels
print('Width of Image: {} pixels'.format(int(image.shape[1])))  # Width of Image: 960 pixels
print('Depth of Image: {} colors components'.format(int(image.shape[2])))  # Depth of Image: 3 colors components

# !/usr/bin/env python
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
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]  # desempaquetamos la relación de aspecto mediante el ancho y el alto
    aspect_ratio = w / h  # calculamos la relación de aspecto
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


def imshow(title="", image=None, size=10):
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w / h
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


#!/usr/bin/env python
# coding: utf-8

###########################
# 03 Espacios de color ####
###########################
'''# # **Color Spaces**
In this lesson we'll learn to:
1. View the individual channels of an RGB Image
2. Manipulate a color space
3. Introduce HSV Color Spaces
'''

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

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


# Carga nuestra imagen de entrada
image = cv2.imread('./images/castara.jpeg')

# Use cv2.split para obtener cada espacio de color por separado
# separa los tuneles en los componenetes de arbol azúl, verde y rojo, conviertiéndose en imágenes bidimensionales
B, G, R = cv2.split(image)
print(B.shape)  # (1280, 960)
print(G.shape)  # (1280, 960)
print(R.shape)  # (1280, 960)

'''Cada espacio de color que esté encendido se verá como una escala de grises ya que carece de los otros canales de 
color, esto es porque tiene sólo una dimensión, son sólo las intensidades en grado de componente de color azul
'''
imshow("Blue Channel Only", B)

import numpy as np

'''Vamos a crear el arbol de la imagen de la dimensión del árbol vamos a hacer todos los otros componentes de color
a cero menos el que queremos visualizar, mediante la siguiente matriz'''
# Vamos a crear una matriz de ceros con dimensiones de la imagen h x w
zeros = np.zeros(image.shape[:2], dtype = "uint8")

imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))

#####
# por otro lado, recargamos la imagen Original
image = cv2.imread('./images/castara.jpeg')

# La función 'dividir' de OpenCV divide la imagen en cada índice de color
B, G, R = cv2.split(image)

# Rehagamos una copia de la imagen original, observando que se muestra la misma imagen
merged = cv2.merge([B, G, R])
imshow("Merged", merged)


# Ampliemos el color azul, se ve extraño
merged = cv2.merge([B+100, G, R])
imshow("Blue Boost", merged)


# ## **The HSV Color Space**
'''#  en vez de usar una combinación de los colores RGB, sua un mapa de color llamado tono (HUE) del azul al amarillo con
la intensidad, que es el brillo pudiendo ver hacia abajo los colores más oscuros, y la saturación, que te dice los
conflictos alimentados en apelación, volíendose más rico y profundo a medida que avanza.
básicamente usando este esquema hay una fomra diferente de representar los colores de diferentes espacios de color'''
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f2/HSV_color_solid_cone.png)
# - Matiz HUE: 0 - 179
# - Saturación: 0 - 255
# - Valor (Intensidad): 0 - 255

# Recargamos la imagen
image = cv2.imread('./images/castara.jpeg')

# convertimos a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)

# #### Esto se ve extraño... ¿por qué?
# Porque nuestra función de trazado fue diseñada solo para imágenes RGB, no para HSV

plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
plt.show()


# ### **Veamos cada tipo de canal en la representación del espacio de color HSV**
# Volviendo a ver la representación RGB mediante el uso de indexación
#  HUE es en realidad el color naranja, para que puedas ver la arena y los árboles ( intensidad)
imshow("Hue", hsv_image[:, :, 0])  #
imshow("Saturation", hsv_image[:, :, 1])  # cuanto mas brillante en la saturación
imshow("Value", hsv_image[:, :, 2])  # intensidad de brillo

#!/usr/bin/env python
# coding: utf-8

###############################
# 04 Dibujando en imágenes ####
###############################

# ## **Dibujar imágenes y formas usando OpenCV**
# Primero, importemos OpenCV y numpy y definamos nuestra función imshow


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


# Empecemos por hacer un lienzo cuadrado en blanco

# Cree una imagen negra usando numpy para crear una matriz de negro
# array tridimensional, tamaño 512x512 de 3 canales de tipo entero de 0 a 255, todo a 0 significa negro
image = np.zeros((512,512,3), np.uint8)

# ¿Podemos hacer esto en blanco y negro? escala de grises
image_gray = np.zeros((512,512), np.uint8)

# El negro sería lo mismo que una imagen en escala de grises o en color (lo mismo para el blanco)
# el 1º ocupa 3 veces más memoria por las 3 dimensiones
imshow("Black Canvas - RGB Color", image)
imshow("Black Canvas - Grayscale", image_gray)


# ### **Dibujemos una línea sobre nuestro cuadrado negro**
# cv2.line(imagen, coordenadas iniciales, coordenadas finales, color, espesor)

# Tenga en cuenta que esta es una operación en el lugar, lo que significa que cambia la imagen de entrada
# A diferencia de muchas otras funciones de OpenCV que devuelven una nueva imagen sin afectar la entrada
# Recuerda que nuestra imagen era el lienzo negro
cv2.line(image, (0,0), (511,511), (255,127,0), 5)

imshow("Black Canvas With Diagonal Line", image)


# ### **Drawing Rectangles**
# cv2.rectangle(imagen, vértice inicial (sup izq), vértice opuesto (inf der), color, espesor)
# Vuelva a crear nuestro lienzo negro porque ahora tiene una línea
image = np.zeros((512,512,3), np.uint8)

# Espesor (ultimo parámetro) - si es positivo. Espesor -1 rellena el objeto
cv2.rectangle(image, (100,100), (300,250), (127,50,127), 10)
imshow("Black Canvas With Pink Rectangle", image)


# ### **Dibujemos algunos círculos**
# cv2.circle(imagen, centro, radio, color, relleno)
# de nuevo la imagen negra ...
image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,150,50), -1)
imshow("Black Canvas With Green Circle", image)


# ### **Polygons**
# ```cv2.polylines(imagen, puntos, ¿Cerrado?, color, grosor)```
# si Cerrado = Verdadero, unimos el primer y último punto.
# De nuevo reseteamos la imagen negra ...
image = np.zeros((512, 512, 3), np.uint8)

# Definamos cuatro puntos mediante un array, una matriz con subpuntos dentro
pts = np.array([[10,50], [400,50], [90,200], [50,500]], np.int32)
pts.shape   # (4,2)
# **Nota** cv2.polylines requiere que nuestros datos tengan la siguiente forma:
# Ahora remodelemos nuestros puntos en la forma requerida por las polilíneas ( en realidad solo cambia el formato)

print(pts)
'''[[ 10  50]
 [400  50]
 [ 90 200]
 [ 50 500]]'''
# estás agregando un 1, en una dimensión adicional en medio por como funciona polylines internamente, como decodifica
# los puntos
pts = pts.reshape((-1, 1, 2))
pts.shape  # (4, 1 ,2)
'''[[[ 10  50]]
 [[400  50]]
 [[ 90 200]]
 [[ 50 500]]]'''
print(pts)

cv2.polylines(image, [pts], True, (0,0,255), 3)
imshow("Black Canvas with Red Polygon", image)

# ### **Y ahora para agregar texto con cv2.putText**
# cv2.putText(imagen, 'Texto para mostrar', punto de inicio inferior izquierdo, Fuente, Tamaño de fuente, Color, Grosor)

# **Fuentes disponibles**
# - FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
# - FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX
# - FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
# - FONT_HERSHEY_SCRIPT_SIMPLEX
# - FONT_HERSHEY_SCRIPT_COMPLEX

image = np.zeros((1000,1000,3), np.uint8)
ourString =  'Hello World!'
cv2.putText(image, ourString, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 4)
imshow("Messing with some text", image)

#!/usr/bin/env python
# coding: utf-8

#########################################################
#  **Transformaciones - Traslaciones y Rotaciones**######
#########################################################

# En esta lección aprenderemos a:
# 1. Realizar traducciones de imágenes
# 2. Rotaciones con getRotationMatrix2D
# 3. Rotaciones con transposición
# 4. Voltear imágenes

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

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ### **translaciones**
# Esta es una transformación afín que simplemente cambia la posición de una imagen. (izquierda o derecha).
# No cambia la relación de aspecto, Básicamente lo mueve hacia la izquierda, hacia arriba o hacia abajo
# Usamos cv2.warpAffine para implementar estas transformaciones.

# cv2.warpAffine(imagen, T, (ancho, alto))
# multiplica la imagen por una matriz T, en el que Tx representa el turno alrededor del eje horizontal y Ty vertical
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/warp.png)


# Carga nuestra imagen
image = cv2.imread('images/Volleyball.jpeg')
imshow("Original", image)

# Almacenar alto y ancho de la imagen
height, width = image.shape[:2]

# Lo cambiamos por un cuarto de la altura y el ancho
quarter_height, quarter_width = height/4, width/4

# Nuestra matriz de translación
#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T es nuestra matriz de translación con un cuarto del ancho para Tx y un cuarto de la altura pars Ty
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
# ¿Cómo se ve T?
print(T)
'''
[[  1.   0. 320.]
 [  0.   1. 180.]]'''
print(height, width )  # 720 1280

# Usamos warpAffine para transformar la imagen usando la matriz, T. Lo que está haciendo es cambiar el punto de
# referencia, viendose la imagen como movida a la izquierda y abajo sobre un fondo negro
img_translation = cv2.warpAffine(image, T, (width, height))
imshow("Translated", img_translation)


# ### **Rotaciones**
# toma el punto de rotación x e y ( punto central o donde esté un pivote) y 'gira' la imagen (como un editor de fotos)
# por el ángulo de rotación elegido ( antihorario)  y escala ( 1 significa mantener)
# cv2.getRotationMatrix2D(rotación_centro_x, rotación_centro_y, ángulo de rotación, escala)
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/rotation.png)

# Carga nuestra imagen
image = cv2.imread('images/Volleyball.jpeg')
height, width = image.shape[:2]

# Divide por dos para rotar la imagen alrededor de su centro, rota la imagen sino que crea la matriz que necesitamos
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)  # obtenemos la matriz 2D de rotación

# Esa matriz de rotación es lo que usamos en la translación
# Ingrese nuestra imagen, la matriz de rotación y nuestro ancho y alto final deseado
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))  # es la función que rota la imagen
imshow("Rotated 90 degrees with scale = 1", rotated_image)

# Otro ejemplo cambiando la escala
# Divide por dos para rotar la imagen alrededor de su centro
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)
# ese 0.5 reduce la escala de la imagen la hace más pequeña
print(rotation_matrix)
'''[[ 3.061617e-17  5.000000e-01  4.600000e+02]
 [-5.000000e-01  3.061617e-17  6.800000e+02]]'''
# Ingrese nuestra imagen, la matriz de rotación y nuestro ancho y alto final deseado
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 0.5", rotated_image)
# Observe todo el espacio negro que rodea la imagen.
# Ahora podemos recortar la imagen ya que podemos calcular su nuevo tamaño (¡aún no hemos aprendido a recortar!).


# ### **Rotaciones con cv2.transpose** (menos flexible)
rotated_image = cv2.transpose(image)  # menos control de lo que hacemos, solo gira la imagen y la hace espejo
imshow("Original", image)
imshow("Rotated using Transpose", rotated_image)


rotated_image = cv2.transpose(image)
# si lo hacemos dos veces obtenemos 'lo contrario', la imagen original
rotated_image = cv2.transpose(rotated_image)

imshow("Rotated using Transpose", rotated_image)


# Vayamos ahora a un giro horizontal 90º, un 'volteo'
flipped = cv2.flip(image, 1)
imshow("Horizontal Flip", flipped)


#!/usr/bin/env python
# coding: utf-8

###################################################################
# # 06 Escalado, cambio de tamaño, interpolaciones y recorte** ####
###################################################################
# **En esta lección aprenderemos:**
# 1. Cómo redimensionar y escalar imágenes
# 2. Pirámides de imágenes
# 3. Recortar

# ### **Cambio de tamaño**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Resizing.png)
#
# Cambiar el tamaño es una función simple que ejecutamos usando la función cv2.resize, sus argumentos son:

# cv2.resize(imagen, dsize(tamaño de la imagen de salida), escala x, escala y, interpolación)
# - si dsize es Ninguno, la imagen de salida se calcula en función de la escala usando la escala x e y

# la interpolación es básicamente un algoritmo para encontrar un valor entre dos puntos. si tuviéramos unos puntos por
# una ruta de gps la interpolación adivinará puntos intermedios entre los originales del camino, aportando información
# adicional, e suna forma de agregar más datos a los existentes para conectar los puntos existentes ( en el ejemplo)
# si estamos agrandando una imagen, estamos tratando de adivinar los puntos que se tomarán en una nueva dimensión.
# algorítmicamente adivina la mejor suposición
# #### **Lista de métodos de interpolación, las diferentes fórmulas que suelen aplicarse:**
# - cv2.INTER_AREA- Bueno para reducir o reducir el muestreo
# - cv2.INTER_NEAREST - Más rápido
# - cv2.INTER_LINEAR- Bueno para hacer zoom o muestreo ascendente (predeterminado)
# - cv2.INTER_CUBIC- Mejor
# - cv2.INTER_LANCZOS4 - El Mejor


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ### **Tipos de métodos de reescalado en OpenCV**
#
# - **INTER_NEAREST** – una interpolación de vecino más cercano
# - **INTER_LINEAR** – una interpolación bilineal (usada por defecto)
# - **INTER_AREA** – remuestreo usando relación de área de píxeles. Puede ser un método preferido para la destrucción
#                    de imágenes, ya que brinda resultados sin muaré. Pero cuando se amplía la imagen, es similar al
#                    método INTER_NEAREST.
# - **INTER_CUBIC**: una interpolación bicúbica sobre una vecindad de 4×4 píxeles
# - **INTER_LANCZOS4**: una interpolación de Lanczos sobre un vecindario de 8×8 píxeles
#
# Vea más sobre su desempeño - https://chadrick-kwag.net/cv2-resize-interpolation-methods/

# carga nuestra imagen de entrada
image = cv2.imread('images/oxfordlibrary.jpeg')
imshow("Scaling - Linear Interpolation", image)

# Si no se especifica ninguna interpolación, cv.INTER_LINEAR se usa por defecto
# Hagamos nuestra imagen 3/4 de su tamaño original
# vamos a usar los efectos del argumento y la forma para reducir la imagen en un 75% (0.75 de ancho y alto)
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
imshow("0.75x Scaling - Linear Interpolation", image_scaled)

# Dupliquemos el tamaño de nuestra imagen
img_scaled2 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
imshow("2x Scaling - Inter Cubic", img_scaled2)

# Dupliquemos el tamaño de nuestra imagen usando la interpolación inter_nearest
img_scaled3 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
imshow("2x Scaling - Inter Nearest", img_scaled3)

# Sesguemos el cambio de tamaño estableciendo dimensiones exactas
img_scaled4 = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
imshow("Scaling - Inter Area", img_scaled4)


# ## **Imagen de pirámides**
# Es una operación mucho más rápida, siendo una forma rápida de ampliar una imagen escalada
# Básicamente, duplique recuzca la mitad del tamaño
image = cv2.imread('images/oxfordlibrary.jpeg')

smaller = cv2.pyrDown(image)  # reduce la mitad
larger = cv2.pyrUp(smaller)  # dobla la imagen

imshow("Original", image)
imshow('Smaller', smaller)
imshow('Larger', larger)

even_smaller = cv2.pyrDown(smaller)
imshow('Even Smaller', even_smaller)


# # **Recorte**
# es una técnica muy útil especialmente con detectores de objetos o OCR donde tienes que recortar segmentos de la imagen
image = cv2.imread('images/oxfordlibrary.jpeg')

# Obtenga las dimensiones de nuestra imagen
height, width = image.shape[:2]

# Obtengamos las coordenadas del píxel inicial (arriba a la izquierda del rectángulo de recorte)
# usando 0.25 para obtener la posición x,y que está 1/4 por debajo de la parte superior izquierda (0,0)

start_row, start_col = int(height * .25), int(width * .25)

# Obtengamos las coordenadas del píxel final (abajo a la derecha)
end_row, end_col = int(height * .75), int(width * .75)

# Simplemente use la indexación para recortar el rectángulo que deseamos
# hace lo que se supone, es decir recorta la imagen
cropped = image[start_row:end_row, start_col:end_col]

imshow("Original Image", image)

# La función cv2.rectangle dibuja un rectángulo sobre nuestra imagen (operación in situ)
copy = image.copy()
cv2.rectangle(copy, (start_col,start_row), (end_col,end_row), (0,255,255), 10)

imshow("Area we are cropping", copy)

imshow("Cropped Image", cropped)

#!/usr/bin/env python
# coding: utf-8

################################################
# 07 Operaciones aritméticas y bit a bit** #####
################################################

# #### **En esta lección aprenderemos:**
# 1. Operaciones aritméticas, aquellas que nos permiten sumar o restar la intensidad o los valores de la imagen
# 2. Operaciones bit a bit

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

'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ## **Operaciones aritméticas**
# Son operaciones sencillas que nos permiten sumar o restar directamente a la intensidad del color.
# Calcula la operación por elemento de dos matrices. El efecto general es aumentar o disminuir el brillo.


# 0 como segundo argumento en cv2.imread carga nuestra imagen como una imagen en escala de grises
image = cv2.imread('images/liberty.jpeg', 0)  # 0 es como escala de grises
imshow("Grayscaled Image",  image)
print(image)


# Crea una matriz de unos con el tamaño de la imagen, luego multiplícala por un escalador de 100
# Esto da una matriz con las mismas dimensiones de nuestra imagen con todos los valores siendo 100
M = np.ones(image.shape, dtype = "uint8") * 100

print(M)

# #### **Brillo creciente**
# Usamos esto para agregar esta matriz M, a nuestra imagen, la función respeta los valores de 0 a 255 dejando el máximo
# Note el aumento en el brillo
added = cv2.add(image, M)
imshow("Increasing Brightness", added)

# Ahora si lo acabamos de agregar, pero al no usar la función el valor sobrepasa el 255 con lo que se resetea a
# 0 sumándole la diferencia por ejemplo si es 288, pues 33 con lo que no se ve como se espera
added2 = image + M
imshow("Simple Numpy Adding Results in Clipping", added2)


# #### **Reducción del brillo**

# Así mismo también podemos restar
# Note la disminución en el brillo
subtracted = cv2.subtract(image, M)
imshow("Subtracted", subtracted)

subtracted = image - M  # aquí pasa lop mismo que antes pero al reves los valores se quedan negativos y al no permitirse
# van de 255 hacia abajo
imshow("Subtracted 2", subtracted)


# ## **Operaciones bit a bit y enmascaramiento**
# Para demostrar estas operaciones, creemos algunas imágenes simples
# Si se pregunta por qué solo dos dimensiones, bueno, esta es una imagen en escala de grises,
# Hacer un cuadrado

# Hacer un cuadrado
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
imshow("square", square)

# Haciendo una elipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
imshow("ellipse", ellipse)


# ### **Experimentando con algunas operaciones bit a bit como AND, OR, XOR y NOT**
# Muestra solo donde se cruzan, es decir donde ambos tienen info distinta de 0 (0 negro, 255 blanco solo 1 si los dos 1)
And = cv2.bitwise_and(square, ellipse)
imshow("AND", And)

# Muestra la información de ambos (1 si alguno de los 2 1) dónde está el cuadrado o la elipse
bitwiseOr = cv2.bitwise_or(square, ellipse)
imshow("bitwiseOr", bitwiseOr)

# Muestra dónde existen por sí mismos (1 si solo uno de ellos 1)
bitwiseXor = cv2.bitwise_xor(square, ellipse)
imshow("bitwiseXor", bitwiseXor)

# Muestra todo lo que no es parte del cuadrado ( lo contrario)
bitwiseNot_sq = cv2.bitwise_not(square)
imshow("bitwiseNot_sq", bitwiseNot_sq)

# Observe que la última operación invierte la imagen totalmente


#!/usr/bin/env python
# coding: utf-8

############################################################
# 08 Convoluciones, desenfoque y nitidez de imágenes** #####
############################################################
# ####**En esta lección aprenderemos:**
# 1. Operaciones de convolución: una convolución es una operación matemática realizada en dos funciones que producen
#    una función escalonada que generalmente es una versión modificada de una de las funciones originales. No es más que
#    una multiplicación de funciones (primer elemento X resto y la suma de los resultados)
# 2. Desenfoque
# 3. Eliminación de ruido
# 4. Afilado


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

'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


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
# templateWindowSize: tamaño en píxeles del parche de plantilla que se utiliza para calcular los pesos. Debería ser
#                     extraño. Valor recomendado 7 píxeles
# - **searchWindowSize**: tamaño en píxeles de la ventana que se utiliza para calcular el promedio ponderado de un
#                         píxel determinado. Debería ser extraño. Afecta el rendimiento de forma lineal: mayor tamaño
#                         de ventana de búsqueda, mayor tiempo de eliminación de ruido. Valor recomendado 21 píxeles
# - **h** – Parámetro que regula la intensidad del filtro para el componente de luminancia. Un valor h más grande
#           elimina perfectamente el ruido pero también elimina los detalles de la imagen, un valor h más pequeño
#           conserva los detalles pero también conserva algo de ruido
# - **hColor** – Lo mismo que h pero para componentes de color. Para la mayoría de las imágenes, el valor igual a 10
#                será suficiente para eliminar el ruido de color y no distorsionar los colores.


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


#!/usr/bin/env python
# coding: utf-8

########################################################################
# 09 Umbralización, binarización y umbralización adaptativa ######
########################################################################
# ####**En esta lección aprenderemos:**
# 1. Imágenes binarizadas, estamos conviertiendo a binario los colores, los píxeles de una imagen a 0 o 1, mediante un
#    algoritmo de sesión binaria.
# 2. Métodos de Umbral
# 3. Umbral adaptativo
# 4. Umbral local de SkImage
# In[1]:


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


# ### **Métodos de umbral**
#  explicación de que la binarización Convirtió pasa a blanco o negro una escala de grises de una imagen en mediante un
#  umbral todo por encima de un cierto umbral se vuelve blanco y por debajo negro mediant eun algortimo , exisitiendo la
#  operación binaria contraria ( en vez de blanco negro y viceversa).
#  El truncamiento es que todo lo que está por encima d eun umbral se convierte en ese valor máximo del umbral
#  TOZERO es que todo lo que es menor que el umbral se vuelve 0 y TOZERO_INV lo contrario
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.57.55%20am.png)
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.58.09%20am.png)
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html


'''
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/scan.jpeg')
'''

# Cargar nuestra imagen en escala de grises
image = cv2.imread('./images/scan.jpg',0)
imshow("Original", image)

# Los valores por debajo de 127 van a 0 o negro, por encima va a 255 (blanco)
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('1 Threshold Binary @ 127', thresh1)

# Los valores por debajo de 127 van a 255 y los valores por encima de 127 van a 0 (inverso de arriba)
ret,thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
imshow('2 Threshold Binary Inverse @ 127', thresh2)

# Los valores por encima de 127 se truncan (se mantienen) en 127 (el argumento 255 no se usa)
ret,thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
imshow('3 THRESH TRUNC @ 127', thresh3)

# Los valores por debajo de 127 van a 0, por encima de 127 no cambian
ret,thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
imshow('4 THRESH TOZERO @ 127', thresh4)

# Inverso de lo anterior, por debajo de 127 no cambia, por encima de 127 pasa a 0
ret,thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
imshow('5 THRESH TOZERO INV @ 127', thresh5)


# #### **parámetros cv2.adaptiveThreshold**
# si queremos calcular automáticamente el umbral, usamos el umbral adaptativo, son pequeños algoritmos que en realidad
# ejecutan algunos cálculos en la imagen y tratan de averiguar el valor umbral óptimo.
# ``**cv2.adaptiveThreshold**(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst``
#
# - **src** – Imagen de origen de un solo canal de 8 bits.
# - **dst** – Imagen de destino del mismo tamaño y del mismo tipo que src .
# - **maxValue**: valor distinto de cero asignado a los píxeles para los que se cumple la condición. Vea los detalles
#                 a continuación.
# - **adaptiveMethod**: algoritmo de umbral adaptable para usar, ADAPTIVE_THRESH_MEAN_C o ADAPTIVE_THRESH_GAUSSIAN_C.
#                       Vea los detalles a continuación.(mejor el segundo)
# - **thresholdType**: tipo de umbral que debe ser THRESH_BINARY o THRESH_BINARY_INV.
# - **blockSize**: tamaño de una vecindad de píxeles que se utiliza para calcular un valor de umbral para el píxel: 3,
#                   5, 7, etc.
# - **C** – Constante restada de la media o media ponderada. Normalmente, es positivo, pero también puede ser cero o
#           negativo.

image = cv2.imread('./images/scan.jpg',0)
imshow("Original", image)

# MANUAL
# Los valores por debajo de 127 van a 0 (negro, todo lo anterior va a 255 (blanco)
# 127 es el umbral y 255 el máximo
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('Threshold Binary', thresh1)

# Uso de umbral adaptativo # 3 y 5 por defecto en la documentación
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
imshow("Adaptive Mean Thresholding", thresh)

# otra forma que se explica en la documentación, no es muy intuitivo por el umbral que establece pero funciona bien
_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Otsu's Thresholding", th2)

# Umbralización de Otsu después del filtrado gaussiano
# Es una buena práctica desenfocar las imágenes ya que elimina el ruido
# imagen = cv2.GaussianBlur(imagen, (3, 3), 0)
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Guassian Otsu's Thresholding", th3)

# ### **Umbral local de SkImage** USABLE EN LA VIDA REAL MUY BUENO
# umbral_local(imagen, tamaño_bloque, desplazamiento=10)
# La función Threshold_local calcula umbrales en regiones con un tamaño característico ``block_size`` que rodea cada
# píxel (es decir, vecindarios locales). Cada valor de umbral es la media ponderada del vecindario local menos un valor
# de ``compensación``
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

from skimage.filters import threshold_local

image = cv2.imread('./images/scan.jpg')

# Obtenemos el componente Valor del espacio de color HSV, lo necesita esta función
# luego aplicamos un umbral adaptativo a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Apply the threshold operation
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)


# ### **Por qué es importante desenfocar
# ## **respuesta - ruido *
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/otsu.jpg)
# https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html


#!/usr/bin/env python
# coding: utf-8
#####################################################
# 10 Detección de dilatación, erosión y bordes ######
#####################################################
# # **Detección de dilatación, erosión y bordes**
# ####**En esta lección aprenderemos:**
# - **Dilatación**: agrega píxeles a los límites de los objetos en una imagen
# - **Erosión**: elimina píxeles en los límites de los objetos en una imagen
# - **Apertura** - Erosión seguida de dilatación
# - **Cierre** - Dilatación seguida de erosión
# 5. Detección de borde astuto

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

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-11-15%20at%205.19.08%20pm.png)

image = cv2.imread('images/opencv_inv.png', 0)
imshow('Original', image)

# Definamos el tamaño de nuestro kernel, es una matriz que usamos para la convolución 2D quw se realiza en estas
# funciones
kernel = np.ones((5, 5), np.uint8)

# Ahora erosionamos
erosion = cv2.erode(image, kernel, iterations = 1)
imshow('Erosion', erosion)

# Dilatar aqui
dilation = cv2.dilate(image, kernel, iterations = 1)
imshow('Dilation', dilation)

# Apertura - Bueno para eliminar el ruido
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
imshow('Opening',opening)

# Cierre - Bueno para eliminar el ruido
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
# - El primer argumento es nuestra imagen de entrada.
# - El segundo y tercer argumento son nuestro minVal y maxVal respectivamente.
# - El cuarto argumento (opcional) es opening_size. Es el tamaño del núcleo Sobel utilizado para encontrar gradientes
#   de imagen. Por defecto es 3.
#
# La detección de bordes necesita un umbral para indicar qué diferencia/cambio debe contarse como borde

image = cv2.imread('images/londonxmas.jpeg',0)

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


#!/usr/bin/env python
# coding: utf-8
#####################
# 11 contornos ######
#####################
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

'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


# ## **¿Qué son los contornos?**
# Los contornos son líneas o curvas continuas que limitan o cubren el límite total de un objeto en una imagen.

# Carguemos una imagen simple de placa de matrícula
image = cv2.imread('images/LP.jpg')
imshow('Input Image', image)


# #### **Aplicando cv2.findContours()**
# cv2.findContours(imagen, modo de recuperación, método de aproximación)
#
# **Modos de recuperación**
# - **RETR_LIST** - Recupera todos los contornos, pero no crea ninguna relación padre-hijo. Padres e hijos son iguales
# bajo esta regla, y son solo contornos. es decir, todos pertenecen al mismo nivel de jerarquía.
# - **RETR_EXTERNAL** - devuelve solo banderas externas extremas. Todos los contornos secundarios se dejan atrás.
# - **RETR_CCOMP** - Esta bandera recupera todos los contornos y los organiza en una jerarquía de 2 niveles. es decir,
# los contornos externos del objeto (es decir, su límite) se colocan en la jerarquía-1. Y los contornos de los agujeros
# dentro del objeto (si los hay) se colocan en la jerarquía-2. Si hay algún objeto dentro de él, su contorno se coloca
# nuevamente en la jerarquía-1 solamente. Y su agujero en la jerarquía-2 y así sucesivamente.
# - **RETR_TREE** - Recupera todos los contornos y crea una lista de jerarquía familiar completa.
#
# **Opciones de método de aproximación**
# - cv2.CHAIN_APPROX_NONE – Almacena todos los puntos a lo largo de la línea (¡ineficiente!)
# - cv2.CHAIN_APPROX_SIMPLE – Almacena los puntos finales de cada línea


image = cv2.imread('images/LP.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# aplicamos el un umbral para modificar una imagen a una representación binaria ( visto en 09 y en 07 info bit)
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


# # **NOTA: Para que findContours funcione, el fondo debe ser negro y de primer plano (es decir, el texto o los objetos)**
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
# 2. Detección de umbral o Canny Edge para binarizar la imagen
# **Nota:** Se recomienda desenfocar antes del Paso 2 para eliminar contornos ruidosos



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

# !/usr/bin/env python
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
def imshow(title="Image", image=None, size=16):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Encuentra bordes Canny
edged = cv2.Canny(gray, 50, 200)
imshow('Canny Edges', edged)

# Encuentre contornos e imprima cuántos se encontraron
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

# Dibuja todos los contornos sobre una imagen en blanco
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
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

print("Contor Areas after sorting...")  # [90222.0, 66579.5, 22901.5, 20587.5]
print(get_contour_areas(sorted_contours))

# Iterar sobre nuestros contornos y dibujar uno a la vez
for (i, c) in enumerate(sorted_contours):
    M = cv2.moments(c)  # estamos sacando el punto central del contorno
    # Esa M contiene un diccionario con las claves m00 m10 y los valores que son los puntos o pixeles... son una forma
    # de calcular los puntos xy
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # vamos a poner en el punto central de cada área de cada contorno un texto
    cv2.putText(image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    # dibujamos los contornos
    cv2.drawContours(image, [c], -1, (255, 0, 0), 3)

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
        return (int(M['m10'] / M['m00']))
    else:
        pass


def label_contour_center(image, c):
    """Coloca un círculo rojo en los centros de los contornos. Usando los momentos para sacar las coordenadas x e y"""
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Dibuja el número de contorno en la imagen
    cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
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
contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

# Etiquetado de contornos de izquierda a derecha
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(orginal_image, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orginal_image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
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
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
template = cv2.imread('images/4star.jpg', 0)
imshow('Template', template)

# Cargue la imagen de destino con las formas que estamos tratando de hacer coincidir
target = cv2.imread('images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

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

cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
imshow('Output', target)

# !/usr/bin/env python
# coding: utf-8

###################################################
# 13 Detección de líneas, círculos y manchas ######
###################################################

# ####**En esta lección aprenderemos:**
# 1. Líneas de vida ( metodo par alíneas finas en imágenes )
# 2. Líneas de Hough probabilísticas
# 3. Círculos de Hough
# 4. Detección de manchas

# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Descarga y descomprime nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''
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
edges = cv2.Canny(gray, 100, 170, apertureSize=3)  # binarizamos la imagen

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
# cv2.HoughLinesP(imagen binarizada, precisión 𝜌, precisión 𝜃, umbral, longitud mínima de línea, espacio máximo entre líneas)


# Escala de grises y Canny Edges extraídos
image = cv2.imread('images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize=3)

# Nuevamente usamos las mismas precisiones rho y theta
# Sin embargo, especificamos un voto mínimo (pts a lo largo de la línea) de 100
# y longitud de línea mínima de 3 píxeles y espacio máximo entre líneas de 25 píxeles
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 3, 25)
print(lines.shape)  # (63, 1, 4)

for x in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

imshow('Probabilistic Hough Lines', image)  # este no funciona tan bien como el otro

# ## **Detección de círculos - Hough Cirlces**
#
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


'''get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Circles_Packed_In_Square_11.jpeg')'''

image = cv2.imread('images/Circles_Packed_In_Square_11.jpeg')
imshow('Circles', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# medianBlur toma una mediana de todos los píxeles debado del kernel_size (5) y reemplaza el elemento  con ese valor
# central, efectivo para reducir el ruido, desenfoca la imagen
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)

cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # dibujar el círculo exterior
    cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 5)

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
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Mostrar puntos clave
imshow("Blobs", blobs)

# !/usr/bin/env python
# coding: utf-8

########################################################
# 14 Contar círculos, elipses y encontrar a Waldo*######
########################################################

# ####**En esta lección aprenderemos:**
# 1. Mini proyecto sobre el conteo de manchas circulares
# 2. Mini proyecto sobre el uso de la coincidencia de plantillas para encontrar a Waldo


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title="Image", image=None, size=12):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# ## **Contar manchas circulares**
# la siguiente imagen ayuda mucho a enterder los parametros!
# ![](https://i.stack.imgur.com/zYL2C.jpg)
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/simpleblob.png)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar imagen
image = cv2.imread("images/blobs.jpg", 0)
imshow('Original Image', image)

# Inicialice el detector usando los parámetros predeterminados
detector = cv2.SimpleBlobDetector_create()

# Detectar manchas
keypoints = detector.detect(image)

# Dibujar manchas en nuestra imagen como círculos rojos
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Mostrar imagen con puntos clave de blob
imshow("Blobs using default parameters", blobs)

# Establecer nuestros parámetros de filtrado
# Inicializa la configuración de parámetros usando cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Establecer parámetros de filtrado de área (tamaño del círculo)
params.filterByArea = True
params.minArea = 100

# Establecer parámetros de filtrado de circularidad (más o menos circular, es decir un triangulo tiene muy
# poca, un cuadrado más, polígono tiene mucha más... hasta llegar al círculo, 0.9 muy circular
params.filterByCircularity = True
params.minCircularity = 0.9

# Establecer parámetros de filtrado de convexidad, si está completo el círculo, imaginándolo como una tarta las
# porciones que tiene
params.filterByConvexity = False
params.minConvexity = 0.2

# Establecer parámetros de filtrado de inercia, si es un circulo perfecto o más una elipse, es decir si es redondo
# o más 'chafado'
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Crear un detector con los parámetros
detector = cv2.SimpleBlobDetector_create(params)

# Detectar manchas
keypoints = detector.detect(image)

# Dibujar manchas en nuestra imagen como círculos rojos
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Mostrar manchas
imshow("Filtering Circular Blobs Only", blobs)

# # **Buscando a Waldo usando la coincidencia de plantillas**
#
# #### **Notas sobre la coincidencia de plantillas**
#
# Hay una variedad de métodos para realizar la comparación de plantillas, pero en este caso estamos usando el
# coeficiente de correlación que se especifica mediante el indicador **cv2.TM_CCOEFF.**
#
# Entonces, ¿qué está haciendo exactamente la función cv2.matchTemplate?
# Esencialmente, esta función toma una "ventana deslizante" de nuestra imagen de consulta de waldo y la desliza a
# través de nuestra imagen de rompecabezas de izquierda a derecha y de arriba a abajo, un píxel a la vez. Luego, para
# cada una de estas ubicaciones, calculamos el coeficiente de correlación para determinar cuán "buena" o "mala" es la
# coincidencia.
#
# Las regiones con una correlación suficientemente alta pueden considerarse "coincidencias" para nuestra plantilla de
# waldo.A partir de ahí, todo lo que necesitamos es una llamada a cv2.minMaxLoc en la Línea 22 para encontrar dónde
#  están nuestras "buenas" coincidencias. ¡Eso es realmente todo lo que hay que hacer para hacer coincidir plantillas!

# http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html


template = cv2.imread('./images/waldo.jpg')
imshow('Template', template)

# Cargue la imagen de entrada y conviértala a escala de grises
image = cv2.imread('./images/WaldoBeach.jpg')
imshow('Where is Waldo?', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cargar imagen de plantilla
template = cv2.imread('./images/waldo.jpg', 0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Crear cuadro delimitador
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 5)

imshow('Where is Waldo?', image)

# !/usr/bin/env python
# coding: utf-8
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


'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

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

########################################################################
# Detección de caras y ojos con clasificadores en cascada Haar ######
########################################################################
# ####**En esta lección aprenderemos:**
# 1. A utilizar un clasificador en cascada de Haar para detectar caras
# 2. utilizar un clasificador Haarcascade para detectar ojos.
# 3. Usar un clasificador Haarcascade para detectar caras y ojos desde su webcam en Colab.


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Descargar y descomprimir nuestras imágenes y clasificadores Haarcascade
'''get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip')

get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq haarcascades.zip')
'''

# ### **Primero, ¿Qué es la Detección de Objetos?**
# ![](https://miro.medium.com/max/739/1*zlWrCk1hBBFRXa5t84lmHQ.jpeg)
#
# **Detección de Objetos** es la capacidad de detectar y clasificar objetos individuales en una imagen y dibujar un
# cuadro delimitador sobre el área del objeto.


# # **Clasificadores en cascada HAAR**
# Desarrollados por Viola y Jones en 2001.
# Método de detección de objetos que utiliza una serie de clasificadores (en cascada) para identificar objetos en una
# imagen. Están entrenados para identificar un tipo de objeto, sin embargo, podemos utilizar varios de ellos en
# paralelo, por ejemplo, detectar ojos y caras juntos.
# Los clasificadores HAAR se entrenan utilizando muchas imágenes
# positivas (es decir, imágenes con el objeto presente) e imágenes negativas (es decir, imágenes sin el objeto
# presente). Estos clasificadores son modelos pre entrenados.
# Fueron los primeros detectores de texturas ópticas de trabajo real que funcionaron bastante bien y muy
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/haar.png)

# utiliza un concepto de ventanas correderas para básicamente deslizar estas imágenes y hace una convolución en la parte
# superior de esta imagen y extrae esas características. Tenemos muchas características de bordes, líneas, rectángulos
# y muchas otras. La combinación de esas características corresponde a un rostro, y esos clasificadores son entrenados
# para identificar las diferentes secuencias.
#
# Probablemente puedo describirlo como que la secuencia de valores que corresponden a la cara de una persona, al
# menos ...lo que sea que esté entrenado. Y para entrenar esto, básicamente sólo necesitas un montón de imágenes
# positivas. Son imágenes donde el objeto está presente e imágenes negativas. Así es como aprende a diferenciar cuando
# una cara está allí y cuando una cara no está allí.
# No va a prendiendo


# Apuntamos la función CascadeClassifier de OpenCV a donde nuestro clasificador (formato de archivo XML) se almacena

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Cargamos nuestra imagen y la convertimos a escala de grises
image = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nuestro clasificador devuelve el ROI de la cara detectada como una tupla
# Almacena la coordenada superior izquierda y la coordenada inferior derecha
"""Así que hemos creado nuestro primer objeto clasificador aquí y ahora que tiene una función llamada CascadeClassifier.
Aquí es donde nos alimentamos en la imagen de entrada. El primer parámetro que podemos establecer scaleFactor, 
así como un minNeighbors. Son parámetros de configuración OPCIONALES que  ajustan la sensibilidad. con ellos se puede 
conseguir más cajas en la cara y el factor de habilidad también. Depende del tipo de imagen y el tipo de cara o el t
amaño de las caras en la imagen. Agarra la cara y extrae en una matriz."""
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No faces found")

# Recorremos la matriz de caras y dibujamos un rectángulo
# sobre cada cara en faces
for (x, y, w, h) in faces:
    # Puntos x e y, asi como el ancho (hacia la izq) y el alto (hacia abajo), para poder calcular el rectágulo
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)  # el último es el grosor
imshow('Face Detection', image)

# ## **Detección simple de ojos y caras usando clasificadores Haarcascade**
import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

img = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No Face Found")

for (x, y, w, h) in faces:
    # Está recortando la cara y luego lo está haciendo de manera similar para la imagen en color también.
    # Así que podemos probar ya sea en el color o gris.
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_classifier.detectMultiScale(roi_gray, 1.2, 3)  # detectamos los ojos
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

imshow('Eye & Face Detection', img)

"""
# ## **Usando los fragmentos de código de Colab accedamos a la webcam para una entrada**
# Nota: Requiere que tu ordenador tenga webcam
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


# In[ ]:


from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))

  # Mostrar la imagen que se acaba de tomar.
  display(Image(filename))
except Exception as err:
    # Se lanzarán errores si el usuario no tiene webcam o si no
    # concedido permiso a la página para acceder a ella.
  print(str(err))


# In[ ]:


import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

img = cv2.imread('photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

imshow('Eye & Face Detection',img)
"""

# Use su webcam para hacer la detección de caras y ojos en directo
# Esto sólo funciona en una máquina local, no funcionará en Colab

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')


def face_detector(img, size=0.5):
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    roi_color = cv2.flip(roi_color, 1)
    return roi_color


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

#################################################
# 17 **Detección de vehículos y peatones** ######
#################################################

# # **Detección de vehículos y peatones**
#
# ####**En esta lección aprenderemos:**
# 1. Usar un clasificador Haarcascade para detectar Peatones
# 2. Usar nuestros clasificadores Haarcascade en vídeos
# 3. Usar un clasificador Haarcascade para detectar Vehículos o coches


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
import IPython


# Definir nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Descarga y descomprime nuestros vídeos y clasificadores Haarcascade
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/videos.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip')
get_ipython().system('unzip -qq haarcascades.zip')
get_ipython().system('unzip -qq videos.zip')
'''

# #### **Pruebas con un solo fotograma de nuestro vídeo**
# Creamos nuestro objeto capturador de vídeo
cap = cv2.VideoCapture('videos/walking.mp4')

# Lectura del primer fotograma
body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# Lectura del primer fotograma
ret, frame = cap.read()

# Ret es True si se ha leído correctamente
if ret:

    # Escala de grises de nuestra imagen para un procesamiento más rápido
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasa la imagen a nuestro clasificador de cuerpos
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extraer las cajas delimitadoras de los cuerpos identificados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Liberar nuestra captura de vídeo
cap.release()
imshow("Pedestrian Detector", frame)

# #### **Prueba en nuestro clip de 15 segundos**
# **NOTA**: Tarda alrededor de 1 minuto en ejecutarse.
# Usamos cv2.VideoWriter para guardar la salida como un archivo AVI. #
# ```cv2.VideoWriter(video_output.avi, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width, height))````
# Los formatos pueden ser:
# - 'M','J','P','G' o MJPG
# - MP4V
# - X264
# - avc1


# Creamos nuestro objeto capturador de vídeo
cap = cv2.VideoCapture('videos/walking.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'walking_output.avi'.
# out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
out = cv2.VideoWriter('walking_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
body_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# Bucle una vez que el vídeo se ha cargado correctamente
while (True):

    ret, frame = cap.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pasar frame a nuestro clasificador de cuerpos
        bodies = body_detector.detectMultiScale(gray, 1.2, 3)

        # Extraer las cajas delimitadoras de los cuerpos identificados
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Escribe el fotograma en el archivo 'output.avi ( + mp4)
        out.write(frame)
    else:
        break

cap.release()
out.release()

# ## **Reproducir Video dentro de Colab**
# Pasos
# 1. Convertir el archivo AVI a MP4 usando FFMPEG
# 2. Cargar los plugins HTML en IPython
# 3. Mostrar nuestro reproductor de vídeo HTML


# Convertir el vídeo y mostrarlo en HTML
# IPython.get_ipython().system('ffmpeg -i /walking_output.avi walking_output.mp4 -y')


from IPython.display import HTML
from base64 import b64encode

mp4 = open('walking_output.mp4', 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# #### **Detección de vehículos en una sola imagen**

# In[ ]:
# Creamos nuestro objeto de captura de vídeo
cap = cv2.VideoCapture('videos/cars.mp4')

# Cargar nuestro clasificador de vehículos
vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Leer primer fotograma
ret, frame = cap.read()

# Ret es True si se ha leído correctamente
if ret:

    # Escala de grises de nuestra imagen para un procesamiento más rápido
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasa la imagen a nuestro clasificador de carrocerías
    vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

    # Extraer las cajas delimitadoras de los cuerpos identificados
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Liberar nuestra captura de vídeo
cap.release()
imshow("Vehicle Detector", frame)

# #### **Prueba en nuestro clip de 15 segundos**

# In[ ]:


# Crear nuestro objeto de captura de vídeo
cap = cv2.VideoCapture('videos/cars.mp4')

#  Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Define el codec y crea el objeto VideoWriter.La salida se almacena en el archivo 'outpy.avi'.
# out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
out = cv2.VideoWriter('cars_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))

vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Bucle una vez que el vídeo se ha cargado correctamente
while (True):

    ret, frame = cap.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pasar frame a nuestro clasificador de carrocerías
        vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

        # Extraer las cajas delimitadoras de los cuerpos identificados
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Escribe el fotograma en el archivo 'output.avi
        out.write(frame)
    else:
        break

cap.release()
out.release()

# Convertir el vídeo y mostrarlo en HTML
# no funione en ubuntu la conversión añadida salida en
# IPython.get_ipython().system('ffmpeg -i /content/cars_output.avi cars_output.mp4 -y')
#

from IPython.display import HTML
from base64 import b64encode

mp4 = open('cars_output.mp4', 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

# no estamos mostrando la salida pero no falla
HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# !/usr/bin/env python
# coding: utf-8
#################################################
# 18 Transformaciones de perspectiva ######
#################################################
# a transformación de perspectiva es una forma en que podemos traducir una señal de imágenes de imagen para convertirla
# en una diferente.
# ####**En esta lección aprenderemos:**
# 1. 1. Usar getPerspectiveTransform de OpenCV
# 2. Usar findContours para obtener esquinas y automatizar la transformación de perspectiva


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Descargar y descomprimir nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

image = cv2.imread('images/scan.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarizamos la imagen
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use una copia de su imagen, por ejemplo edged.copy(), ya que findContours altera la imagen
# sacmos los contornos externos ( RETR_EXTERNAL)
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibuja todos los contornos, ten en cuenta que esto sobrescribe la imagen de entrada (operación inplace)
# Usa '-1' como 3er parámetro para dibujar todo
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))  # Number of Contours found = 54

### **Approxiamamos nuestro contorno anterior a sólo 4 puntos usando approxPolyDP** ( visto en 12)
# Ordenar los contornos de mayor a menor por área (no necesario pero acelera el calculo) además como hay ruido en la
# imagen, la razón por la que estamos ordenando por área en primer lugar sabemos que el área mas grande es el que
# queremos, Es el control más grande, porque los otros van a ser muy pequeños.
#
# Son como píxeles son sólo grupos de píxeles


sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# bucle sobre los contornos
for cnt in sorted_contours:
    #  Aproximación de cada contorno calculando el perímetro y multipicandole el accuracy recomendando (OCV)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

    if len(approx) == 4:  # paramos cuando hayamos encontrado esos 4 puntos
        break

# Nuestras coordenadas x, y de las cuatro esquinas
print("Nuestros 4 puntos de esquina son:")
print(approx)
''''
Nuestros 4 puntos de esquina son:
[[[326  15]]

 [[ 83 617]]

 [[531 779]]

 [[697 211]]]'''

### **Usamos getPerspectiveTransform y warpPerspective para crear nuestra vista de arriba abajo**
#
# Nota: Hemos igualado manualmente el orden de los puntos

# El orden obtenido aquí es arriba a la izquierda, abajo a la izquierda, abajo a la derecha, arriba a la derecha

# acabamos de convertir el tipo de datos aquí porque lo necesitamos como float32
inputPts = np.float32(approx)

# estamos especificando que punto de salida queremos, donde queremos que esté de izq a derecha arriba a abajo
outputPts = np.float32([[0, 0],
                        [0, 800],
                        [500, 800],
                        [500, 0]])

# Obtenemos nuestra matriz de transformación, M
M = cv2.getPerspectiveTransform(inputPts, outputPts)

# Aplica la matriz de transformación M usando Warp Perspective
dst = cv2.warpPerspective(image, M, (500, 800))

imshow("Perspective", dst)

# ### **Ejercicio**
# 1. Ordenar los puntos en ```approx`` ordenando desde arriba a la izquierda en el sentido de las agujas del reloj
# (es decir, arriba a la izquierda, arriba a la derecha, abajo a la izquierda, abajo a la derecha)
# 2. 2. Obtener la relación de aspecto inicial del contorno y ajustar el Warp final para que salga en esa relación de
# aspecto y orientación.

# !/usr/bin/env python
# coding: utf-8
#################################################
# 19 Transformaciones de perspectiva ######
#################################################
# ####**In this lesson we'll learn:**
# 1. Visualizar las representaciones del histograma RGB de las imágenes
# 2. Utilizar K-Means Clustering para obtener los colores dominantes y sus proporciones en las imágenes.
# k-means -> agrupamiento para la causa dominante de una imagen, así que abra ese cuaderno y desplácese hacia arriba.
# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
# un histograma es básicamente un gráfico, un diagrama de barras o un grafico de líneas, Yy un histograma nos da
# básicamente la distribución de algo. Entonces, en el caso de una imagen que vamos a dar, vamos a pasar por una
# distribución de los colores.
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
def imshow(title="Image", image=None, size=8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Descargar y descomprimir nuestras imágenes y clasificadores Haarcascade
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''

# In[ ]:


image = cv2.imread('images/input.jpg')
imshow("Input", image)

# histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

# Trazamos un histograma, la función ravel() aplana nuestra matriz de imágenes, Así que lo tenemos como una gran matriz
# unidimensional
''' 
ORIGINAL
[[ 3  6 11]
  [ 3  6 11]
  [ 2  5 10]
  ...
  [18 23 38]
  [18 23 38]
  [19 24 39]]]

  tras ravel() 
  [12 18 31 ... 19 24 39]'''
print(image)
print(image.ravel())

plt.hist(image.ravel(), 256, [0, 256])  # imagen aplanada, cantidad de contenedores que queremos, el rango
plt.show()  # el básico aplanado muestra el brillo de una imagen en el que el eje vertical es el número de pixeles y
# el horizontal el rango de brillo si se ve un pico al principio significa que hay muchos pixeles oscuros y al final
# claros

# **cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])**
# Usamos la función para calcular el histograma de una imagen
# - **images** : es la imagen de origen de tipo uint8 o float32. Debe darse entre corchetes, es decir, "[img]".
# - **channels** : también se indica entre corchetes. Es el índice del canal para el que calculamos el histograma.
#                  Por ejemplo, si la entrada es una imagen en escala de grises, su valor es [0]. Para una imagen en
#                  color, puede pasar [0], [1] o [2] para calcular el histograma del canal azul, verde o rojo
#                  respectivamente.
# - **mask** : imagen de máscara. Para encontrar el histograma de la imagen completa, se le da como "Ninguno". Pero si
#              desea encontrar el histograma de una región particular de la imagen, tiene que crear una imagen de
#              máscara para eso y darle como máscara. (Mostraré un ejemplo más adelante).
# - **histSize** : esto representa nuestro recuento BIN. Necesita ser dado entre corchetes. Para escala completa,
#                   pasamos [256].
# - **ranges** : este es nuestro RANGO. Normalmente es [0,256].


# Visualización de canales de color separados, las etiquetas de cada color
color = ('b', 'g', 'r')

# Ahora separamos los colores y trazamos cada uno en el Histograma para ver la distribución en el histograma por color
for i, col in enumerate(color):
    # para cada canal ( que contienen el color) de la imagen calculamos su histograma
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color=col)  # color = col para especificar las etiquetas ver cada color de su color
    plt.xlim([0, 256])  # establecemos los límites

plt.show()

# hacemos lo mismo con otra imagen para anailizar su distribución de colores
image = cv2.imread('images/tobago.jpg')
imshow("Input", image)

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Trazamos un histograma, ravel() aplana nuestra matriz de imágenes
plt.hist(image.ravel(), 256, [0, 256]);
plt.show()

# Visualización de canales de color separados
color = ('b', 'g', 'r')

# Ahora separamos los colores y trazamos cada uno en el Histograma
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color=col)
    plt.xlim([0, 256])

plt.show()


# ## **K-Means Clustering para obtener los colores dominantes en una imagen**
# k-means es básicamente un algoritmo de agrupamiento que agrupa píxeles de valor similar.
def centroidHistogram(clt):
    # Crea un histrograma para los clusters basado en los píxeles de cada cluster.
    # Obtener las etiquetas de cada cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Crear nuestro histograma
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # Normalizar el histograma, para que sume uno
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plotColors(hist, centroids):  # nos da la distribución de los colores

    # Crear nuestro gráfico de barras en blanco
    bar = np.zeros((100, 500, 3), dtype="uint8")

    x_start = 0
    # iterar sobre el porcentaje y el color dominante de cada cluster
    for (percent, color) in zip(hist, centroids):
        # trazar el porcentaje relativo de cada cluster
        end = x_start + (percent * 500)
        cv2.rectangle(bar, (int(x_start), 0), (int(end), 100), color.astype("uint8").tolist(), -1)
        x_start = end
    return bar


from sklearn.cluster import KMeans

image = cv2.imread('images/tobago.jpg')
imshow("Input", image)

# Transformamos nuestra imagen en una lista de píxeles RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)  # (1194, 1936, 3)
# remodelamos la imagen porque necesitamos que tenga un formato específico para el CEO de la empresa
image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)  # (2311584, 3) hemos pasado de una imagen tridimensional a una imagen bidimensional

# vamos a crear 5 grupos
number_of_clusters = 5
# ejecutamos el modelo de agrupamiento K
clt = KMeans(number_of_clusters)

# Así que simplemente hacemos el ajuste de puntos de K mientras creamos un sello.
# El objeto clt, que es una clave, significa objeto de agrupación.
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist, clt.cluster_centers_)

# mostrar nuestro color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

# ### **Probemos con otra imagen**

from sklearn.cluster import KMeans

image = cv2.imread('images/Volleyball.jpeg')
imshow("Input", image)

# Transformamos nuestra imagen en una lista de píxeles RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))

number_of_clusters = 3
clt = KMeans(number_of_clusters)
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist, clt.cluster_centers_)

# muestra nuestro color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


#!/usr/bin/env python
# coding: utf-8
#####################################
# 20 Comparación de imágenes** ######
#####################################

# 1. Comparar imágenes utilizando el error cuadrático medio (MSE)
# 2. Comparar imágenes usando similitud estructural
# La diferencia entre las imágenes es bastante importante y tiene muchos casos de uso.
# Uno de ellos, sencillo de entender, es la detección de movimiento. Puede usar fácilmente cambios en las imágenes
# para detectar cuándo ha habido movimiento.


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')

get_ipython().system('unzip -qq images.zip')'''


# #### **Error cuadrático medio (MSE)**
#
# El MSE entre las dos imágenes es la suma de la diferencia al cuadrado entre las dos imágenes. Esto se puede
# implementar fácilmente con numpy.
# Cuanto menor sea el MSE más parecidas son las imágenes.


def mse(image1, image2):
    # Las imágenes deben tener la misma dimensión
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


# #### **Vamos a obtener 3 imágenes**
#
# 1. Fuegos artificiales1
# 2. Fuegos artificiales1 con brillo mejorado
# 3. Fuegos artificiales2


fireworks1 = cv2.imread('images/fireworks.jpeg')
fireworks2 = cv2.imread('images/fireworks2.jpeg')

# aumentamos el brillo de una imagen para su comparación
M = np.ones(fireworks1.shape, dtype = "uint8") * 100
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)


def compare(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print('MSE = {:.2f}'.format(mse(image1, image2)))
    #  la función de structural_similarity es una función que sesga las métricas, lo que significa que es un algoritmo
    #  complicado que nos da básicamente estas similitudes estructurales basadas en relaciones de vecindad entre
    #  matrices para decir la diferencia. 1.0 misma imagen, caunto mas baja mas diferencias
    print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


# Cuando son iguales
compare(fireworks1, fireworks1)


compare(fireworks1, fireworks2)

compare(fireworks1, fireworks1b)

compare(fireworks2, fireworks1b)


###############################
# 21 Filtrado de colores ######
###############################
# 1. Cómo utilizar el espacio de color HSV para filtrar por color
#
# #### **Recordar el Espacio de Color HSV** ( visto en 03)
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

'''# Descargar y descomprimir nuestras imágenes y clasificadores Haarcascade
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')
'''

# Vamos aintentar quitar el camion y la tierra de la imagen dejando solo el cielo

image = cv2.imread('images/truck.jpg')

# Entonces, para hacer eso necesitamos definir un rango superior e inferior.
# definir el rango de color AZUL en HSV, en la imagen de arriba se ve que el azul va del tono 90 al 135
lower = np.array([90,0,0]) # tono , saturación , valor
upper = np.array([135,255,255])

# Convertir la imagen de RBG/BGR a HSV para poder filtrar fácilmente
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Usar inRange para capturar sólo los valores entre inferior y superior, es decir5 crear una máscara, un umbral binario
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


######################################################################################
# 22 Algoritmo Watershed para la segmentación de imágenes basada en marcadores ######
######################################################################################
# 1. Cómo utilizar el algoritmo Watershed para la segmentación de imágenes basada en marcadores



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

'''get_ipython().system('wget https://docs.opencv.org/3.4/water_coins.jpg')'''

# **Teoría de Algoritmos de Cuencas Hidrográficas**
# Cualquier imagen en escala de grises puede ser vista como una superficie topográfica donde la alta intensidad denota
# picos y colinas mientras que la baja intensidad denota valles.

# Este algoritmo utiliza esa analogía y comienza a llenar esos puntos bajos (valles) con una etiqueta de color diferente
# (aka nuestra agua).
#
# A medida que el agua sube, dependiendo de los picos (gradientes) cercanos, el agua de diferentes valles, obviamente
# con diferentes colores comenzará a fusionarse. Para evitar eso, construyes barreras en los lugares donde el agua se
# fusiona. Continúa el trabajo de llenar agua y construir barreras hasta que todos los picos estén bajo el agua.
#
# Las barreras que has creado te dan el resultado de la segmentación. Esta es la "filosofía" detrás de la cuenca.
# Puedes visitar la página web [CMM webpage](http://cmm.ensmp.fr/~beucher/wtshed.html) sobre la cuenca hidrográfica
# para entenderla con la ayuda de algunas animaciones.
# Su enfoque, sin embargo, le da un resultado oversegmented debido al ruido o cualquier otra irregularidad en la imagen.

# MI EXPLICACION
# En resumen la transformación de cuencas hidrográficas se basa en la idea de que, sobre una imagen en escala de grises
# y tomando los cambios de tonalidad de dicha escala, podemos, simulando el negro como el mínimo y el blanco como el
# máximo, indundar desde sus mínimos la imagen con agua evitando la fusión del agua en zonas distinas con los tonos
# blancos, creando así una división o segmentación en la imagen. Debido a problemas que conlleva el ruido y cambios de
# tono en imágenes reales, se establecen marcadores antes de la "inundación" para que se realiza la segmentación de
# forma correcta a raíz de lo deseado de la imagen


#  OpenCV implementó un algoritmo de cuenca basado en marcadores donde se especifica cuáles son todos los
# puntos que se van a fusionar y cuáles no. Da diferentes etiquetas para los objetos que conocemos.
# Etiquetamos la región del primer plano u objeto con un color (o intensidad),
# etiquetamos del fondo o no objeto con otro color y finalmente la región de
# que desconocemos, la etiquetamos con 0.
# Ese es nuestro marcador. A continuación, aplicar el algoritmo el marcador se actualizará con las etiquetas que le
# dimos, y los límites de los objetos tendrán un valor de -1.

#


# Cargar imagen
img = cv2.imread('images/water_coins.jpg')
imshow("Original image", img)

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbral usando OTSU (visto en 09)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

imshow("Thresholded", thresh)


## ## **Eliminar las máscaras de retoque**


# eliminación de ruido
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

# área de fondo, estamos creando los marcadores sobre las monedas dilatantdo  (visto en 10, es decir agregando
# píxeles a los límites de los objetos, el fondo en este caso en una imagen) 3 veces la imagen
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Encontrar el área de primer plano, mediante la función cv2.distanceTransform y la binarización de la imagen resultante
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Encontrar región desconocida restando el fondo al primer plano
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

imshow("SureFG", sure_fg)
imshow("SureBG", sure_bg)
imshow("unknown", unknown)


# Etiquetado de marcadores
# connectedComponents determina la conectividad de regiones tipo blob en una imagen binaria.
ret, markers = cv2.connectedComponents(sure_fg)

# Añadir uno a todas las etiquetas para que el fondo no sea 0, sino 1
markers = markers+1

# Ahora, marca la región de unknown con cero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

imshow("img", img)



#############################################
#23 Substracción de fondo y primer plano ######
#############################################
# 1. 1. Sustracción de fondo con algoritmo de segmentación de fondo/primer plano basado en mezcla gaussiana.
# 2. Modelo de mezcla gaussiana adaptativo mejorado para sustracción de fondo

## La sustracción de fondo (BS) es una técnica común y ampliamente utilizada para generar una máscara de primer plano
# (es decir, una imagen binaria que contiene los píxeles pertenecientes a los objetos en movimiento de la escena)
# mediante el uso de cámaras estáticas.
#
# Como su nombre indica, la BS calcula la máscara de primer plano realizando una sustracción entre el fotograma actual
# y un modelo de fondo, que contiene la parte estática de la escena o, más en general, todo lo que puede considerarse
#  como fondo dadas las características de la escena observada.
#
# ![](https://docs.opencv.org/3.4/Background_Subtraction_Tutorial_Scheme.png)
#
# El modelado del fondo consta de dos pasos principales:
# 1. 1. Inicialización del fondo;
# 2. Actualización del fondo.
#
# En el primer paso se calcula un modelo inicial del fondo, mientras que en el segundo se actualiza dicho modelo para
# adaptarse a posibles cambios en la escena.



# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking_short_clip.mp4')
'''
# **¿Qué es la sustracción de fondo?**

# La sustracción de fondo es una técnica de visión por ordenador en la que buscamos aislar el fondo del primer plano
# 'en movimiento'. Consideremos los vehículos que atraviesan una carretera o las personas que caminan por una acera.
#
# Suena sencillo en teoría (es decir, basta con mantener los píxeles fijos y eliminar los que cambian). Sin embargo,
# cosas como cambios en las condiciones de iluminación, sombras, etc. pueden complicar las cosas.
#
# Se han introducido varios algoritmos para este propósito. A continuación veremos dos algoritmos del módulo **bgsegm**.



# ***Algoritmo de segmentación de fondo/primer plano basado en mezclas gaussianas.
#
# En este trabajo, proponemos un método de sustracción de fondo (BGS) basado en los modelos de mezcla gaussiana
# utilizando información de color y profundidad. Para combinar la información de color y profundidad, utilizamos el
# modelo probabilístico basado en la distribución gaussiana. En particular, nos centramos en resolver el problema del
# camuflaje de color y la eliminación de ruido en profundidad. Para evaluar nuestro método, hemos creado un nuevo
# conjunto de datos que contiene situaciones normales, de camuflaje de color y de camuflaje de profundidad. Los
# archivos del conjunto de datos constan de secuencias de imágenes en color, en profundidad y de la verdad sobre el
# terreno. Con estos archivos, comparamos el algoritmo propuesto con las técnicas convencionales de BGS basadas en el
# color en términos de precisión, recuperación y medida F. El resultado fue que nuestro método demostró ser más preciso
# que los algoritmos convencionales. Como resultado, nuestro método mostró el mejor rendimiento. Así pues, esta técnica
# ayudará a detectar de forma robusta regiones de interés como preprocesamiento en etapas de procesamiento de imágenes
# de alto nivel.
#
#
# Enlace al artículo - https://www.researchgate.net/publication/283026260_Background_subtraction_based_on_Gaussian_mixture_models_using_color_and_depth_information



cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y la anchura del fotograma (se requiere que sea un interger)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('videos/walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()

# Bucle una vez que el vídeo se ha cargado correctamente
while True:

    ret, frame = cap.read()

    if ret:
        #  Aplicar el sustractor de fondo para obtener nuestra máscara de primer plano
        foreground_mask = foreground_background.apply(frame)
        out.write(foreground_mask)
        imshow("Foreground Mask", foreground_mask)
    else:
        break

cap.release()
out.release()


# ### **Probemos el modelo de mezcla gausiano adaptativo mejorado para la sustracción de fondo**
#
# La sustracción de fondo es una tarea común de visión por ordenador. Analizamos el enfoque habitual a nivel de píxel.
# Desarrollamos un algoritmo adaptativo eficiente utilizando la densidad de probabilidad de la mezcla gaussiana.
# Se utilizan ecuaciones recursivas para actualizar constantemente los parámetros y también para seleccionar
# simultáneamente el número apropiado de componentes para cada píxel.
# https://www.researchgate.net/publication/4090386_Improved_Adaptive_Gaussian_Mixture_Model_for_Background_Subtraction


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output_AGMM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

# Bucle una vez que el vídeo se ha cargado correctamente
while True:

    ret, frame = cap.read()
    if ret:
        # Aplicar el sustractor de fondo para obtener nuestra máscara de primer plano
        foreground_mask = foreground_background.apply(frame)
        out.write(foreground_mask)
        imshow("Foreground Mask", foreground_mask)
    else:
      break

cap.release()
out.release()


# ## **Substracción de primer plano**


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
ret, frame = cap.read()

# Crear un array numpy float con los valores de los fotogramas
average = np.float32(frame)

while True:
    # Obtener frame
    ret, frame = cap.read()

    if ret:
        # accumulateWeighted nos permite básicamente, almacenar valores del frame pasado.
        # 0.01 es el peso de la imagen, juega para ver como cambia
        cv2.accumulateWeighted(frame, average, 0.01)
        # Posteriomente con esos valores almacenados podemos obtener con convertScaleAbs el promedio, que es lo que
        # especificamos aquí, obteniendo el valor promedio del marco Es una forma de hacer un seguimiento de lo que es
        # el fondo.
        # Escala, calcula valores absolutos, y convierte el resultado a 8-bit, obtenemos así matemáticamente el fondo
        background = cv2.convertScaleAbs(average)

        imshow('Input', frame)
        imshow('Disapearing Background', background)
        out.write(background)
        # No es tan evidente en estas imágenes. Sin embargo, se acumula con el tiempo, por lo que cuanto más tiempo lo
        # dejemos, más se acumulará (no es el mejor método).

    else:
      break

cap.release()
out.release()




cv2.imshow(background)


### **Background Substraction KKN** ( el mejor de este documento)
#
# Los parámetros si desea desviarse de la configuración predeterminada:
#
# - **history** es el número de fotogramas utilizados para construir el modelo estadístico del fondo. Cuanto menor sea
#               el valor, más rápido serán tenidos en cuenta por el modelo los cambios en el fondo y, por tanto, serán
#               considerados como fondo. Y viceversa.
# - **dist2Threshold** es un umbral para definir si un píxel es diferente del fondo o no. Cuanto menor sea el valor,
#                      más sensible será la detección del movimiento. Y viceversa.
# ** detectShadows **: Si se establece en true, las sombras se mostrarán en gris en la máscara generada.(Ejemplo abajo)
#
# https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Obtenemos la estructura del kernel o o matriz de árboles con getStructuringElement usando MORPH_ELLIPSE
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# inicializamos el substractor de fondo
fgbg = cv2.createBackgroundSubtractorKNN()

while (1):
    ret, frame = cap.read()

    if ret:

        # aplicamos el algoritmo al frame mediante el método apply, obteniendo el 1º plano
        fgmask = fgbg.apply(frame)

        # luego debemos aplicar el primer plano la morfología x, que es, usar la función con el kernel que definimos y
        # obtener la salida
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        imshow('frame', fgmask)
    else:
      break

cap.release()
out.release()

# !/usr/bin/env python
# coding: utf-8

################################################################
# 24 Seguimiento del movimiento con Mean Shift y CAMSHIFT ######
################################################################
# Seguimiento: Imagina que tienes una persona en movimiento o un vehículo en movimiento en un video de CCTV y quieres
# enfocarte en esa persona. Dibujas una caja y la mueves sobre la persona mientras él, ella, el coche, etc se mueve en
# el video. Eso es lo que es el seguimiento

# ####**En esta lección aprenderemos dos Algoritmos de Seguimiento de Objetos:**
# 1. Cómo usar el algoritmo Mean Shift en OpenCV
# 2. Usar CAMSHIFT en OpenCV

# In[1]:


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# from google.colab.patches import cv2_imshow

# Define nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''get_ipython().system('wget https://github.com/makelove/OpenCV-Python-Tutorial/raw/master/data/slow.flv')
'''

# ## **Rastreo de Objetos Meanshif**
#
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_basics.jpg)
#
# La intuición detrás del meanhift es simple. Considera que tienes un conjunto de puntos. (Puede ser una distribución
# de píxeles como la retroproyección del histograma). Se le da una pequeña ventana (puede ser un círculo) y usted tiene
# que mover esa ventana a la zona de máxima densidad de píxeles (o el número máximo de puntos). Se ilustra en la imagen
# simple dada a continuación:
#
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_face.gif)
#
# El desplazamiento medio es un algoritmo de escalada que consiste en desplazar iterativamente este núcleo a una región
# de mayor densidad hasta la convergencia. Cada desplazamiento se define por un vector de desplazamiento medio. El
# vector de desplazamiento medio siempre apunta hacia la dirección del máximo incremento en la densidad.
# ![](https://upload.wikimedia.org/wikipedia/commons/b/bd/Meanshiftred.gif)
#
# Lea el artículo aquí - https://ieeexplore.ieee.org/document/732882
#
# Fuente de la animación - https://fr.wikipedia.org/wiki/Camshift

#  Es decir, estableces una ventana y la mueves iterativamente a la parte mś intensa de la trama, considerando
#  el histograma, esto es, intensidades de color en el cuadro delimitador inicial que establecimos. Acabamos de
#  establecer algunos criterios para mirar, moverse y buscar el siguiente punto más brillante alrededor de esa imagen.
# Y lo mueves iterativamente hacia el área más densa de la intensidad pudiendo utilizar intensidad de rojo,  azul, verde
# .... de saturación y espacio de color HSV.


cap = cv2.VideoCapture('videos/data_slow.flv')

# toma el primer fotograma del video
ret, frame = cap.read()

# Obtener la altura y anchura del fotograma (se requiere que sea un entero)
width = int(cap.get(3))
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('car_tracking_mean_shift.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

# configurar la ubicación inicial de la ventana
r, h, c, w = 250, 90, 400, 125  # simplemente codificar los valores
track_window = (c, r, w, h)

# establecer el ROI para el seguimiento
roi = frame[r:r + h,
      c:c + w]  # establecemos en las coordenadas de la imagen el roi como un rectángulo con los valores conf

hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # pasamos el frame a HSV

# Usar inRange (visto en 21) para capturar sólo los valores entre inferior y superior, es decir5 crear una máscara, un
# umbral binario en la imagen, el blanco sería un SI, entra en la máscara y el negro un NO
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# calcula el histograma del roi con la mascara establecida
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# normaliza el resultado, para asegurarse de que, de cuadro a cuadro, sea consistente en el mismo rango.
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Establecer los criterios de terminación, ya sea 10 iteración o mover por lo menos 1 pt,
# para dejar de rastrear en ese punto. Entonces, dejamos de atender donde no está el movimiento, al menos por un
# punto, eso significa que dejamos de rastrear en ese punto.
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # pasamos el frame a HSV

        # calculamos la retroproyección para el cálculo del histograma.
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # aplicar desplazamiento medio para obtener la nueva ubicación con la imagen, la ubicación de la imagen actual
        # y los criterios de terminación establecidos
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Dibújalo en la imagen
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        out.write(img2)
        # El cuadrado 'pequeño' es el establecido y el otro va buscando las áreas más brillantes de la imagen
        # imshow('Tracking', img2)

    else:
        break

cap.release()
out.release()

# ## **Camshift en OpenCV**
# Es casi igual que meanshift, pero devuelve un rectángulo rotado (que es nuestro resultado) y parámetros de caja
# (que se pasan como ventana de búsqueda en la siguiente iteración).
# Por lo tanto, es una forma más efectiva de seguimiento.
# ![](https://upload.wikimedia.org/wikipedia/commons/8/86/CamshiftStillImage.gif)
#
# Lea el artículo aquí - https://ieeexplore.ieee.org/document/732882
#
# Fuente de animación - https://fr.wikipedia.org/wiki/Camshift


cap = cv2.VideoCapture('videos/data_slow.flv')

# toma el primer fotograma del video
ret, frame = cap.read()

# Obtener la altura y anchura del fotograma (se requiere que sea un entero)
width = int(cap.get(3))
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('car_tracking_cam_shift.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

# configurar la ubicación inicial de la ventana
r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
track_window = (c, r, w, h)

# establecer el ROI para el seguimiento
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Establecer los criterios de terminación, ya sea 10 iteración o mover por lo menos 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # aplicar desplazamiento medio para obtener la nueva ubicación
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Dibújalo en la imagen diferente al anterior porque en lugar de dibujar usando el rectángulo, tenemos que
        # obtener dos puntos y dibuje el polígono de línea para el rectángulo rotado.
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        out.write(img2)
        # imshow('img2',img2)

    else:
        break

cap.release()
out.release()

# !/usr/bin/env python
# coding: utf-8

############################################
# 25 Object Tracking with Optical Flow######
############################################
# 1. Cómo usar Optical Flow en OpenCV
# 2. Luego usar Dense Optical Flow

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


# from google.colab.patches import cv2_imshow

# Define our imshow function
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking_short_clip.mp4')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking.avi')
'''

# ## **El algoritmo de flujo óptico Lucas-Kanade**
#
# El flujo óptico es el patrón de movimiento aparente de los objetos de la imagen entre dos fotogramas consecutivos
# causado por el movimiento del objeto o de la cámara. Se trata de un campo vectorial 2D en el que cada vector es un
# vector de desplazamiento que muestra el movimiento de los puntos del primer fotograma al segundo. Considere la s
# Siguiente imagen (Imagen cortesía: Wikipedia article on Optical Flow).
#
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Optical_flow_example_v2.png/440px-Optical_flow_example_v2.png)
#
# Muestra una bola moviéndose en 5 fotogramas consecutivos. La flecha muestra su vector de desplazamiento. El flujo
# óptico tiene muchas aplicaciones en áreas como:
#
# - Estructura a partir del movimiento
# - Compresión de vídeo
# - Estabilización de vídeo
#
# El flujo óptico funciona en varios supuestos:
#
# - Las intensidades de los píxeles de un objeto no cambian entre fotogramas consecutivos.
# - Los píxeles vecinos tienen un movimiento similar.
#
# Más información - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

# ES DECIR Busca el flujo aparente, el movimiento o la dirección de un objeto que se mueve en una imagen y entre
# fotogramas consecutivos. luego, rastrea eso con el campo vectorial 2D, donde el vector de características representa
# el desplazamiento el movimiento de puntos de fotograma a fotograma.


# Cargar flujo de vídeo, clip corto
cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Cargar flujo de vídeo, clip largo
# cap = cv2.VideoCapture('videos/walking.avi')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3))
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('optical_flow_walking.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

# Establecer parámetros para la detección de esquinas ShiTomasi
# ES uno de los métodos que podemos usar en el flujo óptico para identificar los puntos que necesitamos rastrear.
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parámetros para el flujo óptico lucas kanade
lucas_kanade_params = dict(winSize=(15, 15),  # tamaño de la ventana
                           maxLevel=2,  # indica la cantidad de pirámides
                           #  una herramienta de escala que se abre al usuario para que podamos ver dos, podemos hacer
                           #  que se vean diferentes habilidades y más robusto a los objetos más pequeños o más grandes
                           #  que queremos rastrear.
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Crear algunos colores aleatorios
# Usados para crear nuestras estelas para el movimiento del objeto en la imagen
color = np.random.randint(0, 255, (100, 3))

# Toma el primer fotograma y encuentra las esquinas en él
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Encontrar las esquinas iniciales para establecer nuestro movimiento
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Crear una imagen de máscara para dibujar con las dimensiones del frame
mask = np.zeros_like(prev_frame)

while (1):
    ret, frame = cap.read()

    if ret == True:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calcular el flujo óptico
        # le pasamos frame en escala de grises anterior, el actual, las esquinas previamente calculadas
        # y los parámetros establecidos anteriormente
        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                               frame_gray,
                                                               prev_corners,
                                                               None,
                                                               **lucas_kanade_params)

        # Seleccionar y almacenar los puntos buenos que queremos usar ( los que tienen el estado 1 o correcto)
        good_new = new_corners[status == 1]
        good_old = prev_corners[status == 1]

        # Dibuja las pistas
        try:
            # Zip para hacerlo con una lista o un conjunto como este.
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # los aplanamos para que solo nos de 2 valores
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)  # Jesus fix
                # dibujamos las líneas
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        except Exception as e:
            print(e)
        img = cv2.add(frame, mask)

        # Guardar vídeo
        out.write(img)
        # Mostrar flujo óptico
        # imshow('Optical Flow - Lucas-Kanade',img)

        # Ahora actualiza el fotograma anterior y los puntos anteriores
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1, 1, 2)

    else:
        break

cap.release()
out.release()

# **NOTE** No muestra este ejemplo el vídeo, sino el movimiento sobre un fondo negro
#
# Este código no comprueba cómo de correctos son los siguientes puntos clave. Por lo tanto, incluso si un punto
# desaparece en la imagen, existe la posibilidad de que el flujo óptico encuentre el siguiente punto que se le parezca.
# Así que para un seguimiento robusto, los puntos de esquina deben ser detectados en intervalos particulares.

# Flujo óptico denso
# El método Lucas-Kanade calcula el flujo óptico para un conjunto de características dispersas (en nuestro ejemplo,
# esquinas detectadas usando el algoritmo Shi-Tomasi). OpenCV proporciona otro algoritmo para encontrar el flujo óptico
# denso. Calcula el flujo óptico para todos los puntos del fotograma. Se basa en el algoritmo de Gunner Farneback que
# se explica en "[Two-Frame Motion Estimation Based on Polynomial Expansion]
# (https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)"
# por Gunner Farneback en 2003.


# A continuación se muestra cómo encontrar el flujo óptico denso utilizando el algoritmo anterior.
# Obtenemos una matriz de 2 canales con vectores de flujo óptico, (u,v).
# Encontramos su magnitud y dirección.
# Coloreamos el resultado para una mejor visualización.
#
# - Dirección corresponde al valor Hue de la imagen.
# - Magnitud corresponde al plano Valor. Ver el código a continuación:


# Cargar flujo de vídeo, clip corto
# cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# # Cargar flujo de vídeo, clip largo
cap = cv2.VideoCapture('videos/walking.mp4')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3))
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('dense_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

# Obtener primer fotograma
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:

    # Lectura del archivo de vídeo
    ret, frame2 = cap.read()

    if ret == True:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calcula el flujo óptico denso usando el algoritmo de Gunnar Farneback
        flow = cv2.calcOpticalFlowFarneback(previous_gray, next,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # usa el flujo para calcular la magnitud (velocidad) y el ángulo de movimiento
        # usa estos valores para calcular el color que refleje la velocidad y el ángulo
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * (180 / (np.pi / 2))
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Guardar vídeo
        out.write(final)
        # Mostrar nuestra demo de Dense Optical Flow
        # imshow('Dense Optical Flow', final)

        # Guardar la imagen actual como imagen anterior
        previous_gray = next

    else:
        break

cap.release()
out.release()


############################################
# 26 Simple Rastreo de Objetos por Color######
############################################
# 1. Cómo usar un Filtro de Color HSV para Crear una Máscara y luego Rastrear nuestro Objeto Deseado

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes

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


'''
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/bmwm4.mp4')
'''

# Rastreo de objetos
import cv2
import numpy as np

# Initalizar cámara
# cap = cv2.VideoCapture(0)

# definir rango de color en HSV, establecemos (visto en 21) un filtro para el color amarillo
lower = np.array([20, 50, 90])
upper = np.array([40, 255, 255])

# Crear matriz de puntos vacía, son los puntos que se van a rastrear para que pueda ver una línea.
# Hay una línea histórica de puntos de seguimiento.
points = []

# Obtener el tamaño por defecto de la ventana de la cámara

# Cargar flujo de video, clip largo
cap = cv2.VideoCapture('videos/bmwm4.mp4')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3))
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('bmwm4_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0
radius = 0

# primero filtra y luego introduce los contronos encontrados a raiz de ello en la salida ( visto en 11  y en 12)
# específicamente el controno más grande ( linea 82), es decir El cuadro más grande alrededor de uno de los objetos
# amarillos en la pantalla.
while True:

    # Capturar fotograma webcame
    ret, frame = cap.read()
    if ret:
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Umbral de la imagen HSV para obtener sólo los colores verdes
        mask = cv2.inRange(hsv_img, lower, upper)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        #
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crea una matriz de centros vacía para almacenar el centro de masa del centroide
        center = int(Height / 2), int(Width / 2)

        if len(contours) > 0:

            # Obtener el contorno más grande y su centro
            # obtenga el área, el radio, para un círculo de cierre mínimo para el contorno.
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)  # radius obtentiene el punto y el radio
            M = cv2.moments(c)

            # A veces los contornos pequeños de un punto provocan un error de división por cero
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            except:
                center = int(Height / 2), int(Width / 2)

            # Permitir sólo los contadores que tengan un radio superior a 25 píxeles
            if radius > 25:
                # Dibuja un circulo y deja el ultimo centro creando un rastro
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # Registrar los puntos del centro
            points.append(center)

        # Si el radio es suficientemente grande, usamos 25 píxeles
        # almacenamos todos los puntos aquí y luego dibujamos una línea.
        # Así que esa línea es básicamente el seguimiento histórico.
        if radius > 25:

            # bucle sobre el conjunto de puntos rastreados
            for i in range(1, len(points)):
                try:
                    cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
                except:
                    pass

            # Hacer cero el recuento de fotogramas
            frame_count = 0

        out.write(frame)
        # en el vídeo de ejemplo empieza rasteando una zona que tendra algo de amarillo y cuando
        # encuentra el coche lo sigue, lo pierde y lo vuelve a seguir
    else:
        break

# Libera la cámara y cierra las ventanas abiertas
cap.release()
out.release()

#!/usr/bin/env python
##########################################################################################
# 27 y 28 Detección de puntos de referencia faciales con Dlib e intercambio de caras######
##########################################################################################
# LIBRERIAS
# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import dlib  # librería de machine learning
import numpy as np
from matplotlib import pyplot as plt

# CLASES UTILIZADAS
class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


# FUNCIONES UTILIZADAS EXPLICADAS DESDE 27 y 28 y leyendolas para entender todo el proceso correctamente
def imshow(title = "Image", image = None, size = 10): # Mostrar por pantalla la imagen
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
def annotate_landmarks(im, landmarks):  # Dibuja las marcas de línea que tenemos en la cara.
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,

                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
def get_landmarks(im): # Toma una imagen.
    """
    La función get_landmarks()toma una imagen en forma de matriz numpy y devuelve una matriz de elementos de 68x2, cada
    una de las cuales se corresponde con las coordenadas x, y de un punto de característica particular en la imagen de
    entrada.

    El extractor de características (predictor) requiere un cuadro delimitador aproximado como entrada para el algoritmo
    Esto lo proporciona un detector de rostros tradicional (detector) que devuelve una lista de rectángulos, cada uno de
     los cuales corresponde a un rostro en la imagen.
    """

    rects = detector(im, 1)  # Lo pasa por el detector.

    # resuelve los cuadros delimitadores aqui, pues solo queremos 1
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    # Es donde realmente llamamos a predictor (codemos la imagen , una x en particular , siendo el 1º el único que
    # queremos) lo ejecutamos a través del predictor
    # mediante la lista de comprensión obtenemos las predicciones históricas que obtenemos del predictor. vamos metiendo
    # las coordenadas X e Y de todas esas predicciones históricas.
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


# Es una transformación desde dos puntos. Devuelve la transformación afin para que los puntos se alineen y la
# perspectiva se ajuste correctament
def transformation_from_points(points1, points2):
    """
    ##2. Alineación de caras con un análisis de procrustes
    Así que en este punto tenemos nuestras dos matrices de puntos de referencia, cada fila tiene las coordenadas de un
    rasgo facial en particular (por ejemplo, la fila 30 da las coordenadas de la punta de la nariz). Ahora vamos
    a averiguar cómo rotar, trasladar y escalar los puntos del primer vector para que se ajusten lo más posible a los
    puntos del segundo vector, la idea es que la misma transformación se puede usar para superponer la segunda imagen
    sobre la primera.
    """
    # Resolver el problema procrustes restando centroides, escalando por la
    # desviación estándar, y luego usando el SVD para calcular la rotación. Ver
    # lo siguiente para más detalles:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # 1. Convierte las matrices de entrada en flotantes. Esto es necesario para las operaciones que van a seguir.
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # 2. Resta el centroide de cada uno de los conjuntos de puntos. Una vez que se ha encontrado una escala y una
    # rotación óptimas para los conjuntos de puntos resultantes, los centroides c1 y c2 se pueden usar para encontrar
    # la solución completa.
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    # 3. Del mismo modo, divida cada punto establecido por su desviación estándar. Esto elimina el componente de escala
    # del problema.
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    # 4. Calcule la porción de rotación utilizando la Descomposición de valores singulares . Consulte el artículo de
    # wikipedia https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # sobre el problema de Procrustes ortogonal para obtener detalles sobre cómo funciona.
    U, S, Vt = np.linalg.svd(points1.T * points2)

    # La R que buscamos es en realidad la transpuesta de la dada por U * Vt. Esto
    # es porque la formulación anterior asume que la matriz va a la derecha
    # (con vectores fila) mientras que nuestra solución requiere que la matriz vaya a la
    # izquierda (con vectores columna).
    R = (U * Vt).T

    # Devuelve la transformación completa como una matriz de transformación afín
    """Devuelve una transformación afín [s * R | T] tal que:
        suma ||s*R*p1,i + T - p2,i||^2
    se minimiza."""
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])

#Esta es una función de imagen distorsionada donde simplemente eliminamos la imagen, la matriz y la forma, y
# simplemente emita la imagen final en función de esos parámetros.
def warp_im(im, M, dshape): #  asigna la segunda imagen a la primera
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


# Esta es la corrección de color.
def correct_colours(im1, im2, landmarks1):
    """
    El problema es que las diferencias en el tono de la piel y la iluminación entre las dos imágenes provocan una
    discontinuidad alrededor de los bordes de la región superpuesta Tratamos de corregir eso

    Esta función intenta cambiar el color de im2para que coincida con el de im1. Lo hace dividiendo im2por un desenfoque
    gaussiano de im2y luego multiplicando por un desenfoque gaussiano de im1. La idea aquí es la de una corrección de
    color de escala RGB , pero en lugar de un factor de escala constante en toda la imagen, cada píxel tiene su propio
    factor de escala localizado.

    Con este enfoque, las diferencias de iluminación entre las dos imágenes pueden explicarse, hasta cierto punto. Por
    ejemplo, si la imagen 1 está iluminada desde un lado pero la imagen 2 tiene una iluminación uniforme, entonces la i
    magen 2 con el color corregido aparecerá más oscura en el lado no iluminado también.

    Dicho esto, esta es una solución bastante cruda al problema y un kernel gaussiano de tamaño apropiado es clave.
    Demasiado pequeño y los rasgos faciales de la primera imagen aparecerán en la segunda. Demasiado grande y el kernel
    se desvía fuera del área de la cara para que los píxeles se superpongan y se produce una decoloración. Aquí se
    utiliza un núcleo de 0,6 * la distancia pupilar.
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(  # desenfoque gaussiano para asegurarse de que se vea bien.
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Evitar errores de división por cero.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# draw_convex_hull es un casco convexo de dibujo, que nos permite mapear los puntos correctamente en tres interfaces.
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# get_face_mask para obtener la primera masa para que podamos extraer la cara de la imagen para ponerla en la primera
# imagen.
def get_face_mask(im, landmarks):
    """
    Se define una rutina para generar una máscara para una imagen y una matriz de puntos de referencia. Dibuja dos
    polígonos convexos en blanco: uno que rodea el área de los ojos y otro que rodea el área de la nariz y la boca.
    Luego, desvanece el borde de la máscara hacia afuera en 11 píxeles. El calado ayuda a ocultar las discontinuidades
    remanentes

    Dicha máscara facial se genera para ambas imágenes. La máscara de la segunda se transforma en el espacio de
    coordenadas de la imagen 1, usando la misma transformación que en el paso 2.
    """
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

# Obtienes la función simple de puntos de referencia.
def read_im_and_landmarks(image):
    im = image
    im = cv2.resize(im,None,fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


def swappy(image1, image2):
    # 1. Detección de puntos de referencia faciales : get_landmarks
    # 2. Rotar, escalar y traducir la segunda imagen para que se ajuste a la primera: transformation_from_points y warp_im
    #    Luego, el resultado se puede conectar a la cv2.warpAffinefunción de OpenCV para asignar la segunda imagen a la
    #    primera:
    # 3. Ajuste del balance de color en la segunda imagen para que coincida con el de la primera: correct_colours
    # 4 .Fusión de características de la segunda imagen encima de la primera: *_POINTS, draw_convex_hull, get_face_mask
    im1, landmarks1 = read_im_and_landmarks(image1)
    im2, landmarks2 = read_im_and_landmarks(image2)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    """
    Dicha máscara facial se genera para ambas imágenes. La máscara de la segunda se transforma en el espacio de 
    coordenadas de la imagen 1, usando la misma transformación que en el paso 2.
    """
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    # Luego, las máscaras se combinan en una tomando un máximo de elementos. La combinación de ambas máscaras asegura
    # que las características de la imagen 1 estén cubiertas y que las características de la imagen 2 se vean.
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    return image


# 27. Aplicar la detección de puntos de referencia faciales
'''# Descarga y descomprime nuestras imágenes y el modelo Facial landmark
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/shape_predictor_68_face_landmarks.zip')
get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq shape_predictor_68_face_landmarks.zip')'''


# ## **Detección de puntos de referencia faciales**
PREDICTOR_PATH = "modelos/shape_predictor_68_face_landmarks.dat" # poniendo la parte del modelo en esta variable de aquí
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # cargando el predictor que es un objeto predictor de dylib.
# entra lo que ella predice y solo señalamos la parte del modelo.
detector = dlib.get_frontal_face_detector()  # creamos el detector

# usamos las funciones declaradas y explicadas en este fichero
image = cv2.imread('images/Trump.jpg')
imshow('Original', image)
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
imshow('Result', image_with_landmarks)


# Otra imagen
image = cv2.imread('images/Hillary.jpg')
imshow('Original', image)
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
imshow('Result', image_with_landmarks)


# ##
# ## ** 28 Intercambio de caras**
# ## http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
# El proceso se divide en cuatro pasos, organizado en la funcion swappy:


import sys

PREDICTOR_PATH = "modelos/shape_predictor_68_face_landmarks.dat"
# En primer lugar, declaramos algunas variables, que es un camino para el efecto de escala del modelo de predicción.

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11 # y la cantidad, que es básicamente cuánto estamos haciendo las capas de las caras.

'''Tenemos de cero a 68 puntos, cada uno de esos puntos Cada uno de esos puntos tiene algunos rangos, que corresponden 
a las partes de la cara. Son los puntos que necesitamos engranar y alinear para que podamos alinearnos en la cara.
'''
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Puntos utilizados para alinear las imágenes.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Puntos de la segunda imagen a superponer sobre la primera. Se superpondrá el casco convexo de cada
# elemento se superpondrá.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Cantidad de desenfoque a utilizar durante la corrección de color, como fracción de la
# distancia pupilar, puede hacer que el intercambio de caras se vea un poco más realista.
COLOUR_CORRECT_BLUR_FRAC = 0.6  # parámetro de factor de corrección de color.

#  Detectar y predecir dos objetos.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

## Introduzca aquí las rutas a sus imágenes de entrada
image1 = cv2.imread('images/Hillary.jpg')
image2 = cv2.imread('images/Trump.jpg')

swapped = swappy(image1, image2)
imshow('Face Swap 1', swapped)

swapped = swappy(image2, image1)
imshow('Face Swap 2', swapped)



# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:
    ./faceswap.py <head image> <face image>
If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.
"""





