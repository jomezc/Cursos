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






