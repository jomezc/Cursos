###########################
# 02 Espacios de color ####
###########################
import cv2
from matplotlib import pyplot as plt
'''
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.
Image("images/coca-cola-logo.png")  # mostrar el logo de coca-cola (en notebook) solo .'''

# Definir nuestra función imshow
def imshow(title="", image=None, size=10):
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


img_NZ_bgr = cv2.imread("images/New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)

# ++++ ejemplo imagen Cocacola de grises +++++
# Read and display Coca-Cola logo.
coke_img = cv2.imread("images/coca-cola-logo.png", 1)  # leer imagen. opción formato color
print("Image size is ", coke_img.shape)  # Tamaño de la imagen, Image size is  (700, 700, 3) se ve que tiene 3 canales
print("Data type of image is ", coke_img.dtype)   # Mostrar el tipo de dato de la imagen, Data type of image is  uint8
plt.imshow(coke_img)
plt.show()

'''
El color que se muestra por defecto es diferente de la imagen real. Esto se debe a que 
matplotlib espera la imagen en formato RGB mientras que OpenCV almacena imágenes en formato BGR. Por lo tanto, para una 
visualización correcta, necesitamos invertir los canales de la imagen

Lo de [::-1] es un "truco" frecuentemente usado en python para obtener una lista o una cadena "del revés". 
Se basa en el operador slice (rodaja) cuya sintaxis general es:
- iterable[inicio:fin:paso]
que permite extraer una serie de elementos del iterable, comenzando por el numerado como inicio y terminando por el 
numerado como fin-1, aumentando de paso en paso.

Si omites inicio se empezará en el primer elemento del iterable, si omites fin se terminará en el último elemento del 
iterable. Si el paso es negativo, el iterable se recorre "hacia atrás", y en ese caso los valores por defecto cuando se 
omite inicio y fin se invierten.

Así pues iterable[::-1] devuelve los elementos del iterable, comenzando por el último y terminando por el primero, 
en orden inverso a como estaban.'''

# No se va a ver bien a menos que cambiemos el orden del canal.
coke_img_channels_reversed = coke_img[:, :, ::-1]  # Invierte el orden de ese último miembro de la matriz (700, 700, 3)
plt.imshow(coke_img_channels_reversed)
plt.show()


# ***** Conversión a diferentes espacios de color
# OpenCV almacena los canales de color en un orden diferente al de la mayoría de las otras aplicaciones (BGR vs RGB).
'''
- cv2.cvtColor() Convierte una imagen de un espacio de color a otro. La función convierte una imagen de entrada de un 
                espacio de color a otro. En caso de una transformación del espacio de color RGB, el orden de los canales 
                debe especificarse explícitamente (RGB o BGR). Tenga en cuenta que el formato de color predeterminado 
                en OpenCV a menudo se denomina RGB, pero en realidad es BGR (los bytes están invertidos). Entonces, 
                el primer byte en una imagen de color estándar (24 bits) será un componente azul de 8 bits, el segundo 
                byte será verde y el tercer byte será rojo. Los bytes cuarto, quinto y sexto serían entonces el segundo 
                píxel (azul, luego verde, luego rojo), y así sucesivamente.

Sintaxis de la función
dst = cv2.cvtColor(origen, código)
dst: es la imagen de salida del mismo tamaño y profundidad que src.

La función tiene 2 argumentos requeridos:
- imagen de entrada src: 8 bits sin firmar, 16 bits sin firmar ( CV_16UC... ) o punto flotante de precisión simple.
- código: código de conversión de espacio de color (consulte ColorConversionCodes).

Documentación OpenCV
cv2.cvtColor: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab 
ColorConversionCodes: https://docs.opencv.org/4.5.1/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0'''

# Cambiando BGR a RGB
img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)  # estamos pasando la imagen y un flag que indica la conversión
plt.imshow(img_NZ_rgb)  # con el cambio Simplemente estamos mostrando la imagen original.
plt.show()


image = cv2.imread('./images/castara.jpeg')  # leer imagen
imshow("Castara, Tobago", image)
# Usamos cvtColor, para convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # estamos convirtiendo la imagen a escala de grises
imshow("Converted to Grayscale", gray_image)


# ### **Dimensiones de la imagen en escala de grises**
# Recuerde que las imágenes en color RGB tienen 3 dimensiones, una para cada color primario. La escala de grises solo
# tiene 1, que es la intensidad del gris. 0 es negro y 255 blanco lo demás la escala de gris
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/gray.png)
print(image.shape)  # (1280, 960, 3)
print(gray_image.shape)  # (1280, 960)


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.


# ***** Split y merge imágenes usando OpenCV
'''División y fusión de canales de color
- cv2.split() Divide una matriz multicanal en varias matrices de un solo canal.
- cv2.merge() Combina varias matrices para crear una única matriz multicanal. Todas las matrices de entrada deben tener 
              el mismo tamaño.
Documentación de OpenCV

https://docs.opencv.org/4.5.1/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a

A continuación vamos a cargar una imagen y posteriormente voy a llamar a la función de división abierta para tomar esa 
imagen multicanal y dividirla en sus componentes. B, G y R. Y así, cada una de estas variables representa una matriz 
numpy 2D que contiene las intensidades de píxeles para esos canales de colores.
'''
# Split de la imagen en los componentes B, G, R

# Carga nuestra imagen de entrada
image = cv2.imread('./images/castara.jpeg')
img_NZ_bgr = cv2.imread("images/New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)  # slpit + desempaquetado

'''Cada espacio de color que esté encendido se verá como una escala de grises ya que carece de los otros canales de 
color, esto es porque tiene sólo una dimensión, son sólo las intensidades en grado de componente de color azul
'''
# Ahora, simplemente usaremos Imshow para mostrar cada uno de esas representaciones como un mapa en escala de grises
plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(r, cmap='gray');plt.title("Red Channel");
plt.subplot(142); plt.imshow(g, cmap='gray');plt.title("Green Channel");
plt.subplot(143); plt.imshow(b, cmap='gray');plt.title("Blue Channel");

'''
Y luego este último fragmento de código toma esos canales individuales y usa la función de fusión para fusionar ellos de
nuevo en lo que debería ser la imagen original. Y llamaremos a esa imagen fusionada aquí, y también la mostraremos.
El lago es una especie de azul turquesa, por así decirlo. Seguro que tiene algo de verde y azul, y 
probablemente muy poco de rojo. Entonces, si ahora regresa a estos canales, puede ver que el Canal Rojo para la parte 
del lago es bajo, lo que significa que no hay mucho componente rojo en ese color. Por eso es más oscuro. Está más cerca
de cero. Y fíjate en el verde. Los canales azules tienen una intensidad bastante alta para sus respectivos colores, 
lo que indica que el color de esa agua tiene un poco de rojo, pero un poco de verde y definitivamente bastante azul.'''
# Merge de cada canal en una imagen BGR
imgMerged = cv2.merge((b, g, r))
# mostramos la imagen mergeada (Invertimos el orden de ese último miembro de la matriz)
plt.subplot(144); plt.imshow(imgMerged[:, :, ::-1]); plt.title("Merged Output");
plt.show()


# Vamos a mostrar otro ejemplo y usar con numpy
# con la otra imagen
image = cv2.imread('./images/castara.jpeg')
# Use cv2.split para obtener cada espacio de color por separado
# separa los tuneles en los componenetes de arbol azúl, verde y rojo, conviertiéndose en imágenes bidimensionales
B, G, R = cv2.split(image)
print(B.shape)  # (1280, 960)
print(G.shape)  # (1280, 960)
print(R.shape)  # (1280, 960)
imshow("Blue Channel Only", B)

# Necesitamos usar numpy para realizar esta operación, librería numérica de arrays
import numpy as np

'''Vamos a crear el arbol de la imagen de la dimensión del árbol vamos a hacer todos los otros componentes de color
a cero menos el que queremos visualizar, mediante la siguiente matriz'''
# Vamos a crear una matriz de ceros con dimensiones de la imagen h x w
zeros = np.zeros(image.shape[:2], dtype = "uint8")
imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))

### por otro lado, recargamos la imagen Original
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
'''En vez de usar una combinación de los colores RGB, usa un mapa de color llamado HUE

básicamente usando este esquema hay una forma diferente de representar los colores de diferentes espacios de color
- H representa el color de la saturación de la imagen.
- S representa la intensidad del color y 
- V representa el valor

Es decir, puede pensar en la saturación como un rojo puro versus un rojo opaco, y puede pensar en el valor S ( intensidad)
como cuán blanco u oscuro es el color, independientemente del color en sí. Y luego Hugh se parece más a la 
representación del color real, v como intensidad de brillo
.'''
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
