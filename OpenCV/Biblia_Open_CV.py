#!/usr/bin/env python
# coding: utf-8
###########################################################################
# 01 Primeros pasos carga visualización guardado dimensiones grises  ######
###########################################################################
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
from matplotlib import pyplot as plt

# Veamos qué versión estamos ejecutando
print(cv2.__version__)  # 4.7.0


# ### **Cargamos las imagenes**
# Cargue una imagen usando 'imread' especificando la ruta a la imagen
'''
OpenCV permite leer diferentes tipos de imágenes (JPG, PNG, etc). Puede cargar imágenes en escala de grises, imágenes 
en color o también puede cargar imágenes con canal alfa. Utiliza la función cv2.imread() que tiene la siguiente 
sintaxis:
- retval = cv2.imread( nombre de archivo[, banderas] )
- retval: Es la imagen si se carga correctamente. De lo contrario, es None. Esto puede suceder si el nombre del 
  archivo es incorrecto o si el archivo está dañado.

La función tiene 1 argumento de entrada obligatorio y un indicador opcional:

- nombre de archivo: puede ser una ruta absoluta o relativa. Este es un argumento obligatorio.
- Flags: estas banderas se utilizan para leer una imagen en un formato particular (por ejemplo, 
            escala de grises/color/con canal alfa). Este es un argumento opcional con un valor predeterminado de cv2.
            IMREAD_COLOR o 1 que carga la imagen como una imagen en color.
Flags disponibles:

- cv2.IMREAD_GRAYSCALE o 0: Carga la imagen en modo escala de grises
- cv2.IMREAD_COLOR o 1: Carga una imagen a color. Se descuidará cualquier transparencia de la imagen. Es la bandera 
  por defecto.
- cv2.IMREAD_UNCHANGED o -1: Carga la imagen como tal, incluido el canal alfa.

Documentación OpenCV
**Imread:**https://docs.opencv.org/4.5.1/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
ImreadModes: https://docs.opencv.org/4.5.1/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80
'''
image = cv2.imread('./images/castara.jpeg')  # leer imagen
img_NZ_bgr = cv2.imread("images/New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)

# ### **Mostramos las imagenes**
from matplotlib import pyplot as plt

# Mostramos la imagen con matpoit matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
'''usamos una funcion de open cv para cambiar el color porque Open CV en su dimensión de colores utiliza el orden Blue 
Green red, BGR y matploit usa red green blue, RGB. necesitamos esos espacios de color porque necesitamos esos 3 colores 
primarios para crear cualquier color que queramos'''
plt.show()


# Vamos a crear una función simple para hacer que mostrar nuestras imágenes sea más simple y fácil.
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]  # desempaquetamos la relación de aspecto mediante el ancho y el alto
    aspect_ratio = w / h  # calculamos la relación de aspecto
    # para asegurar que se cumpla la relación de aspecto, multiplicamos el tamaño por la relación calculada como 1º
    # parametro y le pasamos el tamaño a mostrar segundo para poder cambiar el tamaño de la imagen de salida
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Vamos a probarlo
imshow("Displaying Our First Image", image)


# ***** mostrar la imagen con matploit o con opencv

window1 = cv2.namedWindow("w1")  # creamos una ventana
cv2.imshow('image', img_NZ_bgr, )  # llamamos al show de OpenCV, OJO como es el de Open cv se guarda y muestra en BGR
cv2.waitKey(0)  # pulsar una tecla para cerrar la imagen OpenCV si 0, si ponemos numeros seran los segundos de espera

# cv2.waitKey(8000)   # 8 segundos

# keypress = cv2.waitKey(0)  # creamos una variable que contenga la primera tecla introducida
# if keypress == ord('q'):   # si la tecla ( es en ascii) coincide con el ascii de q
#     Alive = False

cv2.destroyWindow(window1)  # destruimos la ventana creada



# ### **Salvamos la imagen**
# ***** Guardar imagen cv2.imwrite()
'''
Guardar la imagen es tan trivial como leer una imagen en OpenCV. Usamos la función cv2.imwrite() con dos argumentos. El 
primero es el nombre del archivo, el segundo argumento es el objeto de la imagen.

La función imwrite guarda la imagen en el archivo especificado. El formato de imagen se elige en función de la 
extensión del nombre de archivo (consulte cv::imread para ver la lista de extensiones). En general, solo las imágenes 
de 8 bits de un solo canal o de 3 canales (con orden de canales 'BGR') se pueden guardar con esta función.

Sintaxis de la función
cv2.imwrite (nombre de archivo, img [, parámetros])
La función tiene 2 argumentos requeridos:

- nombre de archivo: puede ser una ruta absoluta o relativa.
- img: Imagen o Imágenes a guardar.
Documentación OpenCV
Imwrite: https://docs.opencv.org/4.5.1/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce 
**ImwriteFlags:**https://docs.opencv.org/4.5.1/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
'''

# Simplemente use 'imwrite' especificando el nombre del archivo y la imagen que se guardará
cv2.imwrite('output.jpg', image)

# O guárdelo como PNG (gráficos de red portátiles), que es un formato de imagen de mapa de bits sin pérdida
cv2.imwrite('output.png', image)


# ### **mostramos las dimensiones de la imagen**
# Recuerda las imágenes son arrays::
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/array.png?token=ADLZD2HNEL33JAKTYRM3B5C7WMIV4)
print(image.shape)  # (1280, 960, 3), es una estructura tridimensional, de ancho, alto y color

# # Para acceder a una dimensión, simplemente indícela usando 0, 1 o 2.
image.shape[0]


# Puedes ver que la primera dimensión es la altura y tiene 960 píxeles
# La segunda dimensión es el ancho, que es de 1280 píxeles.
# Podemos imprimirlos muy bien así:
print('Height of Image: {} pixels'.format(int(image.shape[0])))  # Height of Image: 1280 pixels
print('Width of Image: {} pixels'.format(int(image.shape[1])))  # Width of Image: 960 pixels
print('Depth of Image: {} colors components'.format(int(image.shape[2])))  # Depth of Image: 3 colors components


# leer la imagen en escala de grises e introducirlo en la variable img
img = cv2.imread('images/checkerboard_fuzzy_18x18.jpg', 0)  # cargamos la imagen con imread, 0 en escala de grises
# Lo que se carga en mi memoria es una matriz 2D de Numpy que representa la imagen.

plt.imshow(img, cmap='gray') # para que se muestren los colores correctamente
plt.show()

print(img)  # pintarlo en consola
'''
[[  0   0  15  20   1 134 233 253 253 253 255 229 130   1  29   2   0]
 [  0   1   5  18   0 137 232 255 254 247 255 228 129   0  24   2   0]
 [  7   5   2  28   2 139 230 254 255 249 255 226 128   0  27   3   2]
 [ 25  27  28  38   0 129 236 255 253 249 251 227 129   0  36  27  27]
 [  2   0   0   4   2 130 239 254 254 254 255 230 126   0   4   2   0]
 [132 129 131 124 121 163 211 226 227 225 226 203 164 125 125 129 131]
 [234 227 230 229 232 205 151 115 125 124 117 156 205 232 229 225 228]
 [254 255 255 251 255 222 102   1   0   0   0 120 225 255 254 255 255]
 [254 255 254 255 253 225 104   0  50  46   0 120 233 254 247 253 251]
 [252 250 250 253 254 223 105   2  45  50   0 127 223 255 251 255 251]
 [254 255 255 252 255 226 104   0   1   1   0 120 229 255 255 254 255]
 [233 235 231 233 234 207 142 106 108 102 108 146 207 235 237 232 231]
 [132 132 131 132 130 175 207 223 224 224 224 210 165 134 130 136 134]
 [  1   1   3   0   0 129 238 255 254 252 255 233 126   0   0   0   0]
 [ 20  19  30  40   5 130 236 253 252 249 255 224 129   0  39  23  21]
 [ 12   6   7  27   0 131 234 255 254 250 254 230 123   1  28   5  10]
 [  0   0   9  22   1 133 233 255 253 253 254 230 129   1  26   2   0]
 [  0   0   9  22   1 132 233 255 253 253 254 230 129   1  26   2   0]]

Son 18 filas y 18 columnas, y cada uno de los valores representa las intensidades de píxel para cada uno de esos píxeles
Y observe que están en el rango de 0-255 porque esta imagen está siendo representada por un entero  de8-bit unsigned 
integer (0 a 255).
'''

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

###############################
# 03 Dibujando en imágenes ####
###############################

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

'''
Comencemos dibujando una línea en una imagen. Usaremos la función cv2.line para esto.
Sintaxis
img = cv2.line(imagen, coordenadas iniciales, coordenadas finales, color, grosor, tipo linea)
img: La imagen de salida que ha sido anotada.

La función tiene 4 argumentos requeridos:
- imagen: Imagen sobre la que dibujaremos una línea
- coordenadas_iniciales: primer punto (ubicación x, y) del segmento de línea
- coordenadas_finales: Segundo punto del segmento de recta
- color: Color de la línea que se dibujará

Otros argumentos opcionales que es importante que sepamos incluyen:

- grosor: Entero que especifica el grosor de la línea. El valor predeterminado es 1.
- lineType: Tipo de línea. El valor predeterminado es 8, que representa una línea conectada a 8. Por lo general, se 
 cv2.LINE_AA (línea suavizada o suavizada) para el tipo de línea.

Documentación de OpenCV¶
https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
'''
# Tenga en cuenta que esta es una operación en el lugar, lo que significa que cambia la imagen de entrada
# A diferencia de muchas otras funciones de OpenCV que devuelven una nueva imagen sin afectar la entrada
# Recuerda que nuestra imagen era el lienzo negro
imageLine = image.copy()  # COPIAR UNA IMAGEN

# La línea comienza en (200,100) y termina en (400,100)
# El color de la línea es AMARILLO (Recordemos que OpenCV usa formato BGR)
# El grosor de la línea es 5px
# El tipo de línea es cv2.LINE_AA'''
cv2.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA);
cv2.line(imageLine, (0,0), (511,511), (255,127,0), 5)

imshow("Black Canvas With Diagonal Line", imageLine)


# ******  Dibujar un rectángulo
''''
Usaremos la función cv2.rectangle para dibujar un rectángulo en una imagen. 

sintaxis 
img = cv2.rectangle(img, pt1, pt2, color[, grosor[, lineType[, shift]]])
cv2.rectangle(imagen, vértice inicial (sup izq), vértice opuesto (inf der), color, espesor)

La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que se va a dibujar el rectángulo.
- pt1: Vértice del rectángulo. Usualmente usamos el vértice superior izquierdo aquí.
- pt2: Vértice del rectángulo opuesto a pt1. Usualmente usamos el vértice inferior derecho aquí.
- color: color del rectángulo

A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, 
dará como resultado un rectángulo relleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Enlaces de documentación de OpenCV
**rectángulo:**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
'''
# Vuelva a crear nuestro lienzo negro porque ahora tiene una línea
image = np.zeros((512,512,3), np.uint8)

# Espesor (ultimo parámetro) - si es positivo. Espesor -1 rellena el objeto
cv2.rectangle(image, (100,100), (300,250), (127,50,127), 10)
imshow("Black Canvas With Pink Rectangle", image)


# ### **Dibujemos algunos círculos**
'''círculo en una imagen. Usaremos la función cv2.circle para esto.
sintaxis funcional
img = cv2.circle(img, centro, radio, color[, grosor[, tipo de línea[, desplazamiento]]])
img: La imagen de salida que ha sido anotada.
La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que dibujaremos una línea
- centro: Centro del círculo
- radio: Radio del círculo
- color: Color del círculo que se dibujará

A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, 
dará como resultado un círculo lleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Documentación de OpenCV¶
círculo: https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
'''
# cv2.circle(imagen, centro, radio, color, relleno)
# de nuevo la imagen negra ...
image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,150,50), -1)
imshow("Black Canvas With Green Circle", image)


# ### **Polígonos**
# ```cv2.polylines(imagen, puntos, ¿Cerrado?, color, grosor)```
# si Cerrado = Verdadero, unimos el primer y último punto.
# De nuevo reseteamos la imagen negra ...
image = np.zeros((512, 512, 3), np.uint8)

# Definamos cuatro puntos mediante un array, una matriz con subpuntos dentro
pts = np.array([[10,50], [400,50], [90,200], [50,500]], np.int32)
pts.shape   # (4,2)
print(pts)
'''[[ 10  50]
 [400  50]
 [ 90 200]
 [ 50 500]]'''

# **Nota** cv2.polylines requiere que nuestros datos tengan la siguiente forma, por lo que hay que remodelarlos:
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
'''
Para escribir texto en una imagen usando la función cv2.putText.
img = cv2.putText(img, text, org, fontFace, fontScale, color[, thick[, lineType[, bottomLeftOrigin]]])
# cv2.putText(imagen, 'Texto para mostrar', punto de inicio inferior izquierdo, Fuente, Tamaño de fuente, Color, Grosor)

La función tiene 6 argumentos requeridos:
- img: Imagen sobre la que se ha de escribir el texto.
- text: Cadena de texto a escribir.
- org: esquina inferior izquierda de la cadena de texto en la imagen.
- fontFace: tipo de fuente
- fontScale: factor de escala de fuente que se multiplica por el tamaño base específico de la fuente.
- color: color de fuente

Otros argumentos opcionales que es importante que sepamos incluyen:
- grosor: número entero que especifica el grosor de línea del texto. El valor predeterminado es 1.
- lineType: Tipo de línea. El valor predeterminado es 8, que representa una línea conectada a 8. Por lo general, se usa 
            cv2.LINE_AA (línea suavizada o suavizada) para el tipo de línea.

Documentación OpenCV
**poner texto:**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576'''

# **Fuentes disponibles**
# - FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
# - FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX
# - FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
# - FONT_HERSHEY_SCRIPT_SIMPLEX
# - FONT_HERSHEY_SCRIPT_COMPLEX

image = np.zeros((1000,1000,3), np.uint8)
ourString = 'Hello World!'
cv2.putText(image, ourString, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 4)
imshow("Messing with some text", image)

#########################################################
#  04 Transformaciones - Traslaciones y Rotaciones**######
#########################################################
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
# por el ángulo de rotación elegido antihorario) y escala ( 1 significa mantener)
# cv2.getRotationMatrix2D(rotación_centro_x, rotación_centro_y, ángulo de rotación, escala)
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/rotation.png)

# Carga nuestra imagen
image = cv2.imread('images/Volleyball.jpeg')
height, width = image.shape[:2]

# Divide por dos para rotar la imagen alrededor de su centro, no rota la imagen, sino que crea la matriz que necesitamos
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

# ****** Voltear imágenes¶
''' Puedes voltearlo horizontalmente, verticalmente o en ambas direcciones
La función flip voltea la matriz en una de tres formas diferentes (los índices de fila y columna están basados en 0):

Sintaxis de la función
dst = cv.flip( src, flipCode )
* dst: matriz de salida del mismo tamaño y tipo que src.

La función tiene 2 argumentos requeridos:
- src: imagen de entrada
- flipCode: un flag para especificar cómo voltear la matriz; 
    - 0 significa girar alrededor del eje x, o voltearlo verticalmente ( boca abajo)
    - un valor positivo (por ejemplo, 1) significa girar alrededor del eje y, o voltearlo horizontalmente  (espejo)
    - Un valor negativo (por ejemplo, -1) significa girar alrededor de ambos ejes.
Documentación OpenCV
flip: https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441
'''
img_NZ_rgb_flipped_horz = cv2.flip(image, 1)  # un giro horizontal 90º, un 'volteo'
img_NZ_rgb_flipped_vert = cv2.flip(image, 0)
img_NZ_rgb_flipped_both = cv2.flip(image, -1)

# mostramos las imágenes
plt.figure(figsize=[18, 5])
plt.subplot(141);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_horz, cv2.COLOR_BGR2RGB));plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_vert, cv2.COLOR_BGR2RGB));plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_both, cv2.COLOR_BGR2RGB));plt.title("Both Flipped");
plt.subplot(144);plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.title("Original");
plt.show()


###################################################################
# # 05 Escalado, cambio de tamaño, interpolaciones y recorte** ####
###################################################################
# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt


def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Cargamos imagen original de pruebas en escala de grises
cb_img = cv2.imread("images/checkerboard_18x18.png", 0)

# Establezca el mapa de colores en escala de grises para una representación adecuada, si no se ven colores incorrectos
plt.imshow(cb_img, cmap='gray')
plt.show()
print(cb_img)
'''
[[  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]'''

# ****** Acceso a píxeles individuales
'''
Veamos cómo acceder a un píxel en la imagen. Para acceder a cualquier píxel en una matriz numpy, debe usar notación de 
matriz:
- matrix[r,c], donde r es el número de fila y c es el número de columna. la matriz está indexada en 0.
Por ejemplo, si desea acceder al primer píxel, debe especificar matrix[0,0]. Veamos con algunos ejemplos. 
Imprimiremos un píxel negro desde la parte superior izquierda y un píxel blanco desde la parte superior central.'''

# Imprime el primer pixel del primer cuadro negro
print(cb_img[0, 0])  # 0
# imprima el primer píxel blanco a la derecha del primer cuadro negro
print(cb_img[0, 6])  # 255

# ****** Modificando los píxeles de las imágenes

cb_img_copy = cb_img.copy()
# cb_img_copy[2, 2] = 200
# cb_img_copy[2, 3] = 200
# cb_img_copy[3, 2] = 200
# cb_img_copy[3, 3] = 200
cb_img_copy[2:4, 2:4] = 200  # Lo mismo que lo de antes de una

plt.imshow(cb_img_copy, cmap='gray')
plt.show()
print(cb_img_copy)

'''[[  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0 200 200   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0 200 200   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [255 255 255 255 255 255   0   0   0   0   0   0 255 255 255 255 255 255]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]
'''

# ### **Cambio de tamaño**

# La función de cambio de tamaño cambia el tamaño de la imagen  aumentando o disminuyendo hasta el tamaño
# especificado, usando la función cv2.resize, sus argumentos son:

# cv2.resize(imagen, dsize(tamaño de la imagen de salida), escala x, escala y, interpolación)
# La función tiene 2 argumentos requeridos:
# - src: imagen de entrada
# - dsize: tamaño de la imagen de salida. Si dsize es Ninguno, la imagen de salida se calcula en función de la escala
# usando la escala x e y
# el tipo de dst es el mismo que el de src.


# Los argumentos opcionales que se utilizan a menudo incluyen:
# - fx: Factor de escala a lo largo del eje horizontal; cuando es igual a 0, se calcula como (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜
# - fy: Factor de escala a lo largo del eje vertical; cuando es igual a 0, se calcula como (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜

# -Interpolación: es básicamente un algoritmo para encontrar un valor entre dos puntos. Si tuviéramos unos puntos por
# una ruta de gps la interpolación adivinará puntos intermedios entre los originales del camino, aportando información
# adicional, es una forma de agregar más datos a los existentes para conectar los puntos existentes ( en el ejemplo)
# si estamos agrandando una imagen, estamos tratando de adivinar los puntos que se tomarán en una nueva dimensión.
# Adivina algorítmicamente la mejor suposición

# #### **Lista de métodos de interpolación, las diferentes fórmulas que suelen aplicarse:**
# - cv2.INTER_LINEAR- Bueno para hacer zoom o muestreo ascendente (predeterminado), una interpolación bilineal
# - cv2.INTER_AREA- Bueno para reducir o reducir el muestreo, remuestreo usando relación de área de píxeles. Puede ser
#                   un método preferido para la disminución de imágenes, ya que brinda resultados sin muaré. Pero cuando
#                   se amplía la imagen, es similar al método INTER_NEAREST.
# - cv2.INTER_NEAREST - Más rápido, una interpolación de vecino más cercano
# - cv2.INTER_CUBIC- Mejor, una interpolación bicúbica sobre una vecindad de 4×4 píxeles
# - cv2.INTER_LANCZOS4 - El Mejor, una interpolación de Lanczos sobre un vecindario de 8×8 píxeles
# Documentación OpenCV
#  https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
# Vea más sobre su desempeño - https://chadrick-kwag.net/cv2-resize-interpolation-methods/


# carga nuestra imagen de entrada
image = cv2.imread('images/oxfordlibrary.jpeg')
imshow("Scaling - Linear Interpolation", image)

# Si no se especifica ninguna interpolación, cv.INTER_LINEAR se usa por defecto
# método 1: Especificación del factor de escala usando fx y fy
# Hagamos nuestra imagen 3/4 de su tamaño original
# vamos a usar los efectos del argumento y la forma para reducir la imagen en un 75% (0.75 de ancho y alto)
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
imshow("0.75x Scaling - Linear Interpolation", image_scaled)

# Dupliquemos el tamaño de nuestra imagen
'''imagen, tamaño de salida ( al usar escala esta bien ponerlo a None factores de escala fx y tener Y. En este ejemplo, 
sólo vamos a establecerlos en dos Así que vamos a duplicar el tamaño.'''
img_scaled2 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
imshow("2x Scaling - Inter Cubic", img_scaled2)

# Dupliquemos el tamaño de nuestra imagen usando la interpolación inter_nearest
img_scaled3 = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
imshow("2x Scaling - Inter Nearest", img_scaled3)

# método 2: Especificación del tamaño exacto de la imagen de salida
desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)
'''
vamos a establecer un ancho y alto específicos para la imagen y vamos a crear este vector bidimensional indicando ambas 
dimensiones y lo usamos como segundo argumento para la función de cambio de. la imagen se ha distorsionado ahora porque 
no mantuvimos la relación de aspecto original.'''
# Cambiar el tamaño de la imagen de fondo al mismo tamaño que la imagen del logotipo
im = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(im)
plt.show()

# EJEMPLO 2
img_scaled4 = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
imshow("Scaling - Inter Area", img_scaled4)


# Cambiar el tamaño manteniendo la relación de aspecto
desired_width = 100
''' ahora vamos a comenzar especificando un ancho de 100 y luego calcularemos la altura deseada asociada manteniendo 
la relación de aspecto. Así que aquí estamos creando esta proporción del ancho deseado al ancho original de la imagen 
y luego usando ese factor para derivar la altura deseada aquí. cuando pasamos esa dimensión revisada a la función de 
cambio de tamaño, obtenemos una imagen de cien píxeles de ancho y la cantidad adecuada de alto para mantener la 
relación adecuada, que resulta ser de unos sesenta y siete píxeles.

------------------------------------------------------------
para saber el ancho y el alto  función shape() (dimensiones) 
------------------------------------------------------------
Las dimensiones de una imagen dada, como la altura de la imagen, el ancho de la imagen y la cantidad de canales en la 
imagen, se denominan shape (forma) de la imagen y  se almacena en numpy.ndarray.
La función shape() puede proporcionar la dimensión de una imagen dada y almacena cada una de las dimensiones de la 
imagen, como la altura de la imagen, el ancho de la imagen y la cantidad de canales en la imagen en diferentes índices.

La altura de la imagen se almacena en el índice 0.
El ancho de la imagen se almacena en el índice 1.
El número de canales en la imagen se almacena en el índice 2.

Ejemplo:

dimensions = input_image.shape
height = input_image.shape[0] 
width = input_image.shape[1] 
number_of_channels = input_image.shape[2]

- input_image: representa la imagen cuyas dimensiones se van a encontrar.
- dimensions: representan las dimensiones de la imagen.
- height: representa la altura de la imagen de entrada.
- width: representa el ancho de la imagen de entrada.
- number_of_channels: representa el número de canales en la imagen.

La relación de aspecto o ratio de una imagen es la proporción entre el ancho y la altura de la imagen. Se calcula
dividiendo la anchura entre la altura, y se expresa normalmente con dos números separados por dos puntos. Por ejemplo 
3:2, significa que por cada tres unidades a lo largo hay dos unidades a lo alto
'''
print(im.shape[1])
aspect_ratio = desired_width / im.shape[1]  # calculamos el radio de aspecto
desired_height = int(im.shape[0] * aspect_ratio)  # calculamos la nueva altura
dim = (desired_width, desired_height)

resized_cropped_region = cv2.resize(im, dsize=dim, interpolation=cv2.INTER_AREA)  # Cambiar el tamaño de img
plt.imshow(resized_cropped_region)

# Ahora, salvemos la imagen redimensionada (recortada)
# cambiamos el orden del canal
im = im[:, :, ::-1]

# Save resized image to disk
cv2.imwrite("images/im.png", im)


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
''' Recortar una imagen se logra simplemente seleccionando una región específica (píxel) de la imagen.
Es simplemente indexar una imagen existente y extraer la región que le interesa.'''

# es una técnica muy útil especialmente con detectores de objetos u OCR donde tienes que recortar segmentos de la imagen
image = cv2.imread('images/oxfordlibrary.jpeg')

# Obtenga las dimensiones de nuestra imagen
height, width = image.shape[:2]

# Obtengamos las coordenadas del píxel inicial (arriba a la izquierda del rectángulo de recorte)
# usando 0.25 para obtener la posición x, y que está 1/4 por debajo de la parte superior izquierda (0,0)

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

# Otro Ejemplo
img_NZ_bgr = cv2.imread("images/New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)  # cargamos una imagen a color
img_NZ_rgb = img_NZ_bgr[:, :, ::-1]  # Invertimos el último color
plt.imshow(img_NZ_rgb)  # Mostramos la imagen
plt.show()  # para que se muestre
# Recortar la región media de la imagen
cropped_region = img_NZ_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()


#!/usr/bin/env python
# coding: utf-8

################################################
# 06 Operaciones aritméticas y bit a bit** #####
################################################
'''
Las técnicas de procesamiento de imágenes aprovechan las operaciones matemáticas para lograr diferentes resultados.
La mayoría de las veces llegamos a una versión mejorada de la imagen usando algunas operaciones básicas. Echaremos un
vistazo a algunas de las operaciones fundamentales que se usan a menudo en las canalizaciones de visión por computadora.
En este cuaderno cubriremos operaciones aritméticas como la suma y la multiplicación.
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



# ## **Operaciones aritméticas**
# Son operaciones sencillas que nos permiten sumar o restar directamente a la intensidad del color.
# Calcula la operación por elemento de dos matrices. El efecto general es aumentar o disminuir el brillo.

'''
La primera operación que analizamos es la simple adición o sustracción de imágenes. Esto da como resultado aumentar o 
disminuir el brillo de la imagen ya que eventualmente estamos aumentando o disminuyendo los valores de intensidad de 
cada píxel en la misma cantidad. Entonces, esto resultará en un aumento/disminución global del brillo.

 M = ...
- numpy.ones(): devuelve un array del tamaño y tipo indicados inicializando sus valores con unos
- crea una matriz del tamaño image.shape (con la dimensión de la imagen) es decir una imagen con el 
  tamaño de la original), tipo entero grande y con todo valor 100, es decir se crea una imagen que si la imprimimos es 
  un gris [[41 41 41 ...  5  5  5]...
Y ahora simplemente vamos a usar las funciones de abrir, sumar y restar para sumar y restar esa matriz de la imagen 
original, siendo todo lo que se requiere para generar una imagen más oscura que la original y una imagen que es mas 
clara que la original
'''

# cv2.imread carga nuestra imagen como una imagen en escala de grises
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


# otro ejemplo completo
img_bgr = cv2.imread("images/New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)  # cargar imagen a color [[[188 183 174],[189....
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cambiar el color a RGB (cv2 por defecto BGR)
matrix = np.ones(img_rgb.shape, dtype = "uint8") * 50
img_rgb_brighter = cv2.add(img_rgb, matrix)  # se le suma a la imagen original la matriz [[[224 233 238], [226...
img_rgb_darker   = cv2.subtract(img_rgb, matrix)  # se le resta a la imagen original la matriz [[[124 133 138], [122...
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Brighter");
plt.show()


# **** Multiplicación o Contraste
'''
Al igual que la suma puede resultar en un cambio de brillo, la multiplicación se puede usar para mejorar el contraste 
de la imagen. El contraste es la diferencia en los valores de intensidad de los píxeles dentro de una imagen. 
Multiplicar los valores de intensidad con una constante puede hacer que la diferencia sea mayor o menor (si el factor de
multiplicación es < 1).
'''
matrix1 = np.ones(img_rgb.shape) * .8  # Crea una matriz del mismo tamaño inicializado todo a 0.8 [[[0.8 0.8 ...
matrix2 = np.ones(img_rgb.shape) * 1.2  # Crea una matriz del mismo tamaño inicializado todo a 1.2 [[[1.2 1.2 ...

# convertimos los puntos de la imagen a flotante y multiplicamos por la matriz, convirtiendo después a un array de uint
# 8-bit unsigned integer (0 a 255).
img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))    # [[[139 146 150 ...
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))  # [[[208 219 255 ....

# mostramos las imagenes
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Higher Contrast");
plt.show()

'''la imagen de alto contraste, hay un código de color extraño al mostrarlo, Y la razón de esto es porque cuando 
multiplicamos la imagen original por esta matriz, tiene un factor de uno punto dos en ella. Potencialmente obtenemos 
valores superiores a 255. Entonces,  la imagen original aquí, las nubes  probablemente estaban cerca de 255. Algunos de 
ellos, al menos. Y cuando multiplicamos por uno punto dos, pasamos a cincuenta y cinco.

Entonces, cuando intentamos convertir esos valores en un número de ocho bits sin signo en lugar de exceder 255,
simplemente pasan a un número pequeño. provocando estos valores de intensidad cercanos a cero y siendo el motivo del
problema.

numpy.clip(): La función se utiliza para recortar (limitar) los valores en una matriz.
Dado un intervalo, los valores fuera del intervalo se recortan a los bordes del intervalo. Por ejemplo, si se especifica
 un intervalo de [0, 1], los valores menores que 0 se convierten en 0 y los valores mayores que 1 se convierten en 1.

Para solucionarlo lo que podemos hacer es usar la función clip de numpy para recortar primero esos valores al
rango de cero a 255 antes de convertirlos un entero de 8 bits (0-255), provocando que esta parte de la imagen se sature
por completo, teniendo algunos valores 255 por lo que realmente no tienen información .
'''
matrix1 = np.ones(img_rgb.shape) * .8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_lower = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_lower);  plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_higher);plt.title("Higher Contrast");
plt.show()

# ********************************************
# ***** 07 Operaciones bit a bit con imágenes
# ********************************************
# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imágenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

'''
Las técnicas de procesamiento de imágenes aprovechan diferentes operaciones lógicas para lograr diferentes resultados. 
La mayoría de las veces llegamos a una versión mejorada de la imagen usando algunas operaciones lógicas básicas como 
las operaciones AND y OR.

Sintaxis:
 cv2.bitwise_and(). Otros incluyen: cv2.bitwise_or(), cv2.bitwise_xor(), cv2.bitwise_not()

dst = cv2.bitwise_and( src1, src2[, dst[, máscara]] )
- dst: matriz de salida que tiene el mismo tamaño y tipo que las matrices de entrada.

La función tiene 2 argumentos requeridos:
- src1: primera matriz de entrada o un escalar.
- src2: segunda matriz de entrada o un escalar.
Un argumento opcional importante es:
- máscara: máscara de operación opcional, matriz de un solo canal de 8 bits, que especifica los elementos de la matriz 
de salida que se cambiarán, es decir, a que parte de estas dos imágenes se aplica la lógica de la operación.

Documentación OpenCV
https://docs.opencv.org/4.5.1/d0/d86/tutorial_py_image_arithmetics.html 
https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
'''
# leemos dos imagenes un rectángulo y un circulo.
img_rec = cv2.imread("images/rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("images/circle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[20, 5])
plt.subplot(121);plt.imshow(img_rec, cmap='gray')
plt.subplot(122);plt.imshow(img_cir, cmap='gray')
plt.show()
print(img_rec.shape)  # (200, 499)


# **** Operación not
''' En el operador NOT, cuando una entrada es verdadera o 1, su salida es falso o  0, y viceversa. En OpenCV se realiza 
el mismo procedimiento, con la diferencia que en vez de 1 se emplea 255, como he dicho antes, para poder visualizar el 
resultado o salida en colores blanco y negro'''
result = cv2.bitwise_not(img_rec)
plt.imshow(result, cmap='gray')
plt.show()

# **** Operación and
'''
Estamos pasando la imagen del rectángulo en la imagen del círculo.Y luego estamos indicando que la máscara es ninguna.
Así que simplemente vamos a hacer una comparación bit a bit entre estas dos imágenes y el valor devuelto de esa 
comparación será 255 (blanco) si los píxeles correspondientes en ambas imágenes son blancos.

Entonces, en este caso, el resultado será solo este lado izquierdo de este semicírculo, ya que ese es el único región en
ambas imágenes donde los píxeles son blancos.'''
result = cv2.bitwise_and(img_rec, img_cir, mask = None)
plt.imshow(result, cmap='gray')
plt.show()

# **** Operación or
'''Ahora el valor de retorno de la operación será blanco si el píxel correspondiente de cualquier punto de la imagen es 
blanco ( 255). EN este ejemplo, obtenemos todo el lado izquierdo del rectángulo, que es blanco y luego el lado derecho
lado de la mano del círculo.'''
result = cv2.bitwise_or(img_rec, img_cir, mask = None)
plt.imshow(result, cmap='gray')
plt.show()

# **** Operación xor
''' Solo devolverá un valor de blanco si el píxel correspondiente es blanco (255) en una imagen, pero no en ambas.'''
result = cv2.bitwise_xor(img_rec, img_cir, mask = None)
plt.imshow(result, cmap='gray')
plt.show()

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




########################################################################
# 09 Umbralización, binarización y umbralización adaptativa ######
########################################################################

#Imágenes binarizadas, estamos conviertiendo a binario los colores, los píxeles de una imagen a 0 o 1, mediante un

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


# ******* Aplicación Operaciones bit a bit: manipulación de logotipos  ##########

# **** Leer imagen en primer plano
img_bgr = cv2.imread("images/coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
print(img_rgb.shape)
logo_w = img_rgb.shape[0]  # guardamos el ancho de la imagen
logo_h = img_rgb.shape[1]  # guardamos el alto de la imagen

# **** leer la imagen de fondo
# Leer en la imagen del fondo del tablero de color
img_background_bgr = cv2.imread("images/checkerboard_color.png")
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

# Establecer el ancho deseado (logo_w) y mantener la relación de aspecto de la imagen
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))

# Cambiar el tamaño de la imagen de fondo al mismo tamaño que la imagen del logotipo
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)
plt.imshow(img_background_rgb)
plt.show()
print(img_background_rgb.shape)

# **** se cra una máscara de la imagen de primer plano
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

'''vamos a pasar el logotipo aquí para ver el color, convertirlo a gris. y luego use la treshold para 
crear una máscara binaria a partir de la imagen en escala de grises.Entonces esto solo va a contener valores de cero y 
255.

Umbralización o thresholding: Consiste en modificar una imagen a una representación binaria, por medio de la 
modificación de los valores de los pixeles estableciendo un valor umbral, es decir realizar la Binarización,  pasar a 
blanco o negro una escala de grises de una imagen en mediante un umbral todo por encima de un  cierto umbral se vuelve 
blanco y por debajo negro mediante un algortimo , exisitiendo la operación binaria contraria ( en vez de blanco negro y 
viceversa). 
El truncamiento es que todo lo que está por encima d eun umbral se convierte en ese valor máximo del umbral
TOZERO es que todo lo que es menor que el umbral se vuelve 0 y TOZERO_INV lo contrario
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.57.55%20am.png)
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.58.09%20am.png)
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

sintaxis: 
ret,thresh = cv2.threshold(img, umbral, valorMax , tipo)

Los parámetros son los siguientes:
- img es la imagen gris que va a ser analizada
- umbral es el valor indicado a analizar en cada píxel
- valorMax Valor que se coloca a un píxel si sobrepasa el umbral
- tipo se elige un tipo de umbralización: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO], 
  THRESH_TOZERO_INV, THRESH_OTSU.

La función devuelve:
- thresh imagen binarizada
- ret valor del umbral

THRESH_BINARY
Y muestra que, si el píxel (src(x,y)) supera el umbral (thresh), en la imagen binarizada a los píxeles que superaron 
el umbral se les asigna el valor máximo establecido.

THRESH_BINARY_INV
si el píxel (src(x,y)) supera el umbral (thresh), en la imagen binarizada a los píxeles que superaron el umbral se les 
asigna cero 0 y a los que no superaron el umbral se les asigna el valor máximo establecido (maxval en este ejemplo es 
255)

THRESH_TRUNC
Estas muestran que, si el píxel (src(x,y)) supera el umbral (thresh), en la imagen binarizada a los píxeles que 
superaron el umbral se les asigna el mismo valor del umbral y a los que no superaron el umbral se les asigna los mismos
valores que tenían originalmente.

THRESH_TOZERO
si el píxel (src(x,y)) supera el umbral (thresh), en la imagen binarizada a los píxeles que superaron el umbral 
mantienen el valor de los pixeles originalmente, y cuando no superan el umbral se les asigna cero.

THRESH_TOZERO_INV
si el píxel (src(x,y)) supera el umbral (thresh), en la imagen binarizada a los píxeles que superaron el umbral se les 
asigna cero, y a los píxeles que no superaron el umbral se les asigna el mismo valor que originalmente tenías.
.'''
# Aplique un umbral global para crear una máscara binaria del logotipo
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_mask, cmap="gray")
plt.show()
print(img_mask.shape)

# **** Se invierte la máscara
# Se cre una máscara inversa
'''
Y luego vamos a realizar una operación similar aquí abajo, pero sin usar la función de umbral.
Aunque podríamos haberlo hecho, podríamos haber usado la función de umbral aquí abajo y especificar un umbral
máscara inversa binaria:
retval2, img_mask2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

pero en su lugar podemos simplemente llamar a la función bitwise_not en la máscara de imagen para devolver la máscara 
inversa
'''
img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray")
plt.show()

# **** Se aplica el fondo a la máscara
'''para mostrar el fondo  "detrás" de las letras del logotipo se utiliza bitwise_and usando 
la imagen de fondo consigo misma pero utilizando la máscara original creada pero solo la va a aplicar a la máscara, que 
es las letras blancas en este caso, es decir vamos a hacer una comparación bit a bit entre estas dos imágenes y el valor
devuelto de esa comparación será el de la imagen si los píxeles correspondientes en ambas imágenes son iguales solo en 
las letras y en el resto 0 (negro), con esto obtenemos solo los colores que se muestran en el logotipo.'''
img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
plt.imshow(img_background)
plt.show()

# **** Se aísla el primer plano de la imagen
'''Aísle el primer plano (rojo de la imagen original) usando la máscara inversa consigo misma y aplicándolo a la mascara
inversa con lo que se aplicará a toda la imagen la comparación de rojo = rojo menos a las letras, quedando éstas a 0
(negro)'''
img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.imshow(img_foreground)
plt.show()

# **** Se obtiene el resultado
'''Ahora sumando las dos imagenes que acabamos de crear obtenemos el resultado del fondo rojo + las letras de colores'''
result = cv2.add(img_background, img_foreground)
plt.imshow(result)
cv2.imwrite("logo_final.png", result[:, :, ::-1])
plt.show()

############

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

# Dilatar aquí, es decir agregando píxeles a los límites de los objetos, el fondo en este caso en una imagen
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
    """devuelve las áreas de todos los contornos como una lista, estamos recorriendo los contornos que antes hemos
    sacado, obteniendo el área de cada contorno y añadiendolo a una lista"""
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
# ( son 2 cuadrados,1 círculo, 1 triángulo) que hemos calculado a raíz de los contornos, es decir, hemos dibujado
# el controno y un número que clasifica de más grande a pequeño las áreas de las figuras del ejemplo
imshow('Contours by area', image)


# #### **Definir algunas funciones que usaremos**
# Funciones que usaremos para ordenar por posición
def x_cord_contour(contours):
    """Devuelve la coorednada X para el centroide del controno ( una función Usando los momentos para sacar
    la coordenada x )"""
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
#                                     pequeños dan aproximaciones precisas, los valores grandes dan una aproximación
#                                     más genérica una buena regla empírica od es menos del 5% del perímetro del
#                                     contorno
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
    # ahora calcular el contorno aproximado con ese porcentaje de precisión
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

###################################################
# 13 Detección de líneas, círculos y manchas ######
###################################################

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
# cv2.HoughLinesP(imagen binarizada, precisión 𝜌, precisión 𝜃, umbral, longitud mínima de línea, espacio máximo entre
#                 líneas)


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

# ******************************************
# ***** 16 Usando la camara en OpenCV
# ******************************************
import cv2
import sys

# especificamos un índice de dispositivo de cámara predeterminado de cero.
s = 0
print(sys.argv)  # contiene los argumentos de la librería sys, por ejemplo 0 es la ruta
# ['C:\\Users\\jgomcano\\PycharmProjects\\guiapython\\OpenCV\\Usando la camara en openCV\\16 Usando_camara_OpenCV.py']
# y simplemente estamos verificando si hubo una especificación de línea de comando para anular ese valor predeterminado.
if len(sys.argv) > 1:
    s = sys.argv[1]
print(s)  # 0
source = cv2.VideoCapture(s)  # llamamos a la clase de captura de video para crear un objeto de captura de video,
#  Con el índice 0 accederá a la cámara predeterminada en su sistema, si no hay que indicarlo
win_name = 'Vista de camara'
# estamos creando una ventana con nombre, que eventualmente vamos a enviar la salida transmitida
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

'''ciclo while nos permitirá transmitir continuamente video desde la cámara y enviarlo a la salida a menos que el 
usuario pulse la tecla de escape.'''
while cv2.waitKey(1) != 27:  # Escape
    '''usa esa fuente de objeto de captura de vídeo  de captura de video para llamar al método read, que  devolverá un 
    solo cuadro de la transmisión de video, así como una variable lógica has_frame.
    Entonces, si hay algún tipo de problema con la lectura de la transmisión de video o el acceso a la cámara, entonces 
    has_frame sería falso y saldríamos del bucle.
    De lo contrario, continuaríamos y llamaríamos a la función de visualización de mensajes instantáneos y abriríamos
     kbps para enviar el video (frame) a la ventana de salida'''
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)

### generar un boceto

# Nuestra función generadora de bocetos
def sketch(image):
    # Convierte la imagen a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Limpia la imagen usando Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Extraer bordes
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # Invertir y binarizar la imagen
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


# Inicializar webcam, cap es el objeto proporcionado por VideoCapture
cap = cv2.VideoCapture(0)

while True:
    # Contiene un booleano indicando si tuvo éxito (ret)
    # También contiene las imágenes recogidas de la webcam (frame)
    ret, frame = cap.read()
    # Pasamos nuestro frame a nuestra función sketch directamente dentro de cv2.imshow()
    cv2.imshow('Nuestro dibujante en vivo', sketch(frame))
    if cv2.waitKey(1) == 13:  # 13 es la tecla Enter
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()


# ************************************************
# ***** 17 uso camara Filtrado de imagen en OpenCV
# ************************************************
import cv2
import sys
import numpy

PREVIEW  = 0   # Vista previa
BLUR     = 1   # filtro de desenfoque
FEATURES = 2   # Detector de características de corner
CANNY    = 3   # Detector de borde astuto

# Estamos definiendo un pequeño diccionario de configuración de parámetros para el detector de características de corner
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)

'''Estamos configurando el índice del dispositivo para la cámara, creando una ventana de salida para los 
resultados transmitidos y luego crea un objeto de captura de video  para que podamos procesar la transmisión de 
video en el bucle '''
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()  # leemos el frame de vídeo
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)  # mediante flip giramos el video horizontalmente

    if image_filter == PREVIEW:  # según la configuración de ejecución del script
        result = frame  # solo cogemos el frame y lo mostramos
    elif image_filter == CANNY:
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
        - opening_size: Tamaño de apertura del filtro Sobel.
        - L2Gradient: Parámetro booleano utilizado para mayor precisión en el cálculo de Edge Gradient.
        el umbral mínimo y el máximo dependerá de cada situación.
        Docu 
        https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html'''
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
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
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convertimos la imagen a escala de grises
        '''La función goodFeaturesToTrack encuentra N esquinas más fuertes 
         cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, [,mask[,blockSize[,useHarrisDetector[,k]]]])

        - imagen: entrada de imagen de un solo canal de 8 bits o punto flotante de 32 bits
        - maxCorners - Número máximo de esquinas a devolver. Si hay más esquinas de las que se encuentran, se devuelve 
          la más fuerte de ellas. si <= 0 implica que no se establece ningún límite en el máximo y se devuelven todas 
          las esquinas detectadas.
        - qualityLevel - Parámetro que caracteriza la calidad mínima aceptada de las esquinas de la imagen. Consulte el 
          párrafo anterior para obtener una explicación.
        - minDistance - Distancia euclidiana mínima posible entre las esquinas devueltas
        - máscara - Región de interés opcional. Si la imagen no está vacía, especifica la región en la que se detectan 
          las esquinas.
        - blockSize - Tamaño de un bloque promedio para calcular una matriz de covariación derivada sobre cada 
          vecindario de píxeles
        - useHarrisDetector - ya sea para usar Shi-Tomasi o Harris Corner
        -k - Parámetro libre del detector de Harris
        Documentación: 
        https://theailearner.com/tag/cv2-goodfeaturestotrack/
        https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html'''
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:  # Devuelve una lista de esquinas encontradas en la imagen
            ''' Y si detectamos una o más esquinas, simplemente anotaremos el resultado con pequeños
             círculos verdes para indicar las ubicaciones de esas características ojo con los parámetros
             al ser capturados las  posiciones x,y PASARLO a entero'''
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    cv2.imshow(win_name, result)  # Enviamos el resultado a la salida

    # para poder cambiar el tratamiento de la imagen dependiendo de la tecla introducida
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)


# ************************************************************
# ***** 18 Caracteristicas de la imagen y alineación de la imagen
# ************************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Demostraremos los pasos a través de un ejemplo en el que alinearemos una foto de un formulario tomado con un teléfono 
móvil con una plantilla del formulario. La técnica que usaremos a menudo se denomina alineación de imágenes "basada en 
funciones" porque en esta técnica se detecta un conjunto escaso de funciones en una imagen y se compara con las 
funciones en la otra imagen. Luego se calcula una transformación basada en estas características combinadas que deforma 
una imagen sobre la otra.

La alineación de imágenes (también conocida como registro de imágenes) es la técnica de deformar una imagen (o, a veces,
ambas imágenes) para que las características de las dos imágenes se alineen perfectamente.
'''

# **** Paso 1: Lea la plantilla y la imagen escaneada
# Leemos la imagen de referencia
refFilename = "images/form.jpg"
print("Reading reference image : ", refFilename)
im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

# leemos la imagen que queremos alinear
imFilename = "images/scanned-form.jpg"
print("Reading image to align : ", imFilename)
im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Mostramos las imágenes cargadas
plt.figure(figsize=[20, 10]);
plt.subplot(121);
plt.axis('off');
plt.imshow(im1);
plt.title("Original Form")
plt.subplot(122);
plt.axis('off');
plt.imshow(im2);
plt.title("Scanned Form")
plt.show()

# ****** Paso 2: encuentra puntos clave en ambas imágenes
'''
objetivo  es tratar de extraer información significativa que esté contextualmente relacionada con la imagen en sí.
Por lo general, buscamos bordes, esquinas y texturas en las imágenes, las función orb() es una forma de hacerlo, 
vamos a crear este objeto orbe, y luego vamos a usar ese objeto para detectar y calcular puntos clave y descriptores 
para cada una de las imágenes.

Entonces, los puntos clave son características interesantes en cada imagen que generalmente se asocian con algunos 
puntos nítidos. borde o esquina, y están descritos por un conjunto de coordenadas de píxeles que describen la ubicación
del punto clave. El tamaño del punto clave. En otras palabras, la escala del punto clave y luego también la orientación 
del punto clave. luego hay una lista asociada de descriptores para cada punto clave, y cada descriptor es en realidad un
vector de alguna información que describe la región alrededor del punto clave, que actúa efectivamente como una firma 
para ese punto clave. Es una representación vectorial de la información de píxeles alrededor del punto clave. Y la idea 
aquí es que si estamos buscando el mismo punto clave en ambas imágenes, podemos intentar usar los descriptores para 
emparejarlos.'''

# Convertimos las imágenes a escala de grises
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detecta características de ORB y calcula descriptores.

MAX_NUM_FEATURES = 500
'''El algoritmo utilizado para la detección de características de la imagen dada junto con la orientación y los 
descriptores de la imagen se denomina algoritmo ORB y es una combinación del detector de punto clave FAST y el 
descriptor BRIEF.

- Localizador : identifica puntos en la imagen que son estables bajo transformaciones de imagen como traslación 
  (desplazamiento), escala (aumento/disminución de tamaño) y rotación. El localizador encuentra las coordenadas x, y de 
  dichos puntos. El localizador que utiliza el detector ORB se llama FAST .
- Descriptor : El localizador del paso anterior solo nos dice dónde están los puntos interesantes. La segunda parte del 
  detector de características es el descriptor que codifica la apariencia del punto para que podamos distinguir un punto
  característico de otro. El descriptor evaluado en un punto característico es simplemente una matriz de números. 
  Idealmente, el mismo punto físico en dos imágenes debería tener el mismo descriptor. ORB usa una versión modificada 
  del descriptor de características llamado BRISK .

sintaxis 
ORB_object = cv.ORB_create()
keypoints = ORB_object.detect(input_image)
keypoints, descriptors = ORB_object.compute(input_image, keypoints)

- El algoritmo ORB se puede implementar usando una función llamada función ORB().
- La implementación del algoritmo ORB funciona creando un objeto de la función ORB().
- Luego hacemos uso de una función llamada función ORB_object.detect() para detectar los puntos clave de una imagen dada
- Luego hacemos uso de una función llamada función ORB_object.compute() para calcular los descriptores de una imagen 
  determinada.
- Luego, la imagen con los puntos clave calculados dibujados en la imagen se devuelve como salida
https://www.educba.com/opencv-orb/
https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/


'''
orb = cv2.ORB_create(MAX_NUM_FEATURES)

# detectAndCompute aúna las dos explicadas anteriormente
keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

'''Estamos dibujando los puntos clave detectados en la imagen usando la función drawKeypoints()
Sintaxis de la función drawKeypoints():
dibujar puntos clave (imagen_de_entrada, puntos_clave, imagen_de_salida, color, bandera)
parámetros:
- input_image: la imagen que se convierte en escala de grises y luego los puntos clave se extraen utilizando los 
                algoritmos SURF o SIFT se denomina imagen de entrada.
- key_points: los puntos clave obtenidos de la imagen de entrada después de usar los algoritmos se denominan puntos 
              clave.
- output_image :   imagen sobre la que se dibujan los puntos clave.
- color : el color de los puntos clave.
- bandera: las características del dibujo están representadas por la bandera.
https://www.geeksforgeeks.org/python-opencv-drawkeypoints-fuction/
'''
im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255, 0, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255, 0, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

'''
Hemos calculado los puntos clave calculados en los descriptores de cada imagen. Y aquí, en estas cifras, se muestran 
solo los puntos clave.  todos estos círculos rojos son puntos clave. El centro del círculo es la ubicación del punto 
clave. El tamaño del círculo representa la escala del punto clave y luego la línea que conecta el centro del círculo al 
exterior del círculo representa la orientación del punto clave. Hay algunos puntos clave en ambas imágenes que tal vez 
sean los mismos, y esos son los que vamos a tratar de encontrar para que podamos calcular el gráfico de Hamas entre 
estas dos representaciones de imágenes.'''

plt.figure(figsize=[20, 10]);
plt.subplot(121);
plt.axis('off');
plt.imshow(im1_display);
plt.title("Original Form")
plt.subplot(122);
plt.axis('off');
plt.imshow(im2_display);
plt.title("Scanned Form")
plt.show()

# **** Paso 3: haga coincidir los puntos clave en las dos imágenes
'''
El primer paso en este proceso de coincidencia es crear una coincidencia u objeto llamando a DescriptorMatcher_create.
le pasamos a esa función DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, una medida de distancia (los descriptores de o cadena 
binaria requieren una métrica de hamming para ese objetivo). lo que hace es  Toma el descriptor de una característica 
en el primer conjunto y se compara con todas las demás características en el segundo conjunto utilizando algún cálculo 
de distancia. Y se devuelve el más cercano.

luego usamos esa coincidencia u objeto para llamar a la función de match, que luego intenta proporcionar una lista de 
las mejores coincidencias asociadas con esa lista de descriptores. tenemos una estructura de datos  que contiene la 
lista de coincidencias de los puntos clave que determinamos arriba.

Y luego, una vez que obtengamos esa lista, ordenaremos la lista en función de la distancia entre los distintos, tras lo 
que vamos a limitar al 10 por ciento superior de las coincidencias devueltas.

https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''
# Coincidir las características encontradas en ambas imágenes.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# match ()para obtener las mejores coincidencias en dos imágenes.
matches = matcher.match(descriptors1, descriptors2, None)
'''
el resutlado de la línea línea 162 es una lista de objetos DMatch. Este objeto DMatch tiene los siguientes atributos:
DMatch.distance - Distancia entre descriptores. Cuanto más bajo, mejor.
DMatch.trainIdx - Índice del descriptor en descriptores de train
DMatch.queryIdx - índice del descriptor en los descriptores de consulta
DMatch.imgIdx - Índice de la imagen de train.
'''
# ordenar las coincidencias por resultado ascendentemente
matches = sorted(matches, key=lambda x: x.distance, reverse=False)  # al ser una tupla sort no.

# Eliminar las coincidencias menos favorables, quedándonos solo con el 10%
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

'''
Y vamos a usar DrewMatches para dibujar las coincidencias en este código, puedes ver que varios puntos clave en una 
imagen coinciden los puntos clave de la otra imagen'''
# Dibujar las mejores coincidencias aportando las dos imágenes, sus puntos y las coincidencias
im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

plt.figure(figsize=[40, 10])
plt.imshow(im_matches);
plt.axis('off');
plt.title("Original Form");
plt.show()

# **** Paso 4: Encuentra la homografía
'''
¿Qué es la Homografía?
Considere dos imágenes de un plano con un libro en diferentes posiciones y distancia.  Si el libro tiene un cuadro con 
una imagen, un punto en la esquina del cuadro representa el mismo punto en las dos imágenes. En la jerga de la visión 
artificial, llamamos a estos puntos correspondientes. Una homografía es una transformación (una matriz de 3×3) que 
asigna los puntos de una imagen a los puntos correspondientes de la otra imagen.

Si conociéramos la homografía, podríamos aplicarla a todos los píxeles de una imagen para obtener una imagen 
deformada que esté alineada con la segunda imagen, es decir , puede aplicar la homografía a la primera imagen y el libro
de la primera imagen se alineará con el libro de la segunda imagen. Si conocemos 4 o más puntos correspondientes en las
dos imágenes, podemos usar la función de OpenCV findHomography para encontrar la homografía

h, status = cv2.findHomography(points1, points2)
donde, puntos1 y puntos2 son vectores/matrices de puntos correspondientes, y h es la matriz homográfica.'''

# Extraer ubicación de las buenas coincidencias
'''Crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla dimensiones cuyos elementos 
son todos ceros básicamente está creando un array de arrays con los puntos inicializados a 0, con el número de puntos 
por la longitud que tiene el objeto matches
'''
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

'''Recorre matches desde la primera posición va introducendo el valor de los puntos de los descriptores de match de 
entrenamiento y consulta'''
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Encuentra la homografía
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# ***** Paso 5: deformar la imagen
# Usar homografía para deformar la imagen
height, width, channels = im1.shape  # desmpaquetamos la dimensión de la imagen de referencia

''' la transformación de perspectva está asociada con el cambio de punto de vista. Este tipo de transformación no
conserva el paralelismo, la longitud y el ángulo pero conserva la colinealidad y la incidencia, lo que significa que 
las líneas rectas permanecerán rectas despues de la transformación. 

para ello seleccionamos 4 puntos de la imagen de entrada y asignamos esos 4 puntos a las ubicaciones deseadas en la 
imagen de salida, realizando

dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]] )
# src: imagen de entrada
# M: Matriz de transformación, en este caso usamos la homografía como esa matriz
# dsize: tamaño de la imagen de salida (ancho, alto)
# flags: método de interpolación a utilizar
https://theailearner.com/tag/cv2-warpperspective/'''

im2_reg = cv2.warpPerspective(im2, h, (width, height))
# Display results
plt.figure(figsize=[20, 10]);
plt.subplot(121);
plt.imshow(im1);
plt.axis('off');
plt.title("Original Form");
plt.subplot(122);
plt.imshow(im2_reg);
plt.axis('off');
plt.title("Scanned Form");
plt.show()

import cv2
import sys

import cv2
import glob
import matplotlib.pyplot as plt
import math

# *****************************************************
# ***** 19 Unión de imágenes y creación de panoramas
# *****************************************************
# Caracteristicas de la imagen y alineación de la imagen

# Creando panoramas usando OpenCV
'''
1. Encuentra puntos clave en todas las imágenes
2. Encuentra correspondencias por pares
3. Estimar homografías por pares
4. Refinar homografías
5. Puntada con mezcla

podemos realizar todos estos pasos con la clase stitcher, es muy similar a los pasos que se explican en  
Caracteristicas de la imagen y alineación de la imagen. stitcher es una clase que nos permite crear panoramas 
simplemente pasando una lista de imágenes.

las imágenes utilizadas para crear panoramas deben tomarse desde el mismo punto de vista Y también es importante tomar 
las fotos aproximadamente al mismo tiempo para minimizar la iluminación.
'''

# Leemos las imágenes,
'''glob incluye funciones para buscar en una ruta todos los nombres de archivos y/o directorios que coincidan con un 
determinado patrón 
glob.glob() devuelve una lista con las entradas que coincidan con el patrón especificado en pathname.
glob.glob(pathname, recursive=False)
La búsqueda se puede hacer también recursiva con el argumento recursive=True y las rutas pueden ser absolutas 
y relativas.'''
imagefiles = glob.glob("images/boat/*")
imagefiles.sort()  # ordenamos la lista obtenida
# ['boat\\boat1.jpg', 'boat\\boat2.jpg', 'boat\\boat3.jpg', 'boat\\boat4.jpg', 'boat\\boat5.jpg', 'boat\\boat6.jpg']

images = []
# recorremos la lista de imágenes y para cada imagen la leemos en color añadiendo los objeto a una lista de imágenes
for filename in imagefiles:
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(img)

num_images = len(images)

# mostramos las imágenes
plt.figure(figsize=[30,10])
num_cols = 3
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
  plt.subplot(num_rows, num_cols, i+1)
  plt.axis('off')
  plt.imshow(images[i])
plt.show()

# Stitch Images
'''
Creamos un objeto Stitcher desde la clase Stitcher_create(). Usamos ese objeto para llamar al método de stitch y 
simplemente pasamos una lista de imágenes. el resultado que obtenemos es la imagen panorámica.
El panorama de retorno incluye estas regiones negras. aquí, que son el resultado de la deformación que se requirió para 
 unir las imágenes.'''
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)
if status == 0:
  plt.figure(figsize=[30,10])
  plt.imshow(result)
plt.show()


# **************************************************
# ***** 20 Seguimiento de Objetos algoritmos opencv
# **************************************************

import zipfile
import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import urllib
'''Objetivo: dada la ubicación inicial de un objeto, realizar un seguimiento de la ubicación en fotogramas posteriores.

El seguimiento generalmente se refiere a estimar la ubicación de un objeto y predecir su ubicación en algún momento
futuro en el tiempo, y en el contexto de la visión por computadora, generalmente equivale a detectar un objeto de
interés en un video para posteriormente predecir la ubicación de ese objeto en cuadros de video subsiguientes Y logramos
esto mediante el desarrollo de un modelo de movimiento y un modelo de apariencia, usando esa información para predecir
su ubicación y futuros cuadros de video.

También podemos usar un modelo de apariencia que codifica el aspecto del objeto y buscar la región alrededor de la
ubicación predicha del modelo de movimiento para ajustar la ubicación del objeto. El modelo de movimiento es una
aproximación a la ubicación del objeto en un cuadro de video futuro, y  se usa el modelo de apariencia para afinar esa
estimación.

Como un ejemplo concreto, supongamos que estamos interesados en rastrear un objeto específico como el coche de carreras
identificado en el primer fotograma de un videoclip. Para iniciar el algoritmo de seguimiento, necesitamos especificar 
la ubicación inicial del objeto y para hacer esto, definimos un cuadro delimitador que se muestra aquí en azul, que 
consta de dos conjuntos de coordenadas de píxeles que definen las esquinas superior izquierda e inferior derecha del 
cuadro delimitador. uUna vez que el algoritmo de seguimiento se inicializa con esta información, el objetivo es realizar
un seguimiento del objeto y los cuadros de video subsiguientes al producir un cuadro delimitador en cada nuevo cuadro de
video.

En OpenCV tenemos 8 algoritmos de seguimiento disponibles:
1. BOOSTING
2. MIL
3. KCF
4. CRST
5. TLD -> Tiende a recuperarse de las oclusiones.
6. MEDIANFLOW -> Bueno para cámara lenta predecible
7. GOTRUN -> Basado en aprendizaje profundo, Más preciso
8. MOSSE -> El más rápido
'''
video_input_file_name = "videos/race_car.mp4"

# *** Definición de funciones


def drawRectangle(frame, bbox):  # Cuadro delimitador, dibujar
    p1 = (int(bbox[0]), int(bbox[1]))  # punto izquierdo superior
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # punto inferior derecho
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    # imagen, vértice sup izq, vértice inf der, color(R,G,B), grosor, tipo de línea


def displayRectangle(frame, bbox):  # Cuadro delimitador, mostrar
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()  # copiamos el fotograma
    drawRectangle(frameCopy, bbox)  # Llamamos al de arriba para dibujar el rectángulo en el fotograma
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)  # cambio de color
    plt.imshow(frameCopy); plt.axis('off')  # mostramos el fotograma


def drawText(frame, txt, location, color = (50,170,50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)  # dibujamos texto en el fotograma

'''Uno de los algoritmos es el modelo GOTURN que requiere un modelo de inferencia, que se entrena teniendo como entrada
el fotograma previo el actual, pasa por el modelo de red neuronal entrenado ( conocido como modelo de inferencia) 
Utiliza el cuadro delimitador del cuadro anterior para recortar ambas imágenes y, por lo tanto, el objeto de interés se 
encuentra en el centro de este cuadro anterior. Y obviamente, si el objeto se ha movido en el marco actual, entonces no 
estará centrado en este recorte de fotograma porque estamos usando el cuadro delimitador del cuadro anterior para 
recortar ambos fotogramas. Y luego es el trabajo del modelo de inferencia predecir cuál es el cuadro delimitador en la 
salida y da como salida el fotograma de seguimiento actual.
'''
# Descargar modelo de seguimiento (solo  GOTURN)
if not os.path.isfile('modelos/goturn/goturn.prototxt') or not os.path.isfile('modelos/goturn/goturn.caffemodel'):
    print("Downloading GOTURN model zip file")
    urllib.request.urlretrieve('https://www.dropbox.com/sh/77frbrkmf9ojfm6/AACgY7-wSfj-LIyYcOgUSZ0Ua?dl=1',
                               'GOTURN.zip')

    # descomprimir el fichero
    '''
    El método extractall() se usa para extraer paratodos los archivos presentes en el archivo zip al directorio de trabajo 
    actual. Los archivos también se pueden extraer a una ubicación diferente sin pasar por el parámetro de ruta.
    sintaxis: ZipFile.extractall(ruta_archivo, miembros=Ninguno, pwd=Ninguno)
    Parámetros:
    - file_path: ubicación donde se debe extraer el archivo comprimido, si file_path es None, el contenido del archivo zip se extraerá al directorio de trabajo actual
    - miembros: Especifica la lista de archivos a extraer, si no se especifica, se extraerán todos los archivos del zip. los miembros deben ser un subconjunto de la lista devuelta por namelist()
    - pwd: la contraseña utilizada para los archivos cifrados. Por defecto, pwd es Ninguno.
    '''
    with zipfile.ZipFile("GOTURN.zip", 'r') as zObject:
        # Extracting all the members of the zip
        # into a specific location.
        zObject.extractall(
            path=None)
    # Delete the zip file
    os.remove('GOTURN.zip')

# **** Crear la instancia de Tracker
# Configurar rastreador definiendo una lista de tracker ( "rastreadores") disponibles en la API
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']

# Cambiar el índice para cambiar el tipo de rastreador

tracker_type = tracker_types[1]
if tracker_type == 'BOOSTING':
    tracker = cv2.legacy_TrackerBoosting.create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy_TrackerCSRT.create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy_TrackerTLD.create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy_TrackerMedianFlow.create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
else:
    tracker = cv2.legacy_TrackerMOSSE.create()

# ***** Leer video de entrada y configuración de salida de video

# Leer video
'''# Estamos configurando las transmisiones de video de salida de entrada, por lo que pasamos la entrada de vídeo (el 
nombre de archivo) y creando un objeto de entrada de vídeo'''
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()  # leemos el primer fotograma del archivo
# plt.imshow(frame[..., ::-1])
# plt.show()
# Salir si no se puede abrir el video
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # capturamos del fotograma el ancho
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # capturamos del fotograma el alto

video_output_file_name = 'race_car-' + tracker_type + '.mp4'  # nombre parametrizado del archivo
'''Para escribir el vídeo, creamos un objeto de salida de vídeo que escriba los resultados del algoritmo de seguimiento 
escogido 
* explicado en "Escribir video en el disco"
'''
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'avc1'), 10, (width, height))

# ****** Definir cuadro delimitador
'''Necesitábamos encontrar un cuadro delimitador alrededor del objeto que nos interesa rastrear, y lo estamos logrando 
aquí de forma manual, Pero en la práctica, seleccionaría eso con una interfaz de usuario o tal vez usaría un algoritmo 
de detección para detectar objetos de interés para el seguimiento '''
bbox = (1300, 405, 160, 120)  # Dos conjuntos de pixeles, esquina sup izq y esquina inf der
#bbox = cv2.selectROI(frame, False)
#print(bbox)
displayRectangle(frame,bbox)

# ****** Inicializar rastreador
'''
Inicializamos el rastreador y para ello llamamos a tracker.init pásandole el primer fotograma y el cuadro delimitador'''
ok = tracker.init(frame, bbox)

# ***** Marco de lectura y objeto de seguimiento
while True:
    '''Comprobamos que existe el objeto inicializado de tracker (ok) y el fotograma, además Está leyendo el siguiente 
    fotograma del vídeo'''
    ok, frame = video.read()
    if not ok:
        break

    # Start empieza el contador
    timer = cv2.getTickCount()

    '''vamos a pasar el fotograma a la función de seguimiento o actualización que nos devolverá un cuadro delimitado 
    para el objeto detectado ( en caso de encontrarlo)'''
    ok, bbox = tracker.update(frame)

    # calcular los frames por segundo (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # dibujar la caja de seguimiento si hemos detectado el objeto
    if ok:
        drawRectangle(frame, bbox)
    else:
        # si no escribiríamos el texto de fallo en el seguimiento
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # mostrar la información calculada (en 175)
    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))

    # escribir el fotograma del vídeo
    video_out.write(frame)
'''El bucle Recorre cada cuadro en el clip de video y llama a la función de actualización del rastreador y luego anota
los fotogramas y los envía al flujo de vídeo de salida.'''

video.release()
video_out.release()

####################################################################
# 21 Detección de caras y ojos con clasificadores Haarcascade ######
####################################################################


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
# Fueron los primeros detectores de texturas ópticas de trabajo real que funcionaron bastante bien
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/haar.png)

# utiliza un concepto de ventanas correderas para básicamente deslizar estas imágenes y hace una convolución en la parte
# superior de esta imagen y extrae esas características. Tenemos muchas características de bordes, líneas, rectángulos
# y muchas otras. La combinación de esas características corresponde a un rostro, y esos clasificadores son entrenados
# para identificar las diferentes secuencias.

# Probablemente puedo describirlo como que la secuencia de valores que corresponden a la cara de una persona, al
# menos ...lo que sea que esté entrenado. Y para entrenar esto, básicamente sólo necesitas un montón de imágenes
# positivas. Son imágenes donde el objeto está presente e imágenes negativas. Así es como aprende a diferenciar cuando
# una cara está allí y cuando una cara no está allí.
# No va a prendiendo


# Apuntamos la función CascadeClassifier de OpenCV a donde esta nuestro clasificador (formato de archivo XML)
# y se almacena

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Cargamos nuestra imagen y la convertimos a escala de grises
image = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nuestro clasificador devuelve el ROI de la cara detectada como una tupla
# Almacena la coordenada superior izquierda y la coordenada inferior derecha

"""Así que hemos creado nuestro primer objeto clasificador aquí y ahora que tiene una función llamada detectMultiScale.
Aquí es donde nos alimentamos en la imagen de entrada. El primer parámetro que podemos establecer scaleFactor, 
así como un minNeighbors. Son parámetros de configuración OPCIONALES que  ajustan la sensibilidad. con ellos se puede 
conseguir más cajas en la cara y el factor de habilidad también. Depende del tipo de imagen y el tipo de cara o el 
tamaño de las caras en la imagen. Agarra la cara y extrae en una matriz."""
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Si no se detectan caras, detectMultiScale devuelve una tupla vacía
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

import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# ******************************************
# ***** 22 Escribir video en el disco
# ******************************************


source = 'videos/race_car.mp4'  # source = 0 for webcam
cap = cv2.VideoCapture(source)  # llamamos a la clase de captura de video para crear un objeto de captura de video,

# Comprobamos si se creó correctamente el objeto y está abierto
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# ****** Leer y mostrar un frame
'''
Los 3 puntos ... son una Ellipsis en Pyhton significan que puedes recibir los que sea y ya el ultimo valor (en este caso
 el ultimo array) , es iterar dándole la vuelta empezando por el ultimo para darle la vuelta a los canales,  sería igual
que [:,:, ::-1]
https://realpython.com/python-ellipsis/
'''
ret, frame = cap.read()
plt.imshow(frame[..., ::-1])
plt.show()

# Mostrar el video del archivo en jupyter
# from IPython.display import HTML
# HTML("""
# <video width=1024 controls>
#   <source src="race_car.mp4" type="video/mp4">
# </video>
# """)

# **** Mostrar el video del archivo desde consola mediante ventana
#
# win_name = 'video'
# # estamos creando una ventana con nombre, que eventualmente vamos a enviar la salida transmitida
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#
# '''ciclo while nos permitirá transmitir continuamente video desde la cámara y enviarlo a la salida a menos que el
# usuario pulse la tecla de escape.'''
# while cv2.waitKey(1) != 27:  # Escape
#     '''usa esa fuente de objeto de captura de vídeo  de captura de video para llamar al método read, que  devolverá un
#     solo cuadro de la transmisión de video, así como una variable lógica has_frame.
#     Entonces, si hay algún tipo de problema con la lectura de la transmisión de video o el acceso a la cámara,
#     has_frame sería falso y saldríamos del bucle.
#     De lo contrario, continuaríamos y llamaríamos a la función de visualización de mensajes instantáneos y abriríamos
#      kbps para enviar el video (frame) a la ventana de salida'''
#     has_frame, frame = cap.read()
#     if not has_frame:
#         break
#     cv2.imshow(win_name, frame)
#
# cap.release()
# cv2.destroyWindow(win_name)


# **** Escribir el vídeo usando OpenCV ( ojo con no haber ya recorrido el objeto de video )
'''
Para escribir el video, debe crear un objeto de videowriter con los parámetros correctos.

Sintaxis de la función
VideoWriter objeto = cv.VideoWriter (nombre de archivo, fourcc, fps, frameSize)
Parámetros
-filename: Nombre del archivo de vídeo de salida.
-fourcc: código de códec de 4 caracteres que se utiliza para comprimir los fotogramas.
 Por ejemplo, VideoWriter::fourcc('P','I','M','1') es un códec MPEG-1, VideoWriter::fourcc('M','J','P','G ') es un códec
 jpeg de movimiento, etc. La lista de códigos se puede obtener en la página Video Codecs by FOURCC. El backend FFMPEG 
 con contenedor MP4 usa de forma nativa otros valores como código fourcc: consulte ObjectType, por lo que puede recibir 
 un mensaje de advertencia de OpenCV sobre la conversión del código fourcc.
- fps: velocidad de fotogramas de la transmisión de video creada.
- frameSize: Tamaño de los fotogramas de vídeo tupla (ancho,alto).

*El tamaño del marco es importante porque deben ser las dimensiones de los marcos que tiene en la memoria que desea 
 escribir en el disco


Lo primero que vamos a hacer es usar el objeto de captura de video para llamar a este método de get(), que
nos va a recuperar las dimensiones del cuadro de video que tenemos en memoria.'''
# Se obtienen las resoluciones predeterminadas del cuadro, int() Convierte las resoluciones de float a entero
frame_width = int(cap.get(3))  # en 3 guarda el ancho
frame_height = int(cap.get(4))  # en 4 guarda el alto

# Define el códec y crea el objeto VideoWriter.
out_avi = cv2.VideoWriter('videos/race_car_out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out_mp4 = cv2.VideoWriter('videos/race_car_out.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

# Leer fotogramas y escribir en el archivo
'''Leeremos los cuadros del video del auto de carreras y escribiremos lo mismo en los dos objetos que creamos en el paso
 anterior. Deberíamos liberar los objetos después de completar la tarea.'''

# leer mientras el video se completa
while (cap.isOpened()):
  # Va capturando frame a frame
  ret, frame = cap.read()

  if ret == True:

    # Escribe cada frame en los ficheros
    out_avi.write(frame)
    out_mp4.write(frame)

  # rompe el bucle
  else:
    break

# Cuando todo esté listo, liberamos los objetos VideoCapture y VideoWriter
cap.release()
out_avi.release()
out_mp4.release()

#################################################
# 23 **Detección de vehículos y peatones** ######
#################################################
# # **Detección de vehículos y peatones**
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


# #### **Pruebas con un solo FOTOGRAMA de nuestro vídeo**
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

# #### **Prueba en nuestro VIDEO de 15 segundos**
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

'''# MOSTRAR VIDEO COLAB
from IPython.display import HTML
from base64 import b64encode

mp4 = open('walking_output.mp4' , 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()



HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
'''

# #### **Detección de vehículos en una sola imagen**
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

'''# Convertir el vídeo y mostrarlo en HTML
# no funione en ubuntu la conversión añadida salida en
# IPython.get_ipython().system('ffmpeg -i /content/cars_output.avi cars_output.mp4 -y')
#

from IPython.display import HTML
from base64 import b64encode

mp4 = open('cars_output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# no estamos mostrando la salida pero no falla
HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)'''

###########################################
# 24 Transformaciones de perspectiva ######
###########################################
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


image = cv2.imread('images/scan.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarizamos la imagen
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# sacamos el contorno EXTERNO ( RETR_EXTERNAL)
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
# Son solo grupos de píxeles
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# bucle sobre los contornos
for cnt in sorted_contours:
    #  Aproximación de cada contorno calculando el perímetro y multipicándole el accuracy recomendando (OCV)
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

#################################################################################
# 25 Representaciones de histogramas Kmeand clustering colores dominantes ######
#################################################################################
# 1. Visualizar las representaciones del histograma RGB de las imágenes
# 2. Utilizar K-Means Clustering para obtener los colores dominantes y sus proporciones en las imágenes.

# k-means -> agrupamiento para la causa dominante de una imagen.

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes

# Un histograma es básicamente un gráfico, un diagrama de barras o un grafico de líneas, Y un histograma nos da
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

plt.show()  # el básico aplanado muestra el brillo de una imagen en el que el eje vertical es el número de píxeles y
# el horizontal el rango de brillo si se ve un pico al principio significa que hay muchos píxeles oscuros y al final
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
    # Crea un histograma para los clusters basado en los píxeles de cada cluster.
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

# Así que simplemente hacemos el ajuste de puntos de K mientras creamos un sello,
# El objeto clt, que es una clave, significa objeto de agrupación, agrupa píxeles de valor similar
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

#####################################
# 26 Comparación de imágenes** ######
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
    #  matrices para decir la diferencia. 1.0 misma imagen, cuanto más baja más diferencias
    print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


# Cuando son iguales
compare(fireworks1, fireworks1)

# cuando no
compare(fireworks1, fireworks2)

compare(fireworks1, fireworks1b)

compare(fireworks2, fireworks1b)


###############################
# 27 Filtrado de colores ######
###############################
# 1. Cómo utilizar el espacio de color HSV para filtrar por color
#
# #### **Recordar el Espacio de Color HSV** ( visto en 02)
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


# Vamos a intentar quitar el camion y la tierra de la imagen dejando solo el cielo

image = cv2.imread('images/truck.jpg')

# Entonces, para hacer eso necesitamos definir un rango superior e inferior.
# definir el rango de color AZUL en HSV, en la imagen de arriba se ve que el azul va del tono 90 al 135

lower = np.array([90,0,0])  # tono , saturación , valor
upper = np.array([135,255,255])

# Convertir la imagen de RBG/BGR a HSV para poder filtrar fácilmente
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Usar inRange para capturar solo los valores entre inferior y superior, es decir crear una máscara, un umbral binario
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
# 28 Algoritmo Watershed para la segmentación de imágenes basada en marcadores ######
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
# tono en imágenes reales, se establecen marcadores antes de la "inundación" para que se realice la segmentación de
# forma correcta a raíz de lo deseado de la imagen


#  OpenCV implementó un algoritmo de cuenca basado en marcadores donde se especifica cuáles son todos los
# puntos que se van a fusionar y cuáles no. Da diferentes etiquetas para los objetos que conocemos.
# Etiquetamos la región del primer plano u objeto con un color (o intensidad),
# etiquetamos del fondo o no objeto con otro color y finalmente la región de
# que desconocemos, la etiquetamos con 0.
# Ese es nuestro marcador. A continuación, aplicar el algoritmo el marcador se actualizará con las etiquetas que le
# dimos, y los límites de los objetos tendrán un valor de -1.


# primero hacemos el gradiente morfológico para sacar el contorno de la imagen y hacemos un kernel
'''
Gradiente morfológico ** morphologyEx**
El gradiente morfológico es ligeramente diferente a las otras operaciones, porque el gradiente morfológico primero 
aplica erosión y dilatación individualmente en la imagen y luego calcula la diferencia entre la imagen erosionada y 
dilatada. 

*** La salida será un contorno de la imagen dada***

Pasos:
1. Lee la imagen
2. Binarizar la imagen.
3. Como se recomienda mantener el primer plano en blanco, estamos realizando la operación de inversión de OpenCV en la 
   imagen binarizada para que el primer plano sea blanco.
Estamos definiendo un kernel 3×3 lleno de unos
Entonces podemos hacer uso de la función Opencv cv.morphologyEx() para realizar un degradado morfológico en la imagen.
'''
# Cargar imagen
img = cv2.imread('images/water_coins.jpg')
imshow("Original image", img)

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbral usando OTSU (visto en 09)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

imshow("Thresholded", thresh)

# eliminación de ruido
kernel = np.ones((3,3), np.uint8)  # creamos una matriz 3x3 como kernel

# Aplicamos el Gradiente morfológico para sacar el contorno de la imagen
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
imshow("opening", opening)


# **** área de fondo
#  dilatantdo (visto en 10, es decir agregando píxeles a los límites de los objetos, el fondo en este caso en una
#  imagen) 3 veces la imagen
sure_bg = cv2.dilate(opening, kernel, iterations=3)
imshow("SureBG", sure_bg)

# **** Encontrar el área de primer plano,
# estamos creando los marcadores sobre las monedas
# mediante la función cv2.distanceTransform y la binarización de la imagen resultante
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
imshow("SureFG", sure_fg)


# **** Encontrar región desconocida restando el fondo al primer plano
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
imshow("unknown", unknown)


# *** Etiquetado de marcadores con connectedComponents
# connectedComponents determina la conectividad de regiones tipo blob en una imagen binaria.
ret, markers = cv2.connectedComponents(sure_fg)

# Añadir uno a todas las etiquetas para que el fondo no sea 0, sino 1
markers = markers+1

# Ahora, marca la región de unknown con cero
markers[unknown == 255] = 0

# Realiza La cuenca hidrográfica, que es un algoritmo clásico utilizado para la segmentación, es decir,
# para separar diferentes objetos en una imagen, con los marcadores establecidos
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]  # Color azul

imshow("img", img)

################################################
# 29 Substracción de fondo y primer plano ######
################################################
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


# **¿Qué es la sustracción de fondo?**

# La sustracción de fondo es una técnica de visión por ordenador en la que buscamos aislar el fondo del primer plano
# 'en movimiento'. Consideremos los vehículos que atraviesan una carretera o las personas que caminan por una acera.
#
# Suena sencillo en teoría (es decir, basta con mantener los píxeles fijos y eliminar los que cambian). Sin embargo,
# cosas como cambios en las condiciones de iluminación, sombras, etc. pueden complicar las cosas.
#
# Se han introducido varios algoritmos para este propósito. A continuación veremos dos algoritmos del módulo **bgsegm**.


# *** Algoritmo de segmentación de fondo/primer plano basado en mezclas gaussianas *****
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
# Enlace al artículo -
# https://www.researchgate.net/publication/283026260_Background_subtraction_based_on_Gaussian_mixture_models_using_color_and_depth_information


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


################################################################
# 30 Seguimiento del movimiento con Mean Shift y CAMSHIFT ######
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


##################################################
# 31 Seguimiento de objetos con flujo optico######
##################################################
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


# ## **El algoritmo de flujo óptico Lucas-Kanade**
#
# El flujo óptico es el patrón de movimiento aparente de los objetos de la imagen entre dos fotogramas consecutivos
# causado por el movimiento del objeto o de la cámara. Se trata de un campo vectorial 2D en el que cada vector es un
# vector de desplazamiento que muestra el movimiento de los puntos del primer fotograma al segundo. Considere la
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
# 32 Simple Rastreo de Objetos por Color######
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
##########################################################################################
# 33 Detección de puntos de referencia faciales con Dlib e intercambio de caras######
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


# FUNCIONES UTILIZADAS EXPLICADAS DESDE  leyendolas para entender todo el proceso correctamente
def imshow(title = "Image", image = None, size = 10): # Mostrar por pantalla la imagen
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def read_im_and_landmarks(image): # Obtienes la función simple de puntos de referencia.
    im = image
    im = cv2.resize(im,None,fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

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

    rects = detector(im, 1)  # Lo pasa por el detector. Es el de la "T"

    # resuelve los cuadros delimitadores aquí, pues solo queremos 1
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    # Es donde realmente llamamos a predictor (cogemos la imagen, una x en particular , siendo el 1º el único que
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

# get_face_mask para obtener la primera masa para que podamos extraer la cara de la imagen para ponerla en la primera
# imagen.
def get_face_mask(im, landmarks):
    """
    Se define una rutina para generar una máscara para una imagen y una matriz de puntos de referencia. Dibuja dos
    polígonos convexos en blanco: uno que rodea el área de los ojos y otro que rodea el área de la nariz y la boca.("T")
    Luego, desvanece el borde de la máscara hacia afuera en 11 píxeles. El calado ayuda a ocultar las discontinuidades
    remanentes
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

# draw_convex_hull es un casco convexo de dibujo, que nos permite mapear los puntos correctamente en tres interfaces.
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# Crea una máscara negra del tamaño de la imagen de la que saca la T con los datos de la matriz M que corresponde a la
# Otra imagen
def warp_im(im, M, dshape): # asigna la segunda imagen a la primera
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    plt.imshow(output_im)
    plt.show()
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









def swappy(image1, image2):
    # 1. Detección de puntos de referencia faciales: read_im_and_landmarks llama a get_landmarks
    # 2. Rotar, escalar y traducir la segunda imagen para que se ajuste a la primera: transformation_from_points y warp_im
    #    Luego, el resultado se puede conectar a la cv2.warpAffinefunción de OpenCV para asignar la segunda imagen a la
    #    primera: obtenemos la máscara de la "T" de la primera imagen get_face_mask, con ella llamamos a warp_im
    # 3. Ajuste del balance de color en la segunda imagen para que coincida con el de la primera: correct_colours
    # 4 .Fusión de características de la segunda imagen encima de la primera: *_POINTS, draw_convex_hull,
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

    # correcciones de color de las imágenes
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    # Luego, las máscaras se combinan en una tomando un máximo de elementos. La combinación de ambas máscaras asegura
    # que las características de la imagen 1 estén cubiertas y que las características de la imagen 2 se vean.
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    return image


# ******  A. Aplicar la detección de puntos de referencia faciales

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
FEATHER_AMOUNT = 11  # y la cantidad, que es básicamente cuánto estamos haciendo las capas de las caras.

'''Tenemos de cero a 68 puntos, cada uno de esos puntos  tiene algunos rangos, que corresponden 
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

##############################################
# 34 **Implement the Tilt Shift Effect**######
##############################################
# Tilt Shift  es un efecto que toma nuestra imagen estándar normal, como el paisaje de una ciudad o de arriba hacia
# abajo, algo que es bonito y lo hace parecer parece que es un modelo miniaturizado, enfocándose en ciertas áreas y
# luego difuminando otras.

# - En esta lección veremos algo de código que genera nuestro efecto Titl Shift en nuestras imágenes de ejemplo
# Fuente - https://github.com/powersj/tilt-shift


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
from matplotlib import pyplot as plt
import cv2
import math
import os
import numpy as np
import scipy.signal
import shutil

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# #### **Nuestras funciones para implementar Tilt Shift**
"""Script para mezclar dos imágenes"""

def get_images(sourcefolder):
    """Carga las fotos de cada carpeta y las devuelve"""
    filenames = os.listdir(sourcefolder)

    # Según el nombre del fichero carga la foto
    for photo in filenames:
        black_img = cv2.imread('images/images_tilt/original/' + photo)
        white_img = cv2.imread('images/images_tilt/blur/' + photo)
        mask_img = cv2.imread('images/images_tilt/mask/' + photo)

        # comprueba la carga de las imágenes
        if mask_img is None:
            print('Oops! There is no mask of image: ', photo)
            continue
        if white_img is None:
            print('Oops! There is no blurred version of image: ', photo)
            continue

        # Comprueba que el tamaño de la imagen, su imagen difuminada y su máscara sean iguales
        assert black_img.shape == white_img.shape, \
            "Error - los tamaños de orignal y blur no son iguales"

        assert black_img.shape == mask_img.shape, \
            "Error - los tamaños del original y la máscara no son iguales"

        print(photo)
        yield photo, white_img, black_img, mask_img

def run_blend(black_image, white_image, mask):
    """ Esta función administra la mezcla de las dos imágenes según la máscara. Asume que todas las imágenes son de
    tipo float, y devuelve un tipo float. Recordar que la función ha recibido un solo canal de cada imagen
    """
    # Calcula automáticamente el tamaño
    min_size = min(black_image.shape)
    # calcula la profundidad, al menos 16x16 en el nivel más alto.
    depth = int(math.floor(math.log(min_size, 2))) - 4

    # llama a gauss_pyramid, para construir una pirámide a partir de la imagen
    gauss_pyr_mask = gauss_pyramid(mask, depth)
    gauss_pyr_black = gauss_pyramid(black_image, depth)
    gauss_pyr_white = gauss_pyramid(white_image, depth)

    # Construye una pirámide laplaciana, reduce la imagen de una mayor a otra menor perdiendo poca calidad
    lapl_pyr_black = lapl_pyramid(gauss_pyr_black)
    lapl_pyr_white = lapl_pyramid(gauss_pyr_white)

    # Mezcla las dos pirámides laplacianas ponderándolas según la máscara gaussiana
    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    # Colapsa una pirámide de entrada, ( une las imágenes de la pirámide en una expandiendo y añadiendo)
    outimg = collapse(outpyr)

    # la mezcla a veces resulta en números ligeramente fuera de límites.
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)

    # *La imagen es de 1 canal*, devuelve el resultado
    return outimg


def gauss_pyramid(image, levels):
    """ Construye una pirámide a partir de la imagen reduciéndola por el número de
    niveles introducidos en la entrada.

    Nota: Necesitas usar tu función reduce en esta función para generar la salida.
    salida.
    Args:
      image (numpy.ndarray): Una imagen en escala de grises de dimensión (r,c) y dtype
      float.
      levels (uint8): Un entero positivo que especifica el número de
                    reducciones que debe hacer. Así, si levels = 0, debe
                    devolver una lista que contenga sólo la imagen de entrada. Si
                    levels = 1, debe hacer una reducción.
                    len(salida) = niveles + 1
    Devuelve:
      output (lista): Una lista de matrices de dtype np.float. El primer elemento de
                la lista (output[0]) es la capa 0 de la pirámide (la imagen
                output[1] es la capa 1 de la pirámide (la imagen reducida
                una vez), etc. Ya hemos incluido la imagen original en
                la matriz de salida. Las matrices son de tipo numpy.ndarray.
    """
    output = [image]
    for level in range(levels):
        output.append(reduce_img(output[level]))
    return output


def lapl_pyramid(gauss_pyr):
    """ Construye una pirámide laplaciana a partir de la pirámide gaussiana, de altura
    niveles. Reduce la imagen de una mayor a otra menor perdiendo poca calidad ( el algoritmo de wats up se basa en
    este vamos.

    Nota: Debes usar tu función expand en esta función para generar la
    salida. La pirámide gaussiana que se pasa es la salida de su función
    gauss_pyramid.

    Args:
      gauss_pyr (lista): Una pirámide gaussiana devuelta por la función gauss_pyramid
                     gauss_pyramid. Es una lista de elementos numpy.ndarray.

    Devuelve:
      output (list): Una pirámide Laplaciana del mismo tamaño que gauss_pyr. Esta pirámide
                   pirámide debe ser representada de la misma manera que guassPyr,
                   como una lista de matrices. Cada elemento de la lista
                   corresponde a una capa de la pirámide de Laplaciano, que contiene
                   la diferencia entre dos capas de la pirámide de Gauss.

           output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])

           Nota: El último elemento de la salida debe ser idéntico a la última
           capa de la pirámide de entrada ya que no se puede restar más.

    Nota: A veces, el tamaño de la imagen expandida será mayor que la capa
    capa dada. Debe recortar la imagen expandida para que coincida en forma con
    con la capa dada.

    Por ejemplo, si mi capa es de tamaño 5x7, reduciendo y expandiendo resultará
    una imagen de tamaño 6x8. En este caso, recorte la capa expandida a 5x7.
    """
    output = []
    # revisa las listas, pero ignora el último elemento ya que no puede ser
    # restado
    for image1, image2 in zip(gauss_pyr[:-1], gauss_pyr[1:]):
        # añadir la diferencia restada
        # expandir y enlazar la segunda imagen en función de las dimensiones de la primera
        output.append(
            image1 - expand(image2)[:image1.shape[0], :image1.shape[1]])

    # añade ahora el último elemento
    plt.imshow(gauss_pyr[-1])
    output.append(gauss_pyr[-1])

    return output


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    """ Mezcla las dos pirámides laplacianas ponderándolas según la máscara gaussiana. Máscara gaussiana.
    Args:
      lapl_pyr_white (lista): Una pirámide laplaciana de una imagen, construida por su función lapl_pyramid.
      lapl_pyr_black (lista): Una pirámide de Laplaciano de otra imagen, construida por la función lapl_pyramid.
                        construida por su función lapl_pyramid.
      gauss_pyr_mask (lista): Una pirámide gaussiana de la máscara. Cada valor está en el rango de [0, 1].

    Las pirámides tendrán el mismo número de niveles. Además, se garantiza que cada capa
    tiene garantizada la misma forma que los niveles anteriores.

    Debe devolver una pirámide Laplaciana que tenga las mismas dimensiones que las
    pirámides de entrada. Cada capa debe ser una mezcla alfa de las correspondientes
    capas de las pirámides de entrada, ponderadas por la máscara gaussiana. Esto significa que
    siguiente cálculo para cada capa de la pirámide:
      salida[i, j] = máscara_actual[i, j] * imagen_blanca[i, j] +
                   (1 - máscara_actual[i, j]) * imagen_negra[i, j]
    Por lo tanto:
      Los píxeles en los que máscara_actual == 1 deben tomarse completamente de la imagen blanca.
      blanca.
      Los píxeles en los que máscara_actual == 0 deben tomarse completamente de la imagen negra.
      negra.

    Nota: máscara_actual, imagen_blanca e imagen_negra son variables que se refieren a la imagen de la capa actual que
    estamos viendo a la imagen de la capa actual que estamos viendo. Este
    cálculo para cada capa de la pirámide.
    """

    blended_pyr = []
    for lapl_white, lapl_black, gauss_mask in \
            zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
        blended_pyr.append(gauss_mask * lapl_white +
                           (1 - gauss_mask) * lapl_black)

    plt.imshow(blended_pyr[0])
    plt.show()
    return blended_pyr


def collapse(pyramid):
    """ Colapsa una pirámide de entrada.

    Args:
      pyramid (list): Una lista de imágenes numpy.ndarray. Se puede asumir que la entrada
            se toma de blend() o lapl_pyramid().

    Devuelve:
      output(numpy.ndarray): Una imagen de la misma forma que la capa base de
            la pirámide y dtype float.

    Plantea este problema de la siguiente manera, empieza por la capa más pequeña de la
    pirámide. Expande la capa más pequeña y añádela a la segunda capa más pequeña.
    más pequeña. Luego, expanda la segunda a la capa más pequeña, y continúe el proceso
    hasta llegar a la imagen más grande. Este es el resultado.

        Nota: a veces expandir devolverá una imagen más grande que la siguiente
    siguiente. En este caso, debe recortar la imagen expandida hasta el tamaño de la siguiente capa.
    la siguiente capa. Mira en numpy slicing / lee nuestro README para hacer esto
    fácilmente.

    Por ejemplo, expandir una capa de tamaño 3x4 resultará en una imagen de tamaño
    6x8. Si la siguiente capa es de tamaño 5x7, recorta la imagen expandida a tamaño 5x7.
    """
    output = pyramid[-1]
    for image in reversed(pyramid[:-1]):
        output = image + expand(output)[:image.shape[0], :image.shape[1]]

    return output

def generating_kernel(parameter):
    """ Devuelve un kernel generador 5x5 basado en un parámetro de entrada.
     Nota: Esta función se proporciona para ti, no la cambies.
     Args:
         parameter (float): Rango de valor: [0, 1].
     output:
        numpy.ndarray: Un núcleo de 5x5.
     """
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                       0.25, 0.25 - parameter / 2.0])
    return np.outer(kernel, kernel)


def reduce_img(image):
    """ Convoluciona la imagen de entrada con un kernel generador de parámetro de 0.4
        y luego reducir su anchura y altura por dos.
        Puedes utilizar cualquiera / todas las funciones para convolucionar y reducir la imagen, aunque
        las conferencias han recomendado métodos que aconsejamos ya que hay un montón de
        de piezas en esta tarea que necesitan trabajar 'justo a la derecha'.
        Args:
        image (numpy.ndarray): una imagen en escala de grises de forma (r, c)
        Devuelve:
        output (numpy.ndarray): una imagen de la forma (ceil(r/2), ceil(c/2))
          Por ejemplo, si la entrada es 5x7, la salida será 3x4.

    """
    # según las instrucciones, utilice 0.4 para la generación del kernel
    kernel = generating_kernel(0.4)

    # usa convolve2d con la imagen y el kernel enviados
    output = scipy.signal.convolve2d(image, kernel, 'same')

    # devuelve cada dos líneas y filas
    return output[:output.shape[0]:2, :output.shape[1]:2]


def expand(image):
    """ Expandir la imagen al doble de tamaño y luego convolucionarla con un
    kernel generador con un parámetro de 0.4.

    Deberías aumentar el tamaño de la imagen y luego convolucionarla con un kernel generador
    kernel generador de a = 0,4.

    Por último, multiplique la imagen de salida por un factor de 4 para volver a escalarla.
    escala. Si no hace esto (y le recomiendo que lo pruebe sin
    esto) verá que sus imágenes se oscurecen al aplicar la convolución.
    Por favor, explica por qué ocurre esto en tu PDF de presentación.

    Por favor, consulte las conferencias y readme para una discusión más a fondo de
    cómo abordar la función de expansión.

    Puede utilizar cualquier / todas las funciones para convolucionar y reducir la imagen, aunque
    las conferencias han recomendado métodos que aconsejamos ya que hay un montón de
    piezas de esta tarea que tienen que funcionar "a la perfección".

    Args:
    image (numpy.ndarray): una imagen en escala de grises de forma (r, c)

    Devuelve:
    output (numpy.ndarray): una imagen de la forma (2*r, 2*c)
    """
    # según las instrucciones, usa 0.4 para la generación del kernel
    kernel = generating_kernel(0.4)

    # hacer un nuevo array del doble de tamaño, asignar valores iniciales
    output = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    output[:output.shape[0]:2, :output.shape[1]:2] = image

    # usa convolve2d para rellenar el resto
    # multiplicar por 4 por instrucciones para volver a escalar
    output = scipy.signal.convolve2d(output, kernel, 'same') * 4
    return output



def main():
    """Dadas las dos imágenes, mézclalas según la máscara"""

    sourcefolder = 'images/images_tilt/original'
    outfolder = 'images/images_tilt/output'

    # si no encuentra el directorio de salida, lo crea
    if os.path.isdir(outfolder):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)

    # Mediante el uso de la función get_images, usándola como un Iterador gracias a Yield,
    # va mos a cargar de cada imagen original su máscara y su blur, cargándola en las 3 variables
    for photo, white_img, black_img, mask_img in get_images(sourcefolder):
        imshow("Original Image", black_img)
        print("...applying blending")
        black_img = black_img.astype(float)
        white_img = white_img.astype(float)
        mask_img = mask_img.astype(float) / 255

        # inicializa la salida
        out_layers = []

        # para cada canal de color (RGB) llama a run_blend (mezcla las imágenes según la máscara)
        for channel in range(3):
            outimg = run_blend(black_img[:, :, channel],
                               white_img[:, :, channel],
                               mask_img[:, :, channel])
            out_layers.append(outimg)

        # la salida es la fusión de cada canal ya tratado
        outimg = cv2.merge(out_layers)

        # escribe en la carpeta de salida la imagen ya calculada
        cv2.imwrite(os.path.join(outfolder, photo), outimg)
        imshow("Tilt Shift Effect", outimg)
        print('...[DONE]')

if __name__ == "__main__":
    main()


#############################################################
# 35 **Algoritmo GrabCut para la eliminación del fondo**#####
#############################################################
# Es un algoritmo de segmentación.
# - En esta lección vamos a utilizar el algoritmo GrabCut para la eliminación de fondo

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import dlib
import sys
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



### **¿Cómo funciona Grab Cut?**
#
# - **El usuario introduce el rectángulo**. Todo lo que esté fuera de este rectángulo se tomará como fondo. Todo dentro
#   del rectángulo es desconocido.
# - El algoritmo etiqueta los píxeles de primer plano y de fondo (o los etiqueta).
#   A continuación, se utiliza un modelo de mezcla gaussiana (GMM) para modelar el primer plano y el fondo.
# - Dependiendo de los datos que hemos dado, GMM aprende y crea una nueva distribución de píxeles. Es decir, los
#   **píxeles desconocidos se etiquetan como probable primer plano o probable fondo** dependiendo de su relación con los
#   otros píxeles etiquetados en términos de estadísticas de color (es como la agrupación).

# A partir de esta distribución de píxeles se construye un gráfico. Los nodos del gráfico son píxeles. Se añaden dos
# nodos adicionales, Source node y Sink node. Cada píxel en primer plano está conectado al nodo Fuente y cada píxel en
# segundo plano está conectado al nodo Sumidero.
# Los pesos de las aristas que conectan los píxeles al nodo origen/nodo final se definen por la probabilidad de que un
# píxel esté en primer plano/fondo. Los pesos entre los píxeles se definen por la información de borde o similitud de
# píxeles. Si hay una gran diferencia en el color de los píxeles, el borde entre ellos tendrá un peso bajo.

# A continuación, se utiliza un algoritmo de corte mínimo para segmentar el gráfico. Corta el gráfico en dos separando
# el nodo de origen y el nodo de destino con la función de coste mínimo. La función de coste es la suma de todos
# los pesos de las aristas que se cortan. Tras el corte, todos los píxeles conectados al nodo origen pasan a primer
# plano y los conectados al nodo sumidero pasan a segundo plano.
# El proceso continúa hasta que la clasificación converge.
#
# ![](https://docs.opencv.org/3.4/grabcut_scheme.jpg)
# Paper - http://dl.acm.org/citation.cfm?id=1015720
# Más información - https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html


# Cargar nuestra imagen
image = cv2.imread('images/woman.jpeg')
copy = image.copy()

# Crear una máscara (de ceros uint8 tipo de datos) que es el mismo tamaño (ancho, alto) como nuestra imagen original
mask = np.zeros(image.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Necesita ser establecido manualmente o seleccionado con cv2.selectROI() ( seleccionarlo vamos)

x1, y1, x2, y2 = 190, 70, 350, 310
start = (x1, y1)
end = (x2, y2)

# El formato es X,Y,W,H
rect = (x1, y1, x2-x1, y2-y1)

# MODIFICADO
rect = cv2.selectROI(copy)  # PARA SELECCIONARLO NOSOTROS

# Mostrar rectángulo
cv2.rectangle(copy, start, end, (0,0,255), 3)
imshow("Input Image", copy) # se muestra el por defecto pero el algoritmo coge el "nuevo"



# #### **Argumentos de recorte**
#
# - **img** - Imagen de entrada
# - **mask** - Es una imagen máscara donde especificamos qué áreas son fondo, primer plano o probable fondo/primer plano
#              , etc. Se hace mediante las siguientes banderas, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, o
#              simplemente pasando 0,1,2,3 a la imagen.
# - **rec**t - Son las coordenadas de un rectángulo que incluye el objeto en primer plano en el formato (x,y,w,h)
# - **bdgModel, fgdModel** - Son matrices utilizadas por el algoritmo internamente. Basta con crear dos arrays de tipo
#                            np.float64 de tamaño cero (1,65).
# - **iterCount** - Número de iteraciones que debe ejecutar el algoritmo.
# - **mode** - Debe ser cv.GC_INIT_WITH_RECT o cv.GC_INIT_WITH_MASK o combinado que decide si estamos dibujando
#              rectángulo o trazos de retoque final.

# Deja que el algoritmo se ejecute durante 5 iteraciones. El modo debe ser cv.GC_INIT_WITH_RECT ya que estamos usando
# rectángulo.
# Grabcut modifica la imagen de máscara.
# En la nueva imagen de máscara, los píxeles serán marcados con cuatro banderas que denotan fondo/primer plano como
# se especificó anteriormente.

cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Así que modificamos la máscara de tal manera que todos los 0-píxeles y 2-píxeles se ponen a 0 (es decir, de fondo)
# y todos los 1-píxeles y 3-píxeles se ponen a 1 (es decir, los píxeles de primer plano).
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Ahora nuestra máscara final está lista. Sólo hay que multiplicarla con la imagen de entrada para obtener la imagen
# segmentada.
image = image * mask2[:,:,np.newaxis]

imshow("Mask", mask * 80)
imshow("Mask2", mask2 * 255)
imshow("Image", image)


#########################################################################
# 35 **Reconocimiento Óptico de Caracteres con PyTesseract & EASY OCR**#####
#########################################################################

# - En esta lección implementaremos OCR en algunas imágenes usando PyTesseract
#
# ![](https://miro.medium.com/max/1400/1*X7RfC5wOZ-Gsoo95Ez1FvQ.png)
# Fuente - https://medium.com/@balaajip/optical-character-recognition-99aba2dad314

# #### **Install PyTesseract **
#  librería de código abierto, a alto nivel toma una entrada de imagen, reconoce el texto de la misma, lo detecta ,
#  trata y limpia devolviendo un texto como una cadena
'''# 
get_ipython().system('sudo apt install tesseract-ocr')
get_ipython().system('pip install pytesseract')
get_ipython().system('pip install easyocr')'''


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from pytesseract import Output
from easyocr import Reader
import pandas as pd
import time

# importamos las librerías instaladas en sistema
pytesseract.pytesseract.tesseract_cmd = (
    r'/usr/bin/tesseract'
)

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



# ## **Nuestra primera prueba de OCR**
img = cv2.imread('images/OCR Samples/OCR1.png')
imshow("Input Image", img)

# Ejecutar nuestra imagen a través de PyTesseract, esta línea es la que realiza la detección, procesamiento y devuelve
# el texto de la imagen
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # PyTesseract Extracted: Welcome to OCR


# ## **¿Funciona el texto blanco sobre fondo negro?**
img = cv2.imread('images/OCR Samples/OCR2.png')
imshow("Input Image", img)
# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # PyTesseract Extracted: Welcome to OCR


# ## **¿Qué pasa con los fondos más desordenados?**
img = cv2.imread('images/OCR Samples/OCR3.png')
imshow("Input Image", img)

# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # no funciona


# ## **¿Qué tal un escaneo de la vida real?**
img = cv2.imread('images/OCR Samples/scan2.jpeg')
imshow("Input Image", img, size = 48)

# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)
print("PyTesseract Extracted: {}".format(output_txt))  # funciona hasta cierto punto


# **Necesitamos limpiar nuestras imágenes**
image = cv2.imread('images/OCR Samples/scan2.jpeg')
imshow("Input Image", image, size = 48)

# Obtenemos el componente Valor del espacio de color HSV
# luego aplicamos umbralización adaptativa a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Aplicar la operación umbral
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh, size = 48)

output_txt = pytesseract.image_to_string(thresh)
print("PyTesseract Extracted: {}".format(output_txt))  # se ve mucho mejor


# ### **Umbralizar ayuda mucho**
# Típicamente un buen pipeline de preprocesamiento para reconocimiento OCR contendrá algunos o más de los siguientes
# procesos:
# 1. Desenfoque
# 2. Umbralización
# 3. Desenfoque
# 4. Dilatación/Erosión/Apertura/Cierre
# 5. 5. Eliminación de ruido

### **Dibujemos sobre regiones reconocidas por PyTesseract**


image = cv2.imread('images/Receipt-woolworth.jpg')

# Obtenemos el componente Valor del espacio de color HSV
# luego aplicamos umbralización adaptativa a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Aplicar la operación umbral
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)

output_txt = pytesseract.image_to_string(thresh)
print("PyTesseract Extracted: {}".format(output_txt))




d = pytesseract.image_to_data(thresh, output_type = Output.DICT)
print(d.keys())  # dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top',
#                             'width', 'height', 'conf', 'text'])
# Usando este diccionario, podemos obtener de cada palabra detectada en la imagen, la información (con coordenadas) de
# su cuadro delimitador, el texto que contiene y las puntuaciones de confianza de cada una.
#
n_boxes = len(d['text'])

# recorre los elementos (etiquetas del diccionario)
for i in range(n_boxes):
    # si la confianza obtenida es mayor de 60
    if int(d['conf'][i]) > 60:
        # extrae de esa etiqueta  las coordenadas y el tamaño para poder dibujar su cuadro delimitador
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # dibuja el rectángula verde
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# muestra la imagen
imshow('Output', image, size = 12)


# ## **EASY OCR**
# esta librería funciona mejor que la anterior, pero necesita más capacidad de procesamiento, siendo lento en CPU pero
# pudiendo aprovechar la GPU
# ### **Instalar OpenCV antiguo (EasyOCR no es compatible con el último OpenCV aquí en Colab)**
'''get_ipython().system('pip uninstall opencv-python -y')
get_ipython().system('pip install opencv-python-headless==4.1.2.30')
'''

# ## **Detectar Texto en Imagen y Mostrar nuestra Imagen de Entrada**

# cargar la imagen de entrada desde el disco
image = cv2.imread("images/whatsapp_conv.jpeg")
imshow("Original Image", image, size = 12)

# OCR de la imagen de entrada utilizando EasyOCR
print("Detecting and OCR'ing text from input image...")
# reader importado de EasyOcr, va a buscar texto en inglés ['en'] y vamos a usarlo sin GPU
# se descarga y utiliza automáticamente el modelo
reader = Reader(['en'], gpu = False)


ts = time.time()
results = reader.readtext(image) # introducimos en results todo el texto detectado ( en "cajas")
# [ ([[24, 12], [192, 12], [192, 38], [24, 38]], 'bmobile _ill < 82', 0.12457801531641248)
#   ([[396, 12], [510, 12], [510, 38], [396, 38]], '"\'0 ^ (50%', 0.33694383567965347) ...

te = time.time()  # usamos time y para comprobar cuanto tiempo ha tardado en realizar el procesamiento de la imagen
td = te - ts
print(f'Completed in {td} seconds')  # 7.667776107788086 seconds


# ## **Mostrar texto superpuesto a nuestra imagen**
all_text = []

# iterar sobre el texto extraído
for (bbox, text, prob) in results:
    # mostrar el texto OCR y la probabilidad asociada de que sea texto
    print(f" Probability of Text: {prob*100:.3f}% OCR'd Text: {text}")

    # obtener las coordenadas del cuadro delimitador
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # Elimina los caracteres no ASCII del texto para que, uniendo el no eliminado con join y usando el codepoint de
    # los caracteres detectados
    # podamos dibujar el recuadro que rodea el texto superpuesto a la imagen original
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    all_text.append(text)
    cv2.rectangle(image, tl, br, (255, 0, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# mostrar la imagen de salida
imshow("OCR'd Image", image, size = 12)


# ## **Ejecutar en nuestro WoolWorth Reciept**

def clean_text(text):
    # elimina el texto no ASCII para que podamos dibujar el texto en la imagen
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

image = cv2.imread('images/Receipt-woolworth.jpg')

reader = Reader(["en","ar"], gpu=False)
results = reader.readtext(image)

# bucle sobre los resultados
for (bbox, text, prob) in results:
    # mostrar el texto OCR y la probabilidad asociada
    print("[INFO] {:.4f}: {}".format(prob, text))

    # descomprimir el cuadro delimitador
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # limpia el texto y dibuja el recuadro que lo rodea a lo largo de
    text = clean_text(text)
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


#Aplicar la operación umbral
#thresh = (V > T).astype("uint8") * 255
imshow("EASY OCR", image)
print("EASY OCR Extracted: {}".format(text))
'''
...
[INFO] 0.0220: 900
[INFO] 0.9213: Woolworths
[INFO] 0.2355: The fregh food
[INFO] 0.6164: VICIURIA HARBOUR PH:  0383476527
[INFO] 0.7185: Store Hanager
[INFO] 0.6325: i$ ٥avid
[INFO] 0.3334: WUULWURIHS TAX INVOICE
[INFO] 0.6674: ABN 88 000 014 675
...EASY OCR Extracted: .50/k9
'''


#########################################################################
# 37 **Generación y lectura de códigos de barras**#####
#########################################################################

# - In this lesson we'll to create barcodes of various standards as well reading what's on them.

# In[1]:


'''# Our Setup, Import Libaries, Create our Imshow Function and Download our Images

get_ipython().system('pip install python-barcode[images]')
get_ipython().system('pip install qrcode')
get_ipython().system('apt install libzbar0')
get_ipython().system('pip install pyzbar')
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from barcode import EAN13
from barcode.writer import ImageWriter

# Define nuestra función imshow

def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ## **Generación de códigos de barras**
# los códigos de barras son solo representaciones lineales del símbolo de la información codificada que podemos
# decodificar de un texto, dígitos o cualquier tipo de información relativa a los códigos de barras de las tiendas.
#
# O podemos almacenar información de manera efectiva.
# Vamos a generar códigos de barras usando nuestro paquete python-barcode.
# Formatos soportados
# En el momento de escribir esto, este paquete soporta los siguientes formatos:
# - EAN-8
# - EAN-13
# - EAN-14
# UPC-A
# - JAN
# - ISBN-10
# - ISBN-13
# - ISSN
# Código 39
# - Código 128
# - PZN

with open('images/barcode.png', 'wb') as f:
    # el número es una entidad para guardar el resultado, podemos introducir el que queramos
    EAN13('123456789102', writer=ImageWriter()).write(f)

barcode = cv2.imread("images/barcode.png")
imshow("Barcode", barcode)


# ## **Generación de Códigos QR**
# Vamos a generar Códigos QR usando nuestro paquete qrcode.

# Un código QR (abreviatura de Quick Response code) es un tipo de código de barras matricial (o código de barras
# bidimensional) diseñado por primera vez en 1994 para la industria del automóvil en Japón. Un código de barras es una
# etiqueta óptica legible por máquina que contiene información sobre el artículo al que está adherido. En la práctica,
# los códigos QR suelen contener datos para un localizador, identificador o rastreador que apunta a un sitio web o una
# aplicación. Un código QR utiliza cuatro modos de codificación estandarizados (numérico, alfanumérico, byte/binario y
# kanji) para almacenar datos de forma eficiente; también se pueden utilizar extensiones.
#
# Un código QR consiste en cuadrados negros dispuestos en una cuadrícula cuadrada sobre un fondo blanco, que pueden ser
# leídos por un dispositivo de imagen como una cámara, y procesados utilizando la corrección de errores Reed-Solomon
# hasta que la imagen puede ser interpretada adecuadamente. A continuación, se extraen los datos necesarios de los
# patrones presentes en los componentes horizontal y vertical de la imagen.
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/QR_Code_Structure_Example_3.svg/800px-QR_Code_Structure_Example_3.svg.png)

import qrcode
from PIL import Image  # librería similar a OpenCV pero no tan extensa

# **Configuración de los códigos QR**:
#
# - version - Controla el tamaño del Código QR. Acepta un número entero de 1 a 40. La versión 1 consiste en una matriz
#             de 21 x 21.
# - error_correction - Controla la corrección de errores utilizada para el Código QR.
# - box_size - Controla el número de píxeles de cada caja del código QR.
# - border - Controla el grosor del borde de las cajas. El valor por defecto es 4, que es también el valor mínimo según
#            la especificación.
#
# Hay 4 constantes disponibles para error_correction. Cuanto mayor sea la corrección de errores, mejor será.
# pero a mayor corrección de errores menos información puedes almacenar en el codigo.
# - ERROR_CORRECT_L - Alrededor del 7% o menos errores pueden ser corregidos.
# - ERROR_CORRECT_M - Alrededor del 15% o menos errores pueden ser corregidos. Este es el valor por defecto.
# ERROR_CORRECT_Q - Se pueden corregir un 25% o menos de errores.
# ERROR_CORRECT_H - Alrededor del 30% o menos de errores pueden ser corregidos.

# creando el objeto QR code con la configuración
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

# Vamos a generar un código QR con la web de openCV
qr.add_data("https://wwww.opencv.org")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("images/qrcode.png")

qrcode = cv2.imread("images/qrcode.png")
imshow("QR Code", qrcode, size = 8)




# ## **Descifrar códigos QR**
from pyzbar.pyzbar import decode
from PIL import Image
img = Image.open('images/qrcode.png')
# decodifica la imagen con la función decode del módulo pyzbar de la librería pyzbar
result = decode(img)
for i in result:
    print(i.data.decode("utf-8"))  # https://wwww.opencv.org


# ### **Detección de códigos QR**
from pyzbar.pyzbar import decode

image = cv2.imread("images/1DwED.jpg")

# Detectar y decodificar el qrcode
codes = decode(image)

# bucle sobre los códigos de barras detectados
for bc in codes:
  # Obtener los rectángulos coordiantes para la colocación del texto
  (x, y, w, h) = bc.rect
  print(bc.polygon)
  pt1,pt2,pt3,pt4 = bc.polygon

  # Dibuja una caja delimitadora sobre nuestro código QR detectado
  pts = np.array( [[pt1.x,pt1.y], [pt2.x,pt2.y], [pt3.x,pt3.y], [pt4.x,pt4.y]], np.int32)
  pts = pts.reshape((-1,1,2))
  cv2.polylines(image, [pts], True, (0,0,255), 3)

  # extraer los datos de información de la cadena y el tipo de nuestro objeto
  barcode_text = bc.data.decode()
  barcode_type = bc.type

  # mostrar nuestro
  text = "{} ({})".format(barcode_text, barcode_type)
  cv2.putText(image, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  cv2.putText(image, barcode_type, (x+w, y+h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  print("QR Code revealed: {}".format(text))

# mostrar nuestra salida
imshow("QR Scanner", image, size = 12)



image = cv2.imread("images/1024px-ISBN.jpg")

# Detectar y decodificar el qrcode
barcodes = decode(image)

# bucle sobre los códigos de barras detectados
for bc in barcodes:
  # Obtener los rectángulos coordiantes para la colocación del texto
  (x, y, w, h) = bc.rect
  cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

  # extraer los datos de información de la cadena y el tipo de nuestro objeto
  barcode_text = bc.data.decode()
  barcode_type = bc.type

  # Mostrar nuestro
  text = "{} ({})".format(barcode_text, barcode_type)
  cv2.putText(image, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  cv2.putText(image, barcode_type, (x+w, y+h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  print("Barcode revealed: {}".format(barcode_text))
  print("Barcode revealed: {}".format(barcode_text))

# mostrar nuestra salida
imshow("QR Scanner", image, size = 16)



#########################################################################
# 38 **YOLOv3 usando cv2.dnn.readNetFrom()**#####
#########################################################################
# tutorial oficial en https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
# https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
# https://neptune.ai/blog/object-detection-with-yolo-hands-on-tutorial
# YOLO es uno de los algoritmos de detección de objetos en tiempo real más rápidos (45 cuadros por segundo) en
# comparación con la familia R-CNN (R-CNN, Fast R-CNN, Faster R-CNN, etc.)
# La familia de algoritmos R-CNN utiliza regiones para localizar los objetos en las imágenes, lo que significa que el
# modelo se aplica a varias regiones y las regiones de la imagen con una puntuación alta se consideran objetos
# detectados. Pero YOLO sigue un enfoque completamente diferente. En lugar de seleccionar algunas regiones, aplica una
# red neuronal a toda la imagen para predecir los cuadros delimitadores y sus probabilidades.

# ####**En esta lección aprenderemos a cargar un Modelo YOLOV3 pre-entrenado y usar OpenCV para ejecutar inferencias
# sobre algunas imágenes**
# YOLOV -> detector de objetos
# importar los paquetes necesarios
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


# Define nuestra función imshow
def imshow(title="Image", image=None, size=8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ## **Detección de Objetos YOLO**
# ![](https://opencv-tutorial.readthedocs.io/en/latest/_images/yolo1_net.png)
# Pasos necesarios
# 1. Usar pesos YOLOV3 preentrenados (237MB)- https://pjreddie.com/media/files/yolov3.weights
# 2. Crear nuestro objeto blob que es nuestro modelo cargado
# 3. Establecer el backend que ejecuta el modelo

# Cargar las etiquetas de clase COCO con las que se ha entrenado nuestro modelo YOLO
# coco es un tipo de conjunto de datos de objetos comunes
# ImageNet es un conjunto de datos clasificado que ha demostrado se invaluable para la investigación de computer vision
# contiene los nombres de los diferentes objetos que nuestro modelo ha sido entrenado para identificar.

labelsPath = "modelos/YOLO3/yolo/coco.names"
'''
person
bicycle
car
motorbike'''
LABELS = open(labelsPath).read().strip().split("\n")

# Ahora necesitamos inicializar una lista de colores para representar cada posible etiqueta de clase
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("Loading YOLO weights...")

# almacena todos los pesos para el modelo
weights_path = "modelos/YOLO3/yolo/yolov3.weights"

# Es la estructura del modelo, define toda la estructura del modelo YOLOV que está codificada en su campo de cisión
cfg_path = "modelos/YOLO3/yolo/yolov3.cfg"

# Crear nuestro objeto blob
'''OpenCV tiene varias funciones de conveniencia que nos permiten leer y pre-entrenar modelos que fueron entrenados 
usando marcos de trabajo como NetFromDarknet y pytorch que son marcos de aprendizaje profundo que permiten diseñar y 
entrenar redes neuronales. Además OpenCV tiene una funcionalidad  integrada para usar redes pre-entrenadas para realizar
inferencias ( es decir, no podemos usare OpenCV  para entrenar una red neuronal, pero puede usarlo para realizar 
inferencias en una red entrenada)

la función cv2.dnn.readNetFromDarknet es una función diseñada específicamente para cargar un modelo . necesita dos
argumentos:
- configuración (modelo)
- pesos
'''
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Establece nuestro backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Our YOLO Layers")
# nombres de las capas, La red neuronal YOLO tiene 254 componentes
ln = net.getLayerNames()

# Hay 254 capas, Los 524 elementos consisten en capas convolucionales ( conv), unidades lineales rectificadoras ( relu),
print(len(ln), ln)  # 254 ('conv_0', 'bn_0', 'leaky_1', 'conv_1', 'bn_1', 'leaky_2', 'conv_2', 'bn_2', 'leaky_3',
# 'conv_3', 'bn_3', 'leaky_4', 'shortcut_4', 'conv_5', 'bn_5', 'leaky_6', 'conv_6', 'bn_6', 'leaky_7', 'conv_7', '
# bn_7', 'leaky_8', 'shortcut_8', 'conv_9', 'bn_9', 'leaky_10', 'conv_10', 'bn_10', 'leaky_11', 'shortcut_11', ...)


# Necesitamos pasar los nombres de las capas para las cuales se calculará la salida. net.getUnconnectedOutLayers()
# devuelve los índices de las capas de salida de la red.
ln_unconnected = net.getUnconnectedOutLayers()

print(len(ln_unconnected), ln_unconnected)  # 3 [200 227 254]

#  sólo queremos los nombres de las capas *de salida* que necesitamos de YOLO
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Hay 3 capas
print(len(ln), ln)  # 3 ['yolo_82', 'yolo_94', 'yolo_106']

# La entrada a la red es un objeto llamado blob.
# Estamos haciendo un preprocesamiento en el fotograma, llamando a este método blobFromImage. lo que
# realiza es un preprocesamiento en la imagen de entrada y ponerla en el formato adecuado para que luego podamos
# realizar inferencias en esa imagen.
# Un blob es un objeto de matriz numpy 4D (imágenes, canales, ancho, alto) y la siguiente función lo transforma a ese
# formato (blob)

# ***    cv.dnn.blobFromImage(img, scale,    size,       mean) ejemplo:
# blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
# Tiene los siguientes parámetros:**
#
# 1. la imagen a transformar
# 2. el factor de escala (1/255 para escalar los valores de los píxeles a [0..1]) no tiene que ser el mismo siempre
# 3. el tamaño, aquí una imagen cuadrada de 416x416 (ancho y alto del fotograma)
# 4. el valor medio que se va a restar de todos los fotogramas (por defecto=0)
# 5. la opción swapBR=True (ya que OpenCV usa BGR) para cambiar el orden de los canales de color en la imagen
# 6. Recorte de argumento de entrada, indica que puede recortar su imagen de entrada para que tenga el tamaño correcto
#     o puede cambiar su tamaño, al ponerlo a False, significa que simplemente vamos a cambiar el tamaño de la imagen para
#      300x300

# La llamada a la función devuelve una representación del blob del fotograma con el pre-procesamiento realizado

# **Nota** Un blob es un objeto 4D numpy array (imágenes, canales, ancho, alto). La imagen de abajo muestra el canal
# rojo del blob. Observa el brillo de la chaqueta roja en el fondo.
#
#


print("Starting Detections...")
# Obtener imágenes ubicadas en la carpeta ./images
mypath = "modelos/YOLO3/images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(file_names)
# Recorre las imágenes y pásalas por nuestro clasificador
for file in file_names:
    # cargar nuestra imagen de entrada y tomar sus dimensiones espaciales
    print(mypath + file)
    image = cv2.imread(mypath + file)
    (H, W) = image.shape[:2]

    #  Ahora construimos nuestro blob a partir de nuestras imágenes de entrada
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Estas dos instrucciones calculan la respuesta de la red:
    # *** 1: Establecemos nuestra entrada a nuestro blob de imagen
    net.setInput(blob)
    # **** 2 A continuación, ejecutamos un pase hacia adelante a través de la red
    # Los outputs por defecto objetos son vectores de longitud 85
    # 4x el cuadro delimitador (centerx, centery, ancho, alto)
    # 1x caja de confianza
    # 80x confianza de clase

    # Pasamos en ln sólo de los componentes de salida que necesitamos
    # La función forward() del módulo cv2.dnn devuelve una lista anidada que contiene información sobre todos los
    # objetos detectados, que incluye las coordenadas x e y del centro del objeto detectado, la altura y el ancho del
    # cuadro delimitador, la confianza y las puntuaciones de todos. las clases de objetos enumerados en coco.names. La
    # clase con la puntuación más alta se considera la clase predicha.
    layerOutputs = net.forward(ln)

    # inicializamos nuestras listas para nuestras cajas delimitadoras, confidencias y clases detectadas
    boxes = []
    confidences = []
    IDs = []

    # Recorremos cada una de las salidas de las capas
    """
    se crea una lista llamada scores que almacena la confianza correspondiente a cada objeto. Luego identificamos 
    el índice de clase con la mayor confianza/puntuación mediante np.argmax() . Podemos obtener el nombre de la clase 
    correspondiente al índice de la lista de clases que creamos en ln .
    """
    for output in layerOutputs:

        # Recorrer cada detección
        for detection in output:
            # [4.4197343e-02 4.8798084e-02 3.2957375e-01 1.4272095e-01 1.1992421e-06
            #  0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
            #  0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00

            # Obtener ID de clase y probabilidad de detección
            scores = detection[5:]
            # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0.]

            classID = np.argmax(scores)  # 0 en este ejemplo, índice de clase con la mayor confianza/puntuación
            confidence = scores[classID]
            """
            He seleccionado todos los cuadros delimitadores previstos con una confianza de más del 75 %. 
            Puedes jugar con este valor.
            """
            # Nos quedamos sólo con las predicciones más probables
            if confidence > 0.75:
                # Escalamos las coordenadas del cuadro delimitador respecto a la imagen
                # Nota: YOLO en realidad devuelve el centro (x, y) de la caja # delimitadora seguido de la anchura y
                # la altura.
                # caja delimitadora seguido de la anchura y la altura de la caja
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Obtener la esquina superior e izquierda de la caja delimitadora
                # Recuerda que ya tenemos la anchura y la altura
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Añade nuestra lista de coordenadas de la caja delimitadora, confidencias e IDs de clase
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)

    """
    Ahora que tenemos los vértices del cuadro delimitador predicho y class_id (índice de la clase de objeto predicha), 
    necesitamos dibujar el cuadro delimitador y agregarle una etiqueta de objeto. Lo haremos con la ayuda de la función 
    draw_labels().
    """
    # NMSBoxes
    # Aunque eliminamos los cuadros delimitadores de baja confianza, existe la posibilidad de que todavía tengamos
    # detecciones duplicadas alrededor de un objeto. Para solucionar esta situación, necesitaremos aplicar la supresión
    # no máxima (NMS) , también llamada supresión no máxima . Pasamos el valor de umbral de confianza y el valor de
    # umbral de NMS como parámetros para seleccionar un cuadro delimitador. Del rango de 0 a 1, debemos seleccionar
    # un valor intermedio como 0.4 o 0.5 para asegurarnos de que detectamos los objetos superpuestos, pero no terminamos
    # obteniendo múltiples cuadros delimitadores para el mismo objeto.

    # Ahora aplicamos la supresión de no-máximos para reducir el solapamiento de las cajas delimitadoras
    # ## **NOTA:** **Cómo realizar la supresión no máxima dadas las cajas y las puntuaciones correspondientes.**
    #
    # ```indices = cv.dnn.NMSBoxes( bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]```
    # Parámetros
    # - bboxes un conjunto de cuadros delimitadores para aplicar NMS.
    # - scores un conjunto de confidencias correspondientes.
    # - score_threshold un umbral usado para filtrar cajas por puntuación.
    # - nms_threshold un umbral utilizado en la supresión no máxima.
    # - índices los índices mantenidos de bboxes después de NMS.
    # - eta un coeficiente en la fórmula del umbral adaptativo: nms_thresholdi+1=eta⋅nms_thresholdi.
    # - top_k if >0, keep at most top_k picked índices.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Procedemos una vez encontrada una detección
    if len(idxs) > 0:
        # iteramos sobre los índices que vamos conservando
        for i in idxs.flatten():
            # Obtenemos las coordenadas de la caja delimitadora
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Dibujar nuestras cajas delimitadoras y poner nuestra etiqueta de clase en la imagen
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # mostrar la imagen de salida
    imshow("YOLO Detections", image, size=12)


##############################
#### 39 YOLOV8 INFERENCIA ########
###############################

from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
model = YOLO("modelos/YOLO8/yolo/Yolov8s.pt")
results = model.predict(source="modelos/YOLO8/1.mov", show=True)  # acepta todos los formatos - img/carpeta/video
print(results)

# *******************************************************
# ***** 40 Transferencia de Estilos Neuronales con OpenCV
# *******************************************************
# ####**En esta lección aprenderemos a usar Modelos pre-entrenados para implementar la Transferencia Neuronal de
# Estilos en OpenCV**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/NSTdemo.png)
#
# **Acerca de la Transferencia Neuronal de Estilos**
#
# Introducido por Leon Gatys et al. en 2015, en su artículo titulado "[A Neural Algorithm for Artistic Style]
# (https://arxiv.org/abs/1508.06576)", el algoritmo Neural Style Transfer se hizo viral dando lugar a una explosión de
# trabajos posteriores y aplicaciones móviles.
#
# ¡Neural Style Transfer permite aplicar el estilo artístico de una imagen a otra! Copia los patrones de color, las
# combinaciones y las pinceladas de la imagen de origen y lo aplica a la imagen de entrada. Y es una de las
# implementaciones más impresionantes de Redes Neuronales en mi opinión.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/NST.png)


# importamos los paquetes necesarios
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


# Define nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


### **Implementar la Transferencia Neuronal de Estilos usando Modelos preentrenados**
#
# Usamos modelos PyTorch t7 preentrenados que pueden ser importados usando ``cv2.dnn.readNetFromTouch()```
#
# Estos modelos que utilizamos provienen del artículo *Perceptual Losses for Real-Time Style Transfer and
# Super-Resolution* de Johnson et al.
#
# Mejoraron proponiendo un algoritmo Neural de Transferencia de Estilo que funcionaba 3 veces más rápido utilizando un
# problema similar a la super-resolución basado en la función de pérdida perceptual.


# Cargar nuestros modelos de transferencia neural t7
model_file_path = "modelos/NeuralStyleTransfer/models/"
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

# Cargar nuestra imagen de prueba
img = cv2.imread("images/city.jpg")

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
for (i, model) in enumerate(model_file_paths):
    # imprimir el modelo utilizado
    print(str(i + 1) + ". Using Model: " + str(model)[:-3])
    style = cv2.imread("modelos/NeuralStyleTransfer/art/" + str(model)[:-3] + ".jpg")
    # cargar nuestro modelo neural style transfer
    neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path + model)

    # Vamos a redimensionar a una altura fija de 640 (siéntete libre de cambiar)
    height, width = int(img.shape[0]), int(img.shape[1])
    newWidth = int((640 / height) * width)
    resizedImg = cv2.resize(img, (newWidth, 640), interpolation=cv2.INTER_AREA)

    # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
    inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False,
                                    crop=False)

    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    # Remodelar el tensor de salida, añadiendo de nuevo la sustracción de la media y reordenando los canales
    # Eso se suma debido a los datos en los que se entrenó el modelo.
    # estos valores preestablecidos que están codificados establecen un valor específico para este
    # modelo y los datos con los que fueron entrenados.

    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)

    # Mostrar nuestra imagen original, el estilo que se está aplicando y la Transposición Neural de Estilos final
    imshow("Original", img)
    imshow("Style", style)
    imshow("Neural Style Transfers", output)

# ## **Utilizando el algoritmo NST actualizado de ECCV16**
#
# En la publicación de Ulyanov et al. de 2017, *Instance Normalization: The Missing Ingredient for Fast Stylization*,
# se descubrió que cambiar la normalización de lotes por la normalización de instancias (y aplicar la normalización de
# instancias tanto en el entrenamiento como en la prueba), conduce a un rendimiento en tiempo real aún más rápido y
# podría decirse que también a resultados estéticamente más agradables.
#
# Usemos ahora los modelos utilizados por Johnson et al. en su documento ECCV.
#
#

# In[ ]:


# Cargar nuestros modelos de transferencia neural t7
model_file_path = "modelos/NeuralStyleTransfer/models/ECCV16/"
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

# Cargar nuestra imagen de prueba
img = cv2.imread("images/city.jpg")

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
for (i, model) in enumerate(model_file_paths):
    # imprimir el modelo utilizado
    print(str(i + 1) + ". Using Model: " + str(model)[:-3])
    style = cv2.imread("modelos/NeuralStyleTransfer/art/" + str(model)[:-3] + ".jpg")
    # cargar nuestro modelo neural style transfer
    neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path + model)

    # Vamos a cambiar el tamaño a una altura fija de 640 (siéntase libre de cambiar)
    height, width = int(img.shape[0]), int(img.shape[1])
    newWidth = int((640 / height) * width)
    resizedImg = cv2.resize(img, (newWidth, 640), interpolation=cv2.INTER_AREA)

    # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
    inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False,
                                    crop=False)

    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    # Remodelar el tensor de salida, añadiendo de nuevo la sustracción de la media y reordenando los canales
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)

    # Mostrar nuestra imagen original, el estilo que se está aplicando y la Transposición Neural de Estilos final
    imshow("Original", img)
    imshow("Style", style)
    imshow("Neural Style Transfers", output)

# Cargar nuestros modelos de transferencia neuronal t7
model_file_path = "modelos/NeuralStyleTransfer/models/ECCV16/starry_night.t7"

# Cargar flujo de vídeo, clip largo
cap = cv2.VideoCapture('modelos/NeuralStyleTransfer/dj.mp4')

# Obtener la altura y la anchura del fotograma (se requiere que sea un interger)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('NST_Starry_Night.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
# for (i,model) in enumerate(model_file_paths):
style = cv2.imread("modelos/NeuralStyleTransfer/art/starry_night.jpg")
i = 0
while (1):

    ret, img = cap.read()

    if ret == True:
        i += 1
        print("Completed {} Frame(s)".format(i))
        # cargar nuestro modelo de transferencia de estilo neural
        neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path)

        # Vamos a cambiar el tamaño a una altura fija de 640 (siéntase libre de cambiar)
        height, width = int(img.shape[0]), int(img.shape[1])
        newWidth = int((640 / height) * width)
        resizedImg = cv2.resize(img, (newWidth, 640), interpolation=cv2.INTER_AREA)

        # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
        inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640),
                                        (103.939, 116.779, 123.68), swapRB=False, crop=False)

        neuralStyleModel.setInput(inpBlob)
        output = neuralStyleModel.forward()

        # Remodelar el tensor de salida, añadiendo la resta de medias
        # y reordenando los canales
        output = output.reshape(3, output.shape[2], output.shape[3])
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.68
        output /= 255
        output = output.transpose(1, 2, 0)

        # Mostrar nuestra imagen original, el estilo aplicado y la Transposición Neural final
        # imshow("Original", img)
        # imshow("Style", style)
        # imshow("Neural Style Transfers", output)
        vid_output = (output * 255).astype(np.uint8)
        vid_output = cv2.resize(vid_output, (w, h), interpolation=cv2.INTER_AREA)
        out.write(vid_output)
    else:
        break

cap.release()
out.release()

# ## **¿Quieres entrenar tu propio modelo NST?**
#
# ## **Mira secciones posteriores del curso donde echaremos un vistazo a la Implementación de nuestro propio
# Algoritmo NST de Aprendizaje Profundo**
#
# Alternativamente, dale una oportunidad a este repositorio de github y pruébalo tú mismo -
# https://github.com/jcjohnson/fast-neural-style

# *******************************************************************
# ***** 41 OpCV Deteccion_rostros_aprendizaje_profundo Caffemodel
# *******************************************************************
'''Para detectar los rostros, podemos utilizar OpenCV que nos permitirá leer en un modelo previamente entrenado y
realizar inferencias usando ese modelo'''
import cv2
import sys

# Establece el índice para la cámara si no se introduce otro por parámetro.
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# crea un objeto de captura de vídeo
source = cv2.VideoCapture(s)

# crea una ventana de salida para enviar todos los resultados a la pantalla
win_name = 'Detección de cámara'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

'''OpenCV tiene varias funciones de conveniencia que nos permiten leer y pre-entrenar modelos que fueron entrenados 
usando marcos de trabajo como NetFromCaffe y pytorch que son marcos de aprendizaje profundo que permiten diseñar y 
entrenar redes neuronales. Además OpenCV tiene una funcionalidad  integrada para usar redes pre-entrenadas para realizar
inferencias ( es decir, no podemos usare OpenCV  para entrenar una red neuronal, pero puede usarlo para realizar 
inferencias en una red entrenada)

la función cv2.dnn.readNetFromCaffe es una función diseñada específicamente para leer un modelo caffemodel. necesita dos
argumentos:
- El primer argumento aquí es el archivo deploy.prototxt, que contiene la información de la arquitectura de la red,
- El segundo archivo es el archivo res10_300x300_ssd_iter_140000_fp16.caffemodel, un archivo mucho más grande que 
contiene los pesos del modelo que ha sido entrenado.

en https://github.com/opencv/opencv/tree/4.x/samples/dnn tenemos varios ejemplos de modelos pre entrenados para diversas
utilidades. Hay un archivo Léame que contiene una descripción e instrucciones sobre cómo usar el script para descargar 
varios modelos. El script hace referencia a un archivo de un modelo con una referencia en el bloque de la parte superior 
al modelo que va a utilizar y la URL para descargar el archivo de pesos, así como otros parámetros relacionados con como
se entrenó ese modelo como el factor de escala, alto, ancho y rgb.

Cuando llamamos a este método readNetFromCaffe, regresa para una instancia de la red neuronal, cuyo objeto se usará a 
continuación para realizar inferencias en nuestras imágenes de prueba de la transmisión de video'
'''
net = cv2.dnn.readNetFromCaffe("modelos/faceDetector/deploy.prototxt", "modelos/faceDetector/res10_300x300_ssd_iter_140000_fp16.caffemodel")

'''Identifica los parámetros del modelo que se asociaron con la forma en que se realizó el modelo entrenado siendo 
importante porque cualquier imagen que pasemos a través del modelo para realizar la inferencia también deben procesarse
de la misma manera que se procesaron las imágenes de entrenamiento.'''
in_width = 300  # se usaron imágenes de 300x300 para entrenar este modelo
in_height = 300
mean = [104, 117, 123]  # lista de valores medios de los canales de color de las imágenes usadas en el entrenamiento
conf_threshold = 0.7  # Umbral de competencia, es un valor que determinará la sensibilidad de las detecciones

while cv2.waitKey(1) != 27:  # mientras no pulsemos la tecla con ord 27 (esc)
    has_frame, frame = source.read()  # leemos un fotograma del vídeo
    if not has_frame:  # lo comprobamos
        break
    frame = cv2.flip(frame, 1)  # giramos horizontalmente el fotograma para mejor interpretación visual de las señales
    # se recupera el tamaño del fotograma
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Cree un blob 4D a partir de un fotograma.
    '''Estamos haciendo un preprocesamiento en el fotograma, llamando a este método blobFromImage. lo que 
    realiza es un preprocesamiento en la imagen de entrada y ponerla en el formato adecuado para que luego podamos 
    realizar inferencias en esa imagen. argumentos:
    - fotograma de la imagen
    - factor de escala (1.0) no tiene que ser el mismo siempre
    - ancho y alto del fotograma (300x300)
    - valor medio que se va a restar de todos los fotogramas
    - cambio de flag swapRB (rojo azul), en este caso no es necesario porque caffemodel y OpenCV usan la misma conveción
     para los 3 canales de color
    - Recorte de argumento de entrada, indica que puede recortar su imagen de entrada para que tenga el tamaño correcto 
    o puede cambiar su tamaño, al ponerlo a False, significa que simplemente vamos a cambiar el tamaño de la imagen para
     300x300
     La llamada a la función devuelve una representación del blob del fotograma con el pre-procesamiento realizado'''
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Corremos el modelo
    net.setInput(blob)  # pasamos el blob a esta función, establecemos la entrada, prepara para la inferencia
    detections = net.forward()  # Avanza a través de la red, realiza la inferencia sobre la representación del fotograma

    for i in range(detections.shape[2]):  # para las detecciones devueltas por la inferencia las recorre
        confidence = detections[0, 0, i, 2]
        # Determina si la competencia de una detección particular excede el umbral de detección establecido
        if confidence > conf_threshold:
            '''si lo hace profundiza y consulta en la lista de detecciones las coordenadas del fotograma de esa  
            detección en particular.'''
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)
            '''Genera un cuadro delimitador ( rectángulo) con los puntos de coordenadas obtenidos, así como un texto con
             el % de confiaza de la detección'''
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # y lo dibuja en el fotograma
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    '''Una vez que ha terminado de procesar todas las detecciones, llama a getPerfProfile, que devuelve el tiempo 
    necesitado para realizar la inferencia, lo convertimos a milisegundos y lo introduce en la imagen'''
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)  # muestra el fotograma relleno en la ventana de salida

source.release()
cv2.destroyWindow(win_name)

# ***************************************************************
# ***** 42 Detectores de disparo único (SSD) con OpenCV Caffemodel
# *****************************************************************
# ####**En esta lección aprenderemos a usar modelos pre-entrenados para implementar un SSD en OpenCV**
# Fuente - https://github.com/datitran/object_detector_app/tree/master/object_detection
'''
SSD significa detección de caja múltiple de un solo disparo. "un solo disparo" se refiere a que vamos a hacer un único
pase hacia adelante por la red neuronal para realizar inferencias y, sin embargo, detectar múltiples objetos dentro de
una imagen. Al igual que otros tipos de redes, los modelos SSD se pueden entrenar con diferentes estructuras troncales
arquitectónicas, lo que esencialmente significa que puede modelar un solo concepto pero usar diferentes columnas
dependiendo de la solicitud.

Entonces, en este caso, estamos usando una arquitectura de red móvil, que es un modelo más pequeño diseñado para
dispositivos móviles.
'''
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
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/SSDs.zip')
get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq SSDs.zip')
'''

# Descargar archivos del repositorio oficial de TensorFlow, con numerosos modelos disponibles
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# Utilizamos un modelo TensorFlow de TensorFlow modelo de detección de objetos zoo se puede utilizar para detectar
# objetos de 90 clases:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
#
# La definición del grafo de texto debe tomarse de opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
## **Encuentra otros modelos preentrenados aquí**
# https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs


# Cargar nuestras imágenes

# frame = cv2.imread('./images/elephant.jpg')
# frame = cv2.imread('./images/Volleyball.jpeg')
# frame = cv2.imread('./images/coffee.jpg')
# frame = cv2.imread('./images/hilton.jpeg')
frame = cv2.imread('./images/tommys_beers.jpeg')
imshow("original", frame)

print("Running our Single Shot Detector on our image...")
# Hacer una copia de nuestra imagen cargada
image = frame.copy()

# Establecer las anchuras y alturas que se necesitan para la entrada en nuestro modelo
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)

# Estos son necesarios para el preprocesamiento de nuestra imagen
inScaleFactor = 0.007843
meanVal = 127.5

# Apuntar a las rutas de nuestros pesos y la arquitectura del modelo en un búfer de protocolo
prototxt = "modelos/SSDs/ssd_mobilenet_v1_coco.pbtxt"  # Esta es la definición del modelo,solo la descripción del modelo
weights = "modelos/SSDs/frozen_inference_graph2.pb"

# Número de clases
num_classes = 90

# Umbral de probabilidad
thr = 0.5

# *****  leer el modelo de Tensorflow
# Toma como entrada, un archivo de modelo y el archivo de configuración y nos devolverá una instancia de la red
net = cv2.dnn.readNetFromTensorflow(weights, prototxt)

'''Hay una gran diferencia entre un detector de objetos de aprendizaje profundo y un objeto de visión artificial 
tradicional ( los revisados hasta ahora) Solíamos tener un detector para cada clase, por ejemplo, teníamos un detector 
de rostros, un detector de personas y así sucesivamente, todos modelos separados. Pero con los modelos de aprendizaje 
profundo, tenemos una enorme capacidad para aprender. Por lo tanto, un solo modelo puede detectar múltiples objetos en 
una amplia gama de ángulos de aspecto y escalas, lo que es la verdadera belleza del aprendizaje profundo'''
# ***** Comprobar las etiquetas de la clase
swapRB = True
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# Crear nuestra imagen de entrada blob necesaria para la entrada en nuestra red
# Crea un blob a partir de la imagen,
'''cuando preparamos una imagen para la inferencia, necesitamos realizar cualquier preprocesamiento en esa archivo 
que se realizó en el conjunto de entrenamiento. Esta función contiene varios argumentos relacionados con el 
preprocesamiento requerido.
- La imagen, 
- Factor de escala, establecido en uno que indica que el conjunto de entrenamiento no se le realizó ninguna 
  escala especial.
- tamaño de las imágenes de entrenamiento, (dim=300) por lo que la imagen de prueba, 
  deberán ser remodelados de acuerdo con este tamaño.
- valor medio, Si a las imágenes de entrenamiento se les hubiera aplicado un valor medio sustraído, entonces esto 
 habría sido otro vector, estas imágenes no requieren ninguna resta de medios, simplemente estamos indicando 0.
- swapRB por si queremos o no cambiar  loa canales de colores rojo y azul. EN este ejemplo queremos hacer eso, ya 
  que las imágenes de entrenamiento usan una convención diferente que lo que usa OpenCV.
- Flag de recorte, que se establece como predeterminada, es decir, las imágenes simplemente cambiarán de tamaño en 
lugar de recortarlas a la derecha.
Esta función nos devuelve una representación de blob de esa imagen que ha sido preprocesada, con lo que hay un paso 
de procesamiento previo, y luego también hay un paso de conversión de formato'''
blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
net.setInput(blob)

# Pasar nuestra imagen/blob de entrada a la red
detections = net.forward()

# Recorta el marco si es necesario, ya que no redimensionamos la entrada sino que tomamos una entrada cuadrada
cols = frame.shape[1]
rows = frame.shape[0]

if cols / float(rows) > WHRatio:
    cropSize = (int(rows * WHRatio), rows)
else:
    cropSize = (cols, int(cols / WHRatio))

y1 = int((rows - cropSize[1]) / 2)
y2 = y1 + cropSize[1]
x1 = int((cols - cropSize[0]) / 2)
x2 = x1 + cropSize[0]
frame = frame[y1:y2, x1:x2]

cols = frame.shape[1]
rows = frame.shape[0]

# Iterar sobre cada detección
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]  # y sus puntuaciones
    # Una vez que la confianza es mayor que el umbral obtenemos nuestra caja delimitadora es decir, Comprueba si la
    # detección es de buena calidad
    if confidence > thr:
        class_id = int(detections[0, 0, i, 1])  # recupera su ID de clase

        # Recuperar las coordenadas originales de las coordenadas normalizadas para el cuadro delimitador
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop = int(detections[0, 0, i, 5] * cols)
        yRightTop = int(detections[0, 0, i, 6] * rows)

        # Dibujar nuestro cuadro delimitador sobre nuestra imagen
        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0), 3)
        # Obtenemos los nombres de nuestras clases y los ponemos sobre nuestra imagen (usando un fondo blanco)
        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Mostrar nuestras detecciones
imshow("detections", frame)

# ********************************************************
# ***** 43 Detección de video mediante aprendizaje profundo
# ********************************************************
# 1.Arquitectura: Multi-Box (SSD) basado en Mobilenet
# 2.Marco: Tensorflow
'''
SSD significa detección de caja múltiple de un solo disparo. "un solo disparo" se refiere a que vamos a hacer un único
pase hacia adelante por la red neuronal para realizar inferencias y, sin embargo, detectar múltiples objetos dentro de
una imagen. Al igual que otros tipos de redes, los modelos SSD se pueden entrenar con diferentes estructuras troncales
arquitectónicas, lo que esencialmente significa que puede modelar un solo concepto pero usar diferentes columnas
dependiendo de la solicitud.

Entonces, en este caso, estamos usando una arquitectura de red móvil, que es un modelo más pequeño diseñado para
dispositivos móviles.

# Descargar archivos del repositorio oficial de TensorFlow, con numerosos modelos disponibles
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

**The cell given below downloads a mobilenet model**
## Download mobilenet model file
The code below will run on Linux / MacOS systems.
Please download the file http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

Uncompress it and put it in models folder.
'''
import cv2
import sys

# Establece el índice para la cámara si no se introduce otro por parámetro.
s = 0
# s = 'video/pr.mp4'
# s = 'rtsp://10.9.0.31/videodevice'
if len(sys.argv) > 1:
    s = sys.argv[1]

# crea un objeto de captura de vídeo
source = cv2.VideoCapture(s)

# crea una ventana de salida para enviar todos los resultados a la pantalla
win_name = 'Prueba_de_concepto'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# **** Escribir el vídeo usando OpenCV ( ojo con no haber ya recorrido el objeto de video)
'''
Para escribir el video, debe crear un objeto de videowriter con los parámetros correctos.

Sintaxis de la función
VideoWriter objeto = cv.VideoWriter (nombre de archivo, fourcc, fps, frameSize)
Parámetros
-filename: Nombre del archivo de vídeo de salida.
-fourcc: código de códec de 4 caracteres que se utiliza para comprimir los fotogramas.
 Por ejemplo, VideoWriter::fourcc('P','I','M','1') es un códec MPEG-1, VideoWriter::fourcc('M','J','P','G ') es un códec
 jpeg de movimiento, etc. La lista de códigos se puede obtener en la página Video Codecs by FOURCC. El backend FFMPEG 
 con contenedor MP4 usa de forma nativa otros valores como código fourcc: consulte ObjectType, por lo que puede recibir 
 un mensaje de advertencia de OpenCV sobre la conversión del código fourcc.
- fps: velocidad de fotogramas de la transmisión de video creada.
- frameSize: Tamaño de los fotogramas de vídeo tupla (ancho,alto).

*El tamaño del marco es importante porque deben ser las dimensiones de los marcos que tiene en la memoria que desea 
 escribir en el disco


Lo primero que vamos a hacer es usar el objeto de captura de video para llamar a este método de get(), que
nos va a recuperar las dimensiones del cuadro de video que tenemos en memoria.'''
# Se obtienen las resoluciones predeterminadas del cuadro, int() Convierte las resoluciones de float a entero
frame_width = int(source.get(3))  # en 3 guarda el ancho
frame_height = int(source.get(4))  # en 4 guarda el alto

# Define el códec y crea el objeto VideoWriter.
out_mp4 = cv2.VideoWriter('Video_camara.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

# frozen_inference_graph2.pb, que es el archivo de pesos para el modelo.
modelFile = "modelos/SSDs/frozen_inference_graph2.pb"
# Archivo de configuración para la red (Ultimo encontrado)
configFile = "modelos/SSDs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# etiquetas de clase para el conjunto de datos que se usó para entrenar este modelo
classFile = "modelos/SSDs/coco_class_labels.txt"

'''Hay una gran diferencia entre un detector de objetos de aprendizaje profundo y un objeto de visión artificial 
tradicional ( los revisados hasta ahora) Solíamos tener un detector para cada clase, por ejemplo, teníamos un detector 
de rostros, un detector de personas y así sucesivamente, todos modelos separados. Pero con los modelos de aprendizaje 
profundo, tenemos una enorme capacidad para aprender. Por lo tanto, un solo modelo puede detectar múltiples objetos en 
una amplia gama de ángulos de aspecto y escalas, lo que es la verdadera belleza del aprendizaje profundo'''

# ***** Comprobar las etiquetas de la clase
with open(classFile) as fp:
    labels = fp.read().split("\n")
print(labels)

# *****  leer el modelo de Tensorflow
# Toma como entrada, un archivo de modelo y el archivo de configuración y nos devolverá una instancia de la red
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# ***** Detectar Objetos
# Definimos una función para detectar archivos, Para cada archivo en el directorio
def detect_objects(net, im):  # toma como entrada la instancia de la red y la imagen
    dim = 300

    # Crea un blob a partir de la imagen,
    '''cuando preparamos una imagen para la inferencia, necesitamos realizar cualquier preprocesamiento en esa archivo 
    que se realizó en el conjunto de entrenamiento. Esta función contiene varios argumentos relacionados con el 
    preprocesamiento requerido.
    - La imagen, 
    - Factor de escala, establecido en uno que indica que el conjunto de entrenamiento no se le realizó ninguna 
      escala especial.
    - tamaño de las imágenes de entrenamiento, (dim=300) por lo que la imagen de prueba, 
      deberán ser remodelados de acuerdo con este tamaño.
    - valor medio, Si a las imágenes de entrenamiento se les hubiera aplicado un valor medio sustraído, entonces esto 
     habría sido otro vector, estas imágenes no requieren ninguna resta de medios, simplemente estamos indicando 0.
    - swapRB por si queremos o no cambiar  loa canales de colores rojo y azul. EN este ejemplo queremos hacer eso, ya 
      que las imágenes de entrenamiento usan una convención diferente que lo que usa OpenCV.
    - Flag de recorte, que se establece como predeterminada, es decir, las imágenes simplemente cambiarán de tamaño en 
    lugar de recortarlas a la derecha.
    Esta función nos devuelve una representación de blob de esa imagen que ha sido preprocesada, con lo que hay un paso 
    de procesamiento previo, y luego también hay un paso de conversión de formato'''
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    # pasa el blob a la red neuronal como entrada
    net.setInput(blob)

    # realiza la predicción, se realiza la inferencia en la imagen mediante el método net.forward()
    objects = net.forward()
    return objects


def display_text(im, text, x, y):  # toma le fotograma, el texto y coordenadas
    '''anotará un cuadro delimitador con la etiqueta de clase dibujando un rectángulo negro y lo  mete en el fotograma
    con algún texto que indique la etiqueta de clase dentro del negro'''
    # Obtener el tamaño del texto
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Usa el tamaño del texto para crear un rectángulo negro
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


# **** Mostrar Objetos
# configuración del texto
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


# toma el fotograma, una lista de objetos detectados y el umbral de detección
def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    # NUEVO probar para fallo
    WHRatio = 330/300
    # Recorta el marco si es necesario, ya que no redimensionamos la entrada sino que tomamos una entrada cuadrada
    cols = im.shape[1]
    rows = im.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    im = im[y1:y2, x1:x2]

    cols = frame.shape[1]
    rows = frame.shape[0]
    # fin nuevo

    # Para cada objeto detectado
    for i in range(objects.shape[2]):
        # Encuentra la clase y la confianza
        classId = int(objects[0, 0, i, 1])  # recupera su ID de clase
        score = float(objects[0, 0, i, 2])  # y sus puntuaciones

        # Recuperar las coordenadas originales de las coordenadas normalizadas para el cuadro delimitador
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Comprueba si la detección es de buena calidad
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)  # llama a la función arriba definida
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)  # introduce en la imagen un rectángulo blanco

    return im


while cv2.waitKey(1) != 27:  # mientras no pulsemos la tecla con ord 27 (esc)
    try:
        has_frame, frame = source.read()  # leemos un fotograma del vídeo
        if not has_frame:  # lo comprobamos
            break
        # frame = cv2.flip(frame, 1)  # giramos horizontalmente el fotograma para mejor interpretación visual de las señales
        # se recupera el tamaño del fotograma
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        objects = detect_objects(net, frame)



        frame = display_objects(frame, objects, 0.2)

        '''Una vez que ha terminado de procesar todas las detecciones, llama a getPerfProfile, que devuelve el tiempo 
        necesitado para realizar la inferencia, lo convertimos a milisegundos y lo introduce en la imagen'''
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow(win_name, frame)  # muestra el fotograma relleno en la ventana de salida

        # **** Escribe cada frame en el fichero
        out_mp4.write(frame)

    except Exception as e:
        print(e)


'''Cuando todo esté listo, liberamos los objetos VideoCapture y VideoWriter'''
source.release()
out_mp4.release()
cv2.destroyWindow(win_name)

# *******************************************************************
# *****44 Estimacion de la pose humana mediante el aprendizaje profundo
# *******************************************************************
'''
La estimación de la pose humana puede ser difícil:
 - Los contornos es no siempre son muy visibles
 - la ropa u otra los objetos pueden oscurecer aún más la imagen.
 - la complejidad añadida de no solo identificar los puntos clave, sino también asociarlos con las personas adecuadas

Usaremos el modelo Open Pose Cafe que se entrenó en el multipropósito conjunto de datos de imagen, y lo haremos usando
una sola imagen, señalando antes que la estimación de la pose humana a menudo se aplica a las transmisiones de video
para varias aplicaciones, como entrenadores inteligentes, por ejemplo.
'''
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython import get_ipython

#
# # Cargamos el Modelo  si no está en el directorio
# if not os.path.isdir('model'):
#   os.mkdir("model")
#
protoFile = "modelos/pose/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "modelos/pose/pose_iter_160000.caffemodel"
#
# # Descargamos el modelo si no se encuentra en el directorio
# if not os.path.isfile(protoFile):
#   # Descargamos el archivo del prototipo
#   get_ipython().system('wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt -O $protoFile')
#
# if not os.path.isfile(weightsFile):
#   # Descargamso el modelo con el archivo de lso pesos de la red
#   get_ipython().system('wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -O $weightsFile')

# Especificamos el número de puntos en el modelo y el asociado de pares de ligamiento por sus índices
'''
cada uno  de estos bloques aquí se refiere a un vínculo en la anatomía humana:
- 0 -> cabeza.
- 1 -> cuello
- 2 -> hombro derecho 
- 3 -> codo derecho
... y así sucesivamente

Es un mapeo que el modelo usa durante el entrenamiento, y vamos a necesitar este mapeo para procesar la salida de la red
'''
nPoints = 15
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
              [11, 12], [12, 13]]
# leemos el modelo pasamos el archivo del prototipo y los pesos y nos devolverá una instancia de la red que usaremos
# en la inferencia
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# leemos la imagen
im = cv2.imread("images/Tiger_Woods.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # intercambiamos los canales de color rojo y azúl
# recuperamos el tamaño de la imagen
inWidth = im.shape[1]
inHeight = im.shape[0]

'''cuando preparamos una imagen para la inferencia, necesitamos realizar cualquier preprocesamiento en esa archivo 
    que se realizó en el conjunto de entrenamiento. Esta función contiene varios argumentos relacionados con el 
    preprocesamiento requerido.
    - La imagen, 
    - Factor de escala, que es el mismo factor de escala que se aplicó a las imágenes de entrenamiento. Así que 
      necesitamos realizar esa misma transformación aquí en la imagen de entrada.
    - tamaño de las imágenes de entrenamiento, (netInputSize) por lo que la imagen de prueba, 
      deberán ser remodelados de acuerdo con este tamaño.
    - valor medio, Si a las imágenes de entrenamiento se les hubiera aplicado un valor medio sustraído, entonces esto 
     habría sido otro vector, estas imágenes no requieren ninguna resta de medios, simplemente estamos indicando 0.
    - swapRB por si queremos o no cambiar  loa canales de colores rojo y azul. EN este ejemplo queremos hacer eso, ya 
      que las imágenes de entrenamiento usan una convención diferente que lo que usa OpenCV.
    - Flag de recorte, que se establece como predeterminada, es decir, las imágenes simplemente cambiarán de tamaño en 
    lugar de recortarlas a la derecha.
    Esta función nos devuelve una representación de blob de esa imagen que ha sido preprocesada, con lo que hay un paso 
    de procesamiento previo, y luego también hay un paso de conversión de formato
'''
netInputSize = (368, 368)
# Convertimos la imagen a blob
inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
net.setInput(inpBlob)

# realiza la predicción, se realiza la inferencia en la imagen mediante el método net.forward(), devuelve es la salida
# de la red, que consta de mapas de confianza y afinidad.
output = net.forward()

# Mostrar mapas de probabilidad
'''
solo usaremos los mapas de confianza para realizar la clave detección de puntos en esta demostración. para cada punto, 
vamos a recibir un mapa de probabilidad '''
plt.figure(figsize=(20, 10))
plt.title('Probability Maps of Keypoints')
for i in range(nPoints):
    probMap = output[0, i, :, :]  # recibimos ese mapa de probabilidad
    '''y luego simplemente vamos a trazar cada uno de estos mapas de probabilidad y se podrá observar que están 
    codificados por colores, sus mapas de calor que indican la probabilidad, de la ubicación del punto clave detectado.
    El rojo es una probabilidad muy alta. en cada uno de estos mapas de probabilidad, la ubicación probable para un 
    punto clave (punto cero, cabeza,  uno cuello  y así sucesivamente.
    '''
    displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
    plt.subplot(3, 5, i + 1);
    plt.axis('off');
    plt.imshow(displayMap, cmap='jet')

'''Podemos usar estos mapas de probabilidad para superponer esos puntos clave en la imagen original. Y para hacer eso,
vamos a tener que escalarlos en la misma escala que la imagen de entrada. Estamos usando la forma de salida de la red, 
es decir, la forma de los mapas de probabilidad y también la forma de entrada de la imagen de prueba para calcular a 
escala los factores X e Y que terminaremos usando a continuación para determinar la ubicación de los puntos clave en 
la imagen de prueba real.
Antes, vamos a necesitar determinar la ubicación de los puntos clave en el mapa de probabilidad
'''

# ***** Extraemos los puntos

# X and Y Scale
scaleX = float(inWidth) / output.shape[3]
scaleY = float(inHeight) / output.shape[2]

# Lista vacía para almacenar los puntos clave detectados
points = []

# Umbral de confianza
threshold = 0.1
# Recorre todos los puntos clave, y para cada punto clave, vamos a recuperar el mapa de probabilidad de la matriz de
# salida de la red.
for i in range(nPoints):
    # Obtener mapa de probabilidad
    probMap = output[0, i, :, :]

    # Encuentra los máximos globales del probMap.
    '''llamamos a la función de OpenCV cv2.minMaxLoc pasándole el mapa de probabilidad. Y esto va devolver La ubicación
    del punto asociado con la máxima probabilidad.
    * En point se encuentran las coordenadas del punto
    '''
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    '''Una vez que tengamos esa ubicación en las coordenadas del mapa de probabilidad, la multiplicaremos por los 
    factores de escala X e Y que calculamos arriba para obtener la ubicación del punto clave en la imagen de prueba 
    original.'''
    # Escale el punto para que encaje en la imagen original
    x = scaleX * point[0]
    y = scaleY * point[1]

    if prob > threshold:  # Si la probabilidad devuelta es mayor que el umbral
        # Tomamos ese punto, agregándolo a la lista.
        points.append((int(x), int(y)))
    else:
        points.append(None)

# Y ahora estamos listos para renderizar esos puntos en la imagen de prueba.

# **** Puntos de visualización y esqueleto
# Estamos haciendo una copia de la imagen de entrada, en uno la llamamos punto y en otro esqueleto
imPoints = im.copy()
imSkeleton = im.copy()

# **** Dibujamos puntos
'''vamos a recorrer todos los puntos que fueron los que acabamos de crear en los bucles anteriores. Y esas son las 
coordenadas de los puntos clave en el cuadro de coordenadas de la imagen de prueba.
'''
for i, p in enumerate(points):
    # vamos a usar el círculo y el texto para dibujar y etiquetar esos puntos en la imagen de los puntos finales (izq)
    cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

# dibujar esqueleto
'''
vamos a renderizar la vista de esqueleto (derecha del resultado). Con este ciclo for, estamos recorriendo todos los 
pares de publicaciones, que definimos antes, y luego estamos recuperando esos pares y vamos a configurar esas dos 
partes A y parte B aquí y luego utilizarlas como índices.
'''
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    '''
    Ingresamos la lista de puntos que creamos anteriormente, que contiene la lista de ubicaciones de puntos clave en la 
    imagen de prueba Y ahora simplemente vamos a usar las funciones de círculo y línea CV abiertas para dibujar una 
    línea desde un punto hasta el siguiente codificado por colores, además de dibujar un círculo en el primer punto 
    clave en ese enlace.'''
    if points[partA] and points[partB]:
        cv2.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
        cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

plt.figure(figsize=(20, 10))
plt.subplot(121);
plt.axis('off');
plt.imshow(imPoints);  # Usamos plt.imshow para mostrar ambas imágenes
# plt.title('Displaying Points')
plt.subplot(122);
plt.axis('off');
plt.imshow(imSkeleton);
# plt.title('Displaying Skeleton')
plt.show()

# ****************************************************************************
# ***** 45 Coloriza fotos en blanco y negro usando un modelo Caffe en OpenCV
# ****************************************************************************
# En esta lección aprenderemos a usar modelos pre-entrenados para colorear automáticamente una foto en blanco y negro
# (escala de grises)
#

# ### **Colorizar imágenes en blanco y negro es una técnica increíblemente útil e increíble lograda por el aprendizaje
# profundo.**
#
# [Colorización de imágenes en blanco y negro ](http://arxiv.org/pdf/1603.08511.pdf)
#
# - Los autores abrazan la incertidumbre subyacente del problema (conversión de blanco y negro a color) planteándolo
#   como una tarea de clasificación y utilizan el reequilibrio de clases en tiempo de entrenamiento para aumentar la
#   diversidad de colores en el resultado.
# - El sistema se implementa como un paso feed-forward en una CNN en tiempo de prueba y se entrena con más de un millón
#   de imágenes en color.
# Evalúan nuestro algoritmo mediante una "prueba de Turing de coloración", en la que se pide a los participantes humanos
# que elijan entre una imagen en color generada y otra real.
# Su método consigue engañar a los humanos en el 32% de las pruebas, un porcentaje significativamente superior al de
# métodos anteriores.
#
# ![](http://richzhang.github.io/colorization/resources/images/teaser3.jpg)
#
# por Richard Zhang, Phillip Isola, Alexei A. Efros. En ECCV, 2016.
#
# Utilizaremos los siguientes archivos de modelo Caffe que descargaremos en la siguiente celda de abajo. Estos serán
# luego cargados en OpenCV:
#
# 1. colorization_deploy_v2.prototext
# 2. colorization_release_v2.caffe
# 3. pts_in_hull.npy


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


# Define nuestra función imshow
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# El script está basado en https://github.com/richzhang/colorization/blob/master/colorize.py
# Para descargar el caffemodel y el prototxt, y npy véase: https://github.com/richzhang/colorization/tree/caffe

# Inicia el programa principal
file_path = "images/color/"
blackandwhite_imgs = [f for f in listdir(file_path) if isfile(join(file_path, f))]
kernel = 'modelos/color/pts_in_hull.npy'

# Selecciona el modelo deseado
if __name__ == '__main__':

    # cargar el modelo y los pesos
    net = cv2.dnn.readNetFromCaffe("modelos/color/colorization_deploy_v2.prototxt",
                                   "modelos/color/colorization_release_v2.caffemodel")

    # cargar centros de cluster del fichero .npy ( array de 2D )
    pts_in_hull = np.load(kernel)
    '''[[ -90   50]
        [ -90   60]...'''
    # rellenar los centros de cluster como kernel de convolución 1x1
    # transpose, realiza una transposición de filas a columnas y a eso se le añaden dimensiones (a 1) con reshape
    # que devuelve una array con los mismos valores pero cambio en las dimensiones
    # pasa a ser un array 4D
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    print(pts_in_hull)
    '''[[[[ -90]]
       [[ -90]]
       [[ -90]]
    '''
    # pasa ese kernel como etiqueta de la red, para poder usarlo posteriormente
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    # para cada imagen
    for image in blackandwhite_imgs:
        # carga la imagen
        img = cv2.imread(file_path + image)

        # cambia el orden de los colores y lo pasa a flotante / 255
        img_rgb = (img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
        # Pasa de BGR a ese fomrato de laboratorio
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

        # sacar canal L
        img_l = img_lab[:, :, 0]

        # obtener el tamaño original de la imagen
        (H_orig, W_orig) = img_rgb.shape[:2]

        # redimensiona la imagen al tamaño de entrada de la red
        img_rs = cv2.resize(img_rgb, (224, 224))

        # redimensiona la imagen al tamaño de entrada de la red
        img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:, :, 0]

        # restar 50 para centrado medio
        img_l_rs -= 50

        # realiza la transformación de la imagen a blob 4D
        net.setInput(cv2.dnn.blobFromImage(img_l_rs))

        # este es nuestro resultado
        # normalmente en net.forward() que realmente realiza el paso de la red con el blob, no introducimos parámetros
        # Sin embargo, en este caso usa la etiqueta antes añadida para pasar el kernel a la red
        ab_dec = net.forward('class8_ab')[0, :, :, :].transpose((1, 2, 0))

        # Saca el ancho y el alto
        (H_out, W_out) = ab_dec.shape[:2]

        ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)

        # concatenar con imagen original L
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

        # mostrar imagen original
        imshow('Original', img)
        # Redimensionar la imagen corlizada a sus dimensiones originales
        img_bgr_out = cv2.resize(img_bgr_out, (W_orig, H_orig), interpolation=cv2.INTER_AREA)
        imshow('Colorized', img_bgr_out)


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
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
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
    image[rnd < prob] = np.random.randint(50, 230)
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
hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

# Obtener la suma acumulada
cdf = hist.cumsum()

# Obtener una distribución acumulativa normalizada
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Trazar nuestro CDF superpuesto a nuestro histograma
plt.plot(cdf_normalized, color='b')
plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
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
hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

# Obtener la suma acumulada
cdf = hist.cumsum()

# Obtener una distribución acumulativa normalizada
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Trazar nuestro CDF superpuesto a nuestro histograma
plt.plot(cdf_normalized, color='b')
plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
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
red_img[:, :, 2] = red
red_img = np.array(red_img, dtype=np.uint8)
imshow("Red", red_img)

green_img = np.zeros(img.shape)
green_img[:, :, 1] = green
green_img = np.array(green_img, dtype=np.uint8)
imshow("Green", green_img)

blue_img = np.zeros(img.shape)
blue_img[:, :, 0] = blue
blue_img = np.array(blue_img, dtype=np.uint8)
imshow("Blue", blue_img)

merged = cv2.merge([blue, green, red])
imshow("Merged", merged)


# *************************************************************
# ***** 48 Detección de Desenfoque_Encontrar Imágenes Enfocadas
# *************************************************************


# ### **Para Detectar Desenfoque, simplemente Convolvemos con el kernel Laplaciano.**
#
# Tomamos la escala de grises de una imagen y la convolucionamos con el kernel Laplaciano (kernel 3 x 3):
#
# Para cuantificar el desenfoque, entonces tomamos la varianza de la salida de respuesta.
#
# El Laplaciano es la 2ª derivada de una imagen y, por tanto, resalta las áreas de una imagen que contienen cambios
# rápidos de intensidad. De ahí su uso en la detección de bordes. Una varianza alta debería, en teoría, indicar la
# presencia tanto de bordes como de no bordes (de ahí el amplio rango de valores que resulta en una varianza alta),
# lo que es típico de una imagen normal enfocada.
#
# Una varianza baja, por lo tanto, podría significar que hay muy pocos bordes en la imagen, lo que significa que podría
# estar borrosa, ya que cuanto más borrosa esté, menos bordes habrá. #



import cv2
from matplotlib import pyplot as plt

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# #### **producir some Blurred Images**

# Cargar nuestra imagen de entrada
image = cv2.imread('./images/liberty.jpeg')
imshow("Original Image", image)

blur_1 = cv2.GaussianBlur(image, (5,5), 0)
imshow('Blurred Image 1', blur_1)

blur_2 = cv2.GaussianBlur(image, (9,9), 0)
imshow('Blurred Image 2', blur_2)

blur_3 = cv2.GaussianBlur(image, (13,13), 0)
imshow('Blurred Image 3', blur_3)


def getBlurScore(image):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cv2.Laplacian(image, cv2.CV_64F).var()


# #### **Mostrar nuestras puntuaciones, ¡recuerda que más alto significa menos desenfoque!**

print("Blur Score = {}".format(getBlurScore(image)))
print("Blur Score = {}".format(getBlurScore(blur_1)))
print("Blur Score = {}".format(getBlurScore(blur_2)))
print("Blur Score = {}".format(getBlurScore(blur_3)))


# ******************************
# ***** 49 Reconocimiento facial
# ******************************

# En esta lección, implementaremos **simples Reconocimientos Faciales usando la librería de python face-recognition**.
#
# 1. Instalar `face-recognition` #
# 2. Comprobar similitud facial
# 3. Reconocer caras en una imagen

'''
get_ipython().system('pip install face-recognition')
'''
# ## **2. Comprobar la similitud facial entre dos caras**


# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


import cv2
from matplotlib import pyplot as plt

biden = cv2.imread('images/biden.jpg')
biden2 = cv2.imread('images/biden2.jpg')
trump = cv2.imread('images/trump2.jpeg')

imshow('Trump', trump)
imshow('Biden', biden)
imshow('Biden', biden2)


# ### **Ahora probemos con las dos imágenes anteriores**

# In[5]:


import face_recognition

known_image = face_recognition.load_image_file("images/biden.jpg")
unknown_image = face_recognition.load_image_file("images/trump2.jpeg")

# ponemos la imagen en la primera posición
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# [biden_encoding] con lo que podríamos coger muchas imágenes diferentes para tomar la salida
result = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# indexamos el primer resultado porque solo queremos comparar con la primera cara

print(f'Face Match is {result[0]}')  # Face Match is False


# ### **Ahora probemos con las dos imágenes de Biden**
import face_recognition

known_image = face_recognition.load_image_file("images/biden.jpg")
unknown_image = face_recognition.load_image_file("images/biden2.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

result = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(f'Face Match is {result[0]}')  # Face Match is True


# ## **3. Reconocer caras en una imagen**
import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carga una imagen de ejemplo y aprende a reconocerla.
trump_image = face_recognition.load_image_file("images/trump2.jpeg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

# Carga una segunda imagen de ejemplo y aprende a reconocerla.
biden_image = face_recognition.load_image_file("images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Crear matrices de codificaciones de caras conocidas y sus nombres
known_face_encodings = [
    trump_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Donald Trump",
    "Joe Biden"
]

# Inicializar algunas variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# Obtener un único fotograma de vídeo
#frame = cv2.imread('images/biden2.jpg')
frame = cv2.imread('images/Trump.jpg')
# Redimensiona el fotograma de vídeo a 1/4 de tamaño para un procesamiento más rápido del reconocimiento facial
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convertir la imagen de color BGR (que utiliza OpenCV) a color RGB (que utiliza face_recognition)
rgb_small_frame = small_frame[:, :, ::-1]

# Sólo procesa cada dos fotogramas de vídeo para ahorrar tiempo
if process_this_frame:
    # Encuentra todas las caras y codificaciones de caras en el fotograma actual del vídeo
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Ver si la cara coincide con la(s) cara(s) conocida(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Si se ha encontrado una coincidencia en codificaciones_cara_conocidas, utiliza sólo la primera.
        # if True in coincidencias:
        # first_match_index = matches.index(True)
        # nombre = nombres_cara_conocidos[indice_primera_pareja]

        # O en su lugar, utilizar la cara conocida con la menor distancia a la nueva cara
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)


# Mostrar los resultados
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Vuelve a escalar las localizaciones de caras ya que el fotograma en el que detectamos se escaló a 1/4 de tamaño
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Dibuja una caja alrededor de la cara
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Dibuja una etiqueta con un nombre debajo de la cara
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Mostrar la imagen resultante
imshow('Face Recognition', frame)









