# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imágenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

# *****************************************************************************************************************
# ***** Leer, imprimir, split, mergear, Conversión a diferentes espacios de color y guardar imágenes usando OpenCV
# *****************************************************************************************************************
Image(filename='checkerboard_18x18.png')  # mostrar 18x18 pixel image (en notebook) solo .

# ***** Leer e imprimir imágenes usando OpenCV
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

# leer la imagen en escala de grises e introducirlo en la variable img
img = cv2.imread('checkerboard_18x18.png', 0)  # cargamos la imagen con imread, o en escala de grises
# Lo que se carga en mi memoria es una matriz 2D de Numpy que representa la imagen.

print(img)  # pintarlo en consola
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
 [  0   0   0   0   0   0 255 255 255 255 255 255   0   0   0   0   0   0]]

Son 18 filas y 18 columnas, y cada uno de los valores representa las intensidades de píxel para cada uno de esos píxeles
Y observe que están en el rango de 0-255 porque esta imagen está siendo representada por un entero  de8-bit unsigned 
integer (0 a 255).
'''

# Imprime el tamaño de la imagen
print("Image size is ", img.shape)  # Image size is  (18, 18)

# Imprime el tipo de dato de la imagen
print("Data type of image is ", img.dtype)  # Data type of image is  uint8

# **** mostrar imagen en una ventana
plt.imshow(img)
plt.show()
''' lo muestra mediante a lib de matploit en notebook como una represantación de puntos corrdenadas x/y
es en realidad un gráfico o una representación matemática de esa imagen, pero no es 18 píxeles de ancho en mi pantalla,
es solo una trama que representa 18 píxeles. Y la razón de esto es que map plot lib usa mapas de color para representar
datos de imagen.'''
plt.imshow(img, cmap='gray')  # con el flag opcional seleccionamos formato en escala de grises


# ++++ ejemplo imagen escala de grises +++++
cb_img_fuzzy = cv2.imread("checkerboard_fuzzy_18x18.jpg", 0)

print(cb_img_fuzzy)  # pintarlo en consola
'''[[  0   0  15  20   1 134 233 253 253 253 255 229 130   1  29   2   0   0]
 [  0   1   5  18   0 137 232 255 254 247 255 228 129   0  24   2   0   0]
 [  7   5   2  28   2 139 230 254 255 249 255 226 128   0  27   3   2   2]
 [ 25  27  28  38   0 129 236 255 253 249 251 227 129   0  36  27  27  27]
 [  2   0   0   4   2 130 239 254 254 254 255 230 126   0   4   2   0   0]
 [132 129 131 124 121 163 211 226 227 225 226 203 164 125 125 129 131 131]
 [234 227 230 229 232 205 151 115 125 124 117 156 205 232 229 225 228 228]
 [254 255 255 251 255 222 102   1   0   0   0 120 225 255 254 255 255 255]
 [254 255 254 255 253 225 104   0  50  46   0 120 233 254 247 253 251 253]
 [252 250 250 253 254 223 105   2  45  50   0 127 223 255 251 255 251 253]
 [254 255 255 252 255 226 104   0   1   1   0 120 229 255 255 254 255 255]
 [233 235 231 233 234 207 142 106 108 102 108 146 207 235 237 232 231 231]
 [132 132 131 132 130 175 207 223 224 224 224 210 165 134 130 136 134 134]
 [  1   1   3   0   0 129 238 255 254 252 255 233 126   0   0   0   0   0]
 [ 20  19  30  40   5 130 236 253 252 249 255 224 129   0  39  23  21  21]
 [ 12   6   7  27   0 131 234 255 254 250 254 230 123   1  28   5  10  10]
 [  0   0   9  22   1 133 233 255 253 253 254 230 129   1  26   2   0   0]
 [  0   0   9  22   1 132 233 255 253 253 254 230 129   1  26   2   0   0]]
'''
# mostrar imagen.
plt.imshow(cb_img_fuzzy, cmap='gray')
plt.show()


# ++++ ejemplo imagen Cocacola de grises +++++
# Read and display Coca-Cola logo.
Image("coca-cola-logo.png")  # mostrar el logo de coca-cola (en notebook) solo .
coke_img = cv2.imread("coca-cola-logo.png", 1)  # leer imagen. opción formato color
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


# ***** Split y mege imágenes usando OpenCV
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
img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)  # slpit + desempaquetado

# Ahora, simplemente usaremos I am show para mostrar cada uno de esas representaciones como un mapa en escala de grises
plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(r, cmap='gray');plt.title("Red Channel");
plt.subplot(142); plt.imshow(g, cmap='gray');plt.title("Green Channel");
plt.subplot(143); plt.imshow(b, cmap='gray');plt.title("Blue Channel");
'''
Y luego este último fragmento de código toma esos canales individuales y usa la función de fusión para fusionar ellos de
nuevo en lo que debería ser la imagen original. Y llamaremos a esa imagen fusionada aquí, y también la mostraremos.
Y vale la pena mencionar un poco aquí que puede obtener algo de intuición con solo echar un vistazo en la imagen 
original. 
por ejemplo, este lago es una especie de azul turquesa, por así decirlo. Seguro que tiene algo de verde y azul, y 
probablemente muy poco de rojo. Entonces, si ahora regresa a estos canales, puede ver que el Canal Rojo para la parte 
del lago es bajo, lo que significa que no hay mucho componente rojo en ese color. Por eso es más oscuro. Está más cerca
de cero. Y fíjate en el verde. Los canales azules tienen una intensidad bastante alta para sus respectivos colores, 
lo que indica que el color de esa agua tiene un poco de rojo, pero un poco de verde y definitivamente bastante azul.'''
# Merge de cada canal en una imagen BGR
imgMerged = cv2.merge((b, g, r))
# mostramos la imagen mergeada (Invertimos el orden de ese último miembro de la matriz)
plt.subplot(144); plt.imshow(imgMerged[:, :, ::-1]); plt.title("Merged Output");
plt.show()


# ***** Conversión a diferentes espacios de color
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
# OpenCV almacena los canales de color en un orden diferente al de la mayoría de las otras aplicaciones (BGR vs RGB).
img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)  # estamos pasando la imagen y un flag que indica la conversión
plt.imshow(img_NZ_rgb)  # con el cambio Simplemente estamos mostrando la imagen original.
plt.show()

'''
vamos a convertir la representación BGR de esa imagen en un HSV representación. HSV significa saturación y valor de tono
, y ese es otro espacio de color que se usa a menudo en la imagen. Procesamiento y visión por computadora.

Y entonces vamos a almacenar ese resultado en una variable llamada image subrayado HSV. Así que ahora puedo dividir esos
canales como hicimos anteriormente y obtener los componentes HSN V, por ejemplo.

H representa el color de la saturación de la imagen, S representa la intensidad del color y V representa el valor, es 
decir, puede pensar en la saturación como un rojo puro versus un rojo opaco, y puede pensar en el valor S ( intensidad)
como cuán blanco u oscuro es el color, independientemente del color en sí. Y luego Hugh se parece más a la 
representación del color real.
'''
img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)  # conversión
h, s, v = cv2.split(img_hsv)  # Split de la imagen al desempaquetado de los componentes h, s, v


# mostramos los canales
plt.figure(figsize=[20, 5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original");
plt.show()

'''
vamos a modificar uno de los canales. Si observa esta primera línea de código, tomaremos el valor Q actual y le 
sumaremos 10 a eso. Así que solo estamos cambiando donde estamos en el espectro de color y luego fusionaré ese nuevo 
canal con los canales de arena originales, obteniendo una imagen fusionada, y luego usaremos el cvtColor() para 
convertir eso de HSV a GB.

En definitiva modifiqué uno de los canales, lo fusioné y ahora lo convertí.

por ello podremos ver la imagen modificada, porque hemos cambiado el tono, observandose diferente a la imagen original 
'''
# Cambio de la saturación
h_new = h+10
img_NZ_merged = cv2.merge((h_new, s, v))
img_NZ_rgb = cv2.cvtColor(img_NZ_merged, cv2.COLOR_HSV2RGB)

# mostramos los canales y la imagen
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Modified");
plt.show()


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
# Guardar la imagen
cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)

# leemos la imagen guardada
# read the image as Color
img_NZ_bgr = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_COLOR)
print("img_NZ_bgr shape is: ", img_NZ_bgr.shape)  # img_NZ_bgr shape is:  (600, 840, 3)

# read the image as Grayscaled
img_NZ_gry = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_GRAYSCALE)
print("img_NZ_gry shape is: ", img_NZ_gry.shape)  # img_NZ_gry shape is:  (600, 840)


# ***** mostrar la imagen con matploit o con opencv

window1 = cv2.namedWindow("w1")  # creamos una ventana
cv2.imshow('image', img_NZ_bgr, )  # llamamos al show de OpenCV, OJO como es el de Open cv se guarda y muestra en BGR
cv2.waitKey(0)  # pulsar una tecla para cerrar la imagen OpenCV si 0, si ponemos numeros seran los segundos de espera

# cv2.waitKey(8000)   # 8 segundos

# keypress = cv2.waitKey(0)  # creamos una variable que contenga la primera tecla introducida
# if keypress == ord('q'):   # si la tecla ( es en ascii) coincide con el ascii de q
#     Alive = False

cv2.destroyWindow(window1)  # destruimos la ventana creada

# *************************************
# ***** Manipulación básica de imágenes
# *************************************

# Cargamos imagen original de pruebas en escala de grises
cb_img = cv2.imread("checkerboard_18x18.png", 0)

# Establezca el mapa de colores en escala de grises para una representación adecuada.
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

# ***** Recortar imágenes
''' Recortar una imagen se logra simplemente seleccionando una región específica (píxel) de la imagen.
Es simplemente indexar una imagen existente y extraer la región que le interesa.'''
img_NZ_bgr = cv2.imread("New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)  # cargamos una imagen a color
img_NZ_rgb = img_NZ_bgr[:, :, ::-1]  # Invertimos el último color
plt.imshow(img_NZ_rgb)  # Mostramos la imagen
plt.show()  # para que se muestre

# Recortar la región media de la imagen
cropped_region = img_NZ_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()

# ***** Cambiar el tamaño de las imágenes
'''La función de cambio de tamaño cambia el tamaño de la imagen src hacia abajo o hacia arriba hasta el tamaño 
especificado. El tamaño y el tipo se derivan de src,dsize,fx y fy.
sintaxis:

 - dst = resize( src, dsize[, dst[, fx[, fy[, interpolation]]]] )

El primero es la imagen de origen, y el segundo argumento requerido es el tamaño de salida deseado de la imagen.
Y luego hay varios argumentos opcionales  Fx e Fy,  los cuáles son sus factores de escala, que vamos a demostrar a 
continuación. luego está el método de interpolación ( hay varios a seleccionar ), Por ejemplo, cuando aumenta el tamaño 
de una imagen, tiene que inventar nuevos píxeles y, por lo tanto, hay una interpolación que se requiere para hacer eso. 

La función tiene 2 argumentos requeridos:

- src: imagen de entrada
- dsize: tamaño de la imagen de salida
Los argumentos opcionales que se utilizan a menudo incluyen:
- fx: Factor de escala a lo largo del eje horizontal; cuando es igual a 0, se calcula como (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜
- fy: Factor de escala a lo largo del eje vertical; cuando es igual a 0, se calcula como (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜

La imagen de salida tiene el tamaño dsize (cuando no es cero) o el tamaño calculado a partir de src.size(), fx y fy; el 
tipo de dst es el mismo que el de src.

Documentación OpenCV
 https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
'''

# método 1: Especificación del factor de escala usando fx y fy
resized_cropped_region_2x = cv2.resize(cropped_region, None, fx=2, fy=2)
'''imagen, tamaño de salida ( al usar escala esta bien ponerlo a None factores de escala fx y tener Y.En este ejemplo, 
sólo vamos a establecerlos en dos Así que vamos a duplicar el tamaño de esta región recortada.Ahora tiene cuatrocientos 
píxeles de alto y 600 píxeles de ancho.'''
plt.imshow(resized_cropped_region_2x)
plt.show()

# método 2: Especificación del tamaño exacto de la imagen de salida
desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)
'''
vamos a establecer un ancho y alto específicos para la imagen. en este caso, cien y doscientos respectivamente , y vamos
a crear este vector bidimensional indicando ambas dimensiones y lo usamos como segundo argumento para la función de 
cambio de tamaño y mostramos la región recortada redimensionada. la imagen se ha distorsionado ahora porque no 
mantuvimos la relación de aspecto original.'''
# Cambiar el tamaño de la imagen de fondo al mismo tamaño que la imagen del logotipo
resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()

# Cambiar el tamaño manteniendo la relación de aspecto
# método 2: usando 'dsize'
desired_width = 100
''' ahora vamos a comenzar especificando un ancho de 100 y luego calcularemos la altura deseada asociada manteniendo 
la relación de aspecto. Así que aquí estamos creando esta proporción del ancho deseado al ancho original de la imagen 
y luego usando ese factor para derivar la altura deseada aquí. cuando pasamos esa dimensión revisada a la función de 
cambio de tamaño, obtenemos una imagen de cien píxeles de ancho y la cantidad adecuada de alto para mantener la 
relación adecuada, que resulta ser de unos sesenta y siete píxeles.

------------------------------------------------------------
para saber el ancho y el alto  función shape() (dimensiones) 
------------------------------------------------------------
dimensiones de una imagen dada, como la altura de la imagen, el ancho de la imagen y la cantidad de canales en la 
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
print(cropped_region.shape[1])
aspect_ratio = desired_width / cropped_region.shape[1]  # calculamos el radio de aspecto
desired_height = int(cropped_region.shape[0] * aspect_ratio)  # calculamos la nueva altura
dim = (desired_width, desired_height)

resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)  # Cambiar el tamaño de img
plt.imshow(resized_cropped_region)
plt.imshow(resized_cropped_region)

# Ahora, salvemos la imagen redimensionada (recortada)
# cambiamos el orden del canal
resized_cropped_region_2x = resized_cropped_region_2x[:, :, ::-1]

# Save resized image to disk
cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

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

img_NZ_rgb_flipped_horz = cv2.flip(img_NZ_rgb, 1)
img_NZ_rgb_flipped_vert = cv2.flip(img_NZ_rgb, 0)
img_NZ_rgb_flipped_both = cv2.flip(img_NZ_rgb, -1)

# mostramos las imágenes
plt.figure(figsize=[18, 5])
plt.subplot(141);plt.imshow(img_NZ_rgb_flipped_horz);plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(img_NZ_rgb_flipped_vert);plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(img_NZ_rgb_flipped_both);plt.title("Both Flipped");
plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original");
plt.show()

# ***************************
# ***** Anotación de imágenes
# ***************************

# Leemos la imagen
image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

# Mostramos la imagen original
plt.imshow(image[:, :, ::-1])
plt.show()

# ******  Dibujar una línea
'''
Comencemos dibujando una línea en una imagen. Usaremos la función cv2.line para esto.
Sintaxis
img = cv2.line(img, pt1, pt2, color[, grosor[, lineType[, shift]]])
img: La imagen de salida que ha sido anotada.

La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que dibujaremos una línea
- pt1: primer punto (ubicación x, y) del segmento de línea
- pt2: Segundo punto del segmento de recta
- color: Color de la línea que se dibujará
Otros argumentos opcionales que es importante que sepamos incluyen:
- grosor: Entero que especifica el grosor de la línea. El valor predeterminado es 1.
- lineType: Tipo de línea. El valor predeterminado es 8, que representa una línea conectada a 8. Por lo general, se 
 cv2.LINE_AA (línea suavizada o suavizada) para el tipo de línea.

Documentación de OpenCV¶
https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
'''
imageLine = image.copy()  # COPIAR UNA IMAGEN

'''
# La línea comienza en (200,100) y termina en (400,100)
# El color de la línea es AMARILLO (Recordemos que OpenCV usa formato BGR)
# El grosor de la línea es 5px
# El tipo de línea es cv2.LINE_AA'''

cv2.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA);

# Mostramos la imagen
plt.imshow(imageLine[:, :, ::-1])
plt.show()

# ******  Dibujar un círculo

'''
círculo en una imagen. Usaremos la función cv2.circle para esto.
sintaxis funcional
img = cv2.circle(img, centro, radio, color[, grosor[, tipo de línea[, desplazamiento]]])
img: La imagen de salida que ha sido anotada.

La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que dibujaremos una línea
- centro: Centro del círculo
- radio: Radio del círculo
- color: Color del círculo que se dibujará
A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, dará como resultado un círculo lleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Documentación de OpenCV¶
círculo: https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670

'''
# Dibujamos el circulo
imageCircle = image.copy()
cv2.circle(imageCircle, (900, 500), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

# Mostramos la imagen
plt.imshow(imageCircle[:, :, ::-1])
plt.show()

# ******  Dibujar un rectángulo
''''
Usaremos la función cv2.rectangle para dibujar un rectángulo en una imagen. 

sintaxis 
img = cv2.rectangle(img, pt1, pt2, color[, grosor[, lineType[, shift]]])
img: La imagen de salida que ha sido anotada.

La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que se va a dibujar el rectángulo.
- pt1: Vértice del rectángulo. Usualmente usamos el vértice superior izquierdo aquí.
- pt2: Vértice del rectángulo opuesto a pt1. Usualmente usamos el vértice inferior derecho aquí.
- color: color del rectángulo
A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, dará como resultado un rectángulo relleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Enlaces de documentación de OpenCV
**rectángulo:**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
'''
# Dibujamos un rectángulo
imageRectangle = image.copy()
# pt1, pt2, esquina izquierda del rectángulo en la esquina inferior derecha del rectángulo.
cv2.rectangle(imageRectangle, (500, 100), (700, 600), (255, 0, 255), thickness=5, lineType=cv2.LINE_8);
# Mostramos la imagen
plt.imshow(imageRectangle[:, :, ::-1])
plt.show()

# ****** Agregar texto
'''
Para escribir texto en una imagen usando la función cv2.putText.

sintaxis funcional
img = cv2.putText(img, text, org, fontFace, fontScale, color[, thick[, lineType[, bottomLeftOrigin]]])
img: La imagen de salida que ha sido anotada.

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

imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2

cv2.putText(imageText, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Mostramos la imagen
plt.imshow(imageText[:, :, ::-1])
plt.show()


# ******************************************
# ***** Operaciones aritméticas con imágenes
# ******************************************
'''
Las técnicas de procesamiento de imágenes aprovechan las operaciones matemáticas para lograr diferentes resultados. 
La mayoría de las veces llegamos a una versión mejorada de la imagen usando algunas operaciones básicas. Echaremos un 
vistazo a algunas de las operaciones fundamentales que se usan a menudo en las canalizaciones de visión por computadora.
En este cuaderno cubriremos operaciones aritméticas como la suma y la multiplicación.
'''
img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)  # cargar imagen a color [[[188 183 174],[189....
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cambiar el color a RGB (cv2 por defecto BGR)

# ***** Adición o Brillo
'''
La primera operación que analizamos es la simple adición de imágenes. Esto da como resultado aumentar o disminuir el 
brillo de la imagen ya que eventualmente estamos aumentando o disminuyendo los valores de intensidad de cada píxel en 
la misma cantidad. Entonces, esto resultará en un aumento/disminución global del brillo.

primera instrucción matrix = ...
- numpy.ones(): devuelve un array del tamaño y tipo indicados inicializando sus valores con unos
- crea una matriz del tamaño img_rgb.shape (con la dimensión de la imagen (600, 840, 3) es decir una imagen con el 
  tamaño de la original), tipo entero grande y con todo valor 50, es decir se crea una imagen que si la imprimimos es 
  un gris [[[50 50 50],v[50...
Y ahora simplemente vamos a usar las funciones de abrir, sumar y restar para sumar y restar esa matriz de la imagen 
original, siendo todo lo que se requiere para generar una imagen más oscura que la original y una imagen que es mas 
clara que la original
'''
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


# ******************************************
# ***** Operaciones bit a bit con imágenes
# ******************************************
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
img_rec = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("circle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[20, 5])
plt.subplot(121);
plt.imshow(img_rec, cmap='gray')
plt.subplot(122);
plt.imshow(img_cir, cmap='gray')
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
result = cv2.bitwise_and(img_rec, img_cir, mask=None)
plt.imshow(result, cmap='gray')
plt.show()

# **** Operación or
'''Ahora el valor de retorno de la operación será blanco si el píxel correspondiente de cualquier punto de la imagen es 
blanco ( 255). EN este ejemplo, obtenemos todo el lado izquierdo del rectángulo, que es blanco y luego el lado derecho
lado de la mano del círculo.'''
result = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(result, cmap='gray')
plt.show()

# **** Operación xor
''' Solo devolverá un valor de blanco si el píxel correspondiente es blanco (255) en una imagen, pero no en ambas.'''
result = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result, cmap='gray')
plt.show()

# ******* Aplicación: manipulación de logotipos  ##########

# **** Leer imagen en primer plano
img_bgr = cv2.imread("coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
print(img_rgb.shape)
logo_w = img_rgb.shape[0]  # guardamos el ancho de la imagen
logo_h = img_rgb.shape[1]  # guardamos el alto de la imagen

# **** leer la imagen de fondo
# Leer en la imagen del fondo del tablero de color
img_background_bgr = cv2.imread("checkerboard_color.png")
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
modificación de los valores de los pixeles estableciendo un valor umbral.

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


import cv2
import sys

# ******************************************
# ***** Usando la camara en OpenCV
# ******************************************

# especificamos un índice de dispositivo de cámara predeterminado de cero.
s = 0
print(sys.argv)  # contiene los argumentos de la librería sys, por ejemplo 0 es la ruta
# ['C:\\Users\\jgomcano\\PycharmProjects\\guiapython\\OpenCV\\Usando la camara en openCV\\Usando_camara_OpenCV.py']
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


import cv2
import matplotlib.pyplot as plt
# %matplotlib inline


# ******************************************
# ***** Escribir video en el disco
# ******************************************


source = './race_car.mp4'  # source = 0 for webcam
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
out_avi = cv2.VideoWriter('race_car_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out_mp4 = cv2.VideoWriter('race_car_out.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

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

import cv2
import sys
import numpy




# **********************************
# ***** Filtrado de imagen en OpenCV
# **********************************

PREVIEW  = 0   # Vista previa
BLUR     = 1   # filtro de desenfoque
FEATURES = 2   # Detector de características de corner
CANNY    = 3   # Detector de borde astuto

# Estamos definiendo un pequeño diccionario de configuración de parámetros para el detector de características de corner
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)

'''Estamos configurando el índice del dispositivo para la cámara (linea22), creando una ventana de salida para los 
resultados transmitidos (30)y luego crea un objeto de captura de video (33) para que podamos procesar la transmisión de 
video en el bucle (36)'''
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

    frame = cv2.flip(frame,1)  # mediante flip giramos el video horizontalmente

    if image_filter == PREVIEW:  # según la configuración de ejecución del script (línea 27)
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
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertimos la imagen a escala de grises
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
        https://theailearner.com/tag/cv2-goodfeaturestotrack/'''
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:  # Devuelve una lista de esquinas encontradas en la imagen
            ''' Y si detectamos una o más esquinas, simplemente anotaremos el resultado con pequeños
             círculos verdes para indicar las ubicaciones de esas características ojo con los parámetros
             al ser capturados las  posiciones x,y PASARLO a entero'''
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255 , 0), 1)

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

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ************************************************************
# ***** Caracteristicas de la imagen y alineación de la imagen
# ************************************************************
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
refFilename = "form.jpg"
print("Reading reference image : ", refFilename)
im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

# leemos la imagen que queremos alinear
imFilename = "scanned-form.jpg"
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
import glob
import matplotlib.pyplot as plt
import math

# ***********************************************
# ***** Unión de imágenes y creación de panoramas
# ***********************************************
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
imagefiles = glob.glob("boat/*")
imagefiles.sort()  # ordenamos la lista obtenida
# ['boat\\boat1.jpg', 'boat\\boat2.jpg', 'boat\\boat3.jpg', 'boat\\boat4.jpg', 'boat\\boat5.jpg', 'boat\\boat6.jpg']

images = []
# recorremos la lista de imagenes y para cada imagen la leemos en color añadiendo los objeto a una lista de imágenes
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


import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import urllib
import zipfile
# ****************************
# ***** Seguimiento de Objetos
# ****************************
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
video_input_file_name = "race_car.mp4"

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
if not os.path.isfile('goturn.prototxt') or not os.path.isfile('goturn.caffemodel'):
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
tracker_types = ['BOOSTING', 'MIL','KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN','MOSSE']

# Cambiar el índice para cambiar el tipo de rastreador
tracker_type = tracker_types[6]

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
# bbox = cv2.selectROI(frame, False)
# print(bbox)
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


# ********************************************************
# ***** Deteccion de rostros mediante aprendizaje profundo
# ********************************************************
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
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

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

# ********************************************************
# ***** Detección de Objetos mediante aprendizaje profundo
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
import os
import cv2
import numpy as np
import urllib
import matplotlib.pyplot as plt

# **** Crear un archivo de configuración a partir de un gráfico congelado
''' NO funciona con tensorflow macosx 2.11 tf_text_graph_ssd.py por el módulo tensorflow.tools.graph_transforms
1. Extrae los archivos
2. Ejecute el archivo tf_text_graph_ssd.py con la entrada como ruta al archivo frozen_graph.pb y la salida como desee.
Se ha incluido un archivo de configuración de muestra en la carpeta de modelos'''


# frozen_inference_graph.pb, que es el archivo de pesos para el modelo.
modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
# Archivo de configuración para la red
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# etiquetas de clase para el conjunto de datos que se usó para entrenar este modelo
classFile = "coco_class_labels.txt"

if not os.path.isdir('models'):
    os.mkdir("models")

if not os.path.isfile(modelFile):
    os.chdir("models")
    # Download the tensorflow Model
    urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

    # Se descomprime el fichero

    # Se borra el comprimido
    os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

    # volvemos al directorio anterior
    os.chdir("../Training")

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
    rows = im.shape[0];
    cols = im.shape[1]

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

    # Convertir imagen a RGB, ya que estamos usando Matplotlib para mostrar la imagen
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10));
    plt.imshow(mp_img);
    plt.show();

# **** Resultados
'''estamos leyendo en una imagen de prueba, y ahora vamos a usar la función que creamos para detectar los objetos que 
pasan en la instancia de red y la imagen leída, retornando la lista de objetos detectados'''
im = cv2.imread('images/street.jpg')
# llamamos a la función de visualización de objetos pasando la imagen de prueba y la matriz de objetos.
objects = detect_objects(net, im)
# este es un algoritmo de detección de objetos muy robusto que tiene alrededor de 80 clases.
display_objects(im, objects)

im = cv2.imread('images/baseball.jpg')
objects = detect_objects(net, im)
display_objects(im, objects, 0.2)

im = cv2.imread('images/soccer.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)

# *******************************************************************
# ***** Estimacion de la pose humana mediante el aprendizaje profundo
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
protoFile = "model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"
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
im = cv2.imread("Tiger_Woods.png")
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


