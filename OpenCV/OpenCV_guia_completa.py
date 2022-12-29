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
# ['C:\\Users\\jgomcano\\PycharmProjects\\guiapython\\OpenCV\\Usando la camara en openCV\\Usanod la camara en OpenCV.py']
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