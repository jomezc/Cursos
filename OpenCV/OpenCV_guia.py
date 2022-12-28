# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imagenes del cuaderno
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
Y observe que están en el rango de 0-255 porque esta imagen está siendo representada por un entero grande de ocho bits.
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
visualización correcta, necesitamos invertir los canales de la imagen'''
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
s dimensiones de una imagen dada, como la altura de la imagen, el ancho de la imagen y la cantidad de canales en la 
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
# mostramos las imágenes
plt.figure(figsize=[18, 5])
plt.subplot(141);plt.imshow(img_NZ_rgb_flipped_horz);plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(img_NZ_rgb_flipped_vert);plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(img_NZ_rgb_flipped_both);plt.title("Both Flipped");
plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original");
plt.show()
