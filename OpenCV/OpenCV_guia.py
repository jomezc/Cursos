'''
Imports
'''
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imagenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

# ***********************************
# ***** Leer imágenes usando OpenCV
# ***********************************
Image(filename='checkerboard_18x18.png') # mostrar 18x18 pixel image (en notebook) solo .

# **********************************************
# ***** Leer e imprimir imágenes usando OpenCV
# **********************************************
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

# **** leer la imagen en escala de grises e introducirlo en la variable img
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
''' lo muestra mediante a lib de matploit en notebook como una represantación de puntos corrdenadas x/y
es en realidad un gráfico o una representación matemática de esa imagen, pero no es 18 píxeles de ancho en mi pantalla,
es solo una trama que representa 18 píxeles.'''
# Y la razón de esto es que map plot lib usa mapas de color para representar datos de imagen.
plt.imshow(img, cmap='gray')  # con el flag opcional seleccionamos formato en escala de grises
cv2.imshow('image', img, )  # llamamos al show de OpenCV
cv2.waitKey(0)  # Esperar a pulsar una tecla para cerrar la imagen OpenCV

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
# plt.imshow(cb_img_fuzzy, cmap='gray')
# en consola
cv2.imshow('image', cb_img_fuzzy, )  # llamamos al show de OpenCV
cv2.waitKey(0)  # Esperar a pulsar una tecla para cerrar la imagen OpenCV

# ++++ ejemplo imagen Cocacola de grises +++++
# Read and display Coca-Cola logo.
Image("coca-cola-logo.png")  # mostrar el logo de coca-cola (en notebook) solo .


coke_img = cv2.imread("coca-cola-logo.png", 1)  # leer imagen. opción formato color

print("Image size is ", coke_img.shape)  # Tamaño de la imagen, Image size is  (700, 700, 3) se ve que tiene 3 canales

print("Data type of image is ", coke_img.dtype)   # Mostrar el tipo de dato de la imagen, Data type of image is  uint8

plt.imshow(coke_img)
'''
El color que se muestra por defecto es diferente de la imagen real. Esto se debe a que 
matplotlib espera la imagen en formato RGB mientras que OpenCV almacena imágenes en formato BGR. Por lo tanto, para una 
visualización correcta, necesitamos invertir los canales de la imagen'''
# No se va a ver bien a menos que cambiemos el orden del canal.
coke_img_channels_reversed = coke_img[:, :, ::-1]  # Invierte el orden de ese último miembro de la matriz (700, 700, 3)
plt.imshow(coke_img_channels_reversed)
# en consola
cv2.imshow('image', coke_img, )  # llamamos al show de OpenCV, OJO como es el de Open cv se guarda y muestra en BGR
cv2.waitKey(0)  # Esperar a pulsar una tecla para cerrar la imagen OpenCV


# ***********************************
# ***** Split y mege imágenes usando OpenCV
# ***********************************
'''División y fusión de canales de color
cv2.split() Divide una matriz multicanal en varias matrices de un solo canal.
cv2.merge() Combina varias matrices para crear una única matriz multicanal. Todas las matrices de entrada deben tener el mismo tamaño.
Documentación de OpenCV¶
https://docs.opencv.org/4.5.1/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a

A continuación vamos a cargar una imagen y posteriormente  Voy a llamar a la función de división abierta para tomar esa 
imagen multicanal y dividirla en sus componentes. B, G y R. Y así, cada una de estas variables representa una matriz 
numpy 2D que contiene las intensidades de píxeles para esos.

canales de colores.
'''

# Split de la imagen en los componentes B, G, R

img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg",cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)
'''
Entonces, en la siguiente sección del código aquí, simplemente usaremos I am show para mostrar cada uno de esos 
representaciones como un mapa en escala de grises. '''
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(r,cmap='gray');plt.title("Red Channel");
plt.subplot(142);plt.imshow(g,cmap='gray');plt.title("Green Channel");
plt.subplot(143);plt.imshow(b,cmap='gray');plt.title("Blue Channel");

'''
Y luego este último fragmento de código toma esos canales individuales y usa la función de fusión para fusionar ellos de
nuevo en lo que debería ser la imagen original. Y llamaremos a esa imagen fusionada aquí, y también la mostraremos.
Y vale la pena mencionar un poco aquí que puede obtener algo de intuición con solo echar un vistazo
en la imagen original.'''
# Merge de cada canal en una imagen BGR
imgMerged = cv2.merge((b, g, r))
# mostramos la imagen mergeada (Invertimos el orden de ese último miembro de la matriz)
plt.subplot(144);plt.imshow(imgMerged[:, :, ::-1]);plt.title("Merged Output");
cv2.imshow('image', imgMerged, )  # llamamos al show de OpenCV, OJO como es el de Open cv se guarda y muestra en BGR
cv2.waitKey(0)  # Esperar a pulsar una tecla para cerrar la imagen OpenCV


'''
Conversión a diferentes espacios de color
cv2.cvtColor() Convierte una imagen de un espacio de color a otro. La función convierte una imagen de entrada de un espacio de color a otro. En caso de una transformación del espacio de color RGB, el orden de los canales debe especificarse explícitamente (RGB o BGR). Tenga en cuenta que el formato de color predeterminado en OpenCV a menudo se denomina RGB, pero en realidad es BGR (los bytes están invertidos). Entonces, el primer byte en una imagen de color estándar (24 bits) será un componente azul de 8 bits, el segundo byte será verde y el tercer byte será rojo. Los bytes cuarto, quinto y sexto serían entonces el segundo píxel (azul, luego verde, luego rojo), y así sucesivamente.'''