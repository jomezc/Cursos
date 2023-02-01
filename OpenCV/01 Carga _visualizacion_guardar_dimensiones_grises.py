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
