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
