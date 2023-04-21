####################################################################
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


# Cargamos imagen original de SOCKET en escala de grises
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

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Resizing.png)
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

