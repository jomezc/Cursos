# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imágenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

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