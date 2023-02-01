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
