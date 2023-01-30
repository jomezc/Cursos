# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imágenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

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
result = cv2.add(img_background,img_foreground)
plt.imshow(result)
cv2.imwrite("logo_final.png", result[:,:,::-1])
plt.show()