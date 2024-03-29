#########################################################
#  04 Transformaciones - Traslaciones y Rotaciones**######
#########################################################
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



# ### **translaciones**
# Esta es una transformación afín que simplemente cambia la posición de una imagen. (izquierda o derecha).
# No cambia la relación de aspecto, Básicamente lo mueve hacia la izquierda, hacia arriba o hacia abajo
# Usamos cv2.warpAffine para implementar estas transformaciones.

# cv2.warpAffine(imagen, T, (ancho, alto))
# multiplica la imagen por una matriz T, en el que Tx representa el turno alrededor del eje horizontal y Ty vertical
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/warp.png)


# Carga nuestra imagen
image = cv2.imread('images/Volleyball.jpeg')
imshow("Original", image)

# Almacenar alto y ancho de la imagen
height, width = image.shape[:2]

# Lo cambiamos por un cuarto de la altura y el ancho
quarter_height, quarter_width = height/4, width/4

# Nuestra matriz de translación
#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T es nuestra matriz de translación con un cuarto del ancho para Tx y un cuarto de la altura pars Ty
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
# ¿Cómo se ve T?
print(T)
'''
[[  1.   0. 320.]
 [  0.   1. 180.]]'''
print(height, width )  # 720 1280

# Usamos warpAffine para transformar la imagen usando la matriz, T. Lo que está haciendo es cambiar el punto de
# referencia, viendose la imagen como movida a la izquierda y abajo sobre un fondo negro
img_translation = cv2.warpAffine(image, T, (width, height))
imshow("Translated", img_translation)


# ### **Rotaciones**
# toma el punto de rotación x e y ( punto central o donde esté un pivote) y 'gira' la imagen (como un editor de fotos)
# por el ángulo de rotación elegido antihorario) y escala ( 1 significa mantener)
# cv2.getRotationMatrix2D(rotación_centro_x, rotación_centro_y, ángulo de rotación, escala)
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/rotation.png)

# Carga nuestra imagen
image = cv2.imread('images/Volleyball.jpeg')
height, width = image.shape[:2]

# Divide por dos para rotar la imagen alrededor de su centro, no rota la imagen, sino que crea la matriz que necesitamos
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)  # obtenemos la matriz 2D de rotación

# Esa matriz de rotación es lo que usamos en la translación
# Ingrese nuestra imagen, la matriz de rotación y nuestro ancho y alto final deseado
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))  # es la función que rota la imagen
imshow("Rotated 90 degrees with scale = 1", rotated_image)

# Otro ejemplo cambiando la escala

# Divide por dos para rotar la imagen alrededor de su centro
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)
# ese 0.5 reduce la escala de la imagen la hace más pequeña

print(rotation_matrix)
'''[[ 3.061617e-17  5.000000e-01  4.600000e+02]
 [-5.000000e-01  3.061617e-17  6.800000e+02]]'''

# Ingrese nuestra imagen, la matriz de rotación y nuestro ancho y alto final deseado
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 0.5", rotated_image)
# Observe todo el espacio negro que rodea la imagen.
# Ahora podemos recortar la imagen ya que podemos calcular su nuevo tamaño (¡aún no hemos aprendido a recortar!).


# ### **Rotaciones con cv2.transpose** (menos flexible)
rotated_image = cv2.transpose(image)  # menos control de lo que hacemos, solo gira la imagen y la hace espejo
imshow("Original", image)
imshow("Rotated using Transpose", rotated_image)


rotated_image = cv2.transpose(image)
# si lo hacemos dos veces obtenemos 'lo contrario', la imagen original
rotated_image = cv2.transpose(rotated_image)

imshow("Rotated using Transpose", rotated_image)

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
img_NZ_rgb_flipped_horz = cv2.flip(image, 1)  # un giro horizontal 90º, un 'volteo'
img_NZ_rgb_flipped_vert = cv2.flip(image, 0)
img_NZ_rgb_flipped_both = cv2.flip(image, -1)

# mostramos las imágenes
plt.figure(figsize=[18, 5])
plt.subplot(141);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_horz, cv2.COLOR_BGR2RGB));plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_vert, cv2.COLOR_BGR2RGB));plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(cv2.cvtColor(img_NZ_rgb_flipped_both, cv2.COLOR_BGR2RGB));plt.title("Both Flipped");
plt.subplot(144);plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.title("Original");
plt.show()

