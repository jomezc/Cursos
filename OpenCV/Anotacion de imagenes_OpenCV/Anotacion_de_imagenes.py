# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imagenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

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
