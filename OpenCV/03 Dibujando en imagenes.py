###############################
# 03 Dibujando en imágenes ####
###############################

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


# Empecemos por hacer un lienzo cuadrado en blanco

# Cree una imagen negra usando numpy para crear una matriz de negro
# array tridimensional, tamaño 512x512 de 3 canales de tipo entero de 0 a 255, todo a 0 significa negro
image = np.zeros((512,512,3), np.uint8)

# ¿Podemos hacer esto en blanco y negro? escala de grises
image_gray = np.zeros((512,512), np.uint8)

# El negro sería lo mismo que una imagen en escala de grises o en color (lo mismo para el blanco)
# el 1º ocupa 3 veces más memoria por las 3 dimensiones
imshow("Black Canvas - RGB Color", image)
imshow("Black Canvas - Grayscale", image_gray)


# ### **Dibujemos una línea sobre nuestro cuadrado negro**

'''
Comencemos dibujando una línea en una imagen. Usaremos la función cv2.line para esto.
Sintaxis
img = cv2.line(imagen, coordenadas iniciales, coordenadas finales, color, grosor, tipo linea)
img: La imagen de salida que ha sido anotada.

La función tiene 4 argumentos requeridos:
- imagen: Imagen sobre la que dibujaremos una línea
- coordenadas_iniciales: primer punto (ubicación x, y) del segmento de línea
- coordenadas_finales: Segundo punto del segmento de recta
- color: Color de la línea que se dibujará

Otros argumentos opcionales que es importante que sepamos incluyen:

- grosor: Entero que especifica el grosor de la línea. El valor predeterminado es 1.
- lineType: Tipo de línea. El valor predeterminado es 8, que representa una línea conectada a 8. Por lo general, se 
 cv2.LINE_AA (línea suavizada o suavizada) para el tipo de línea.

Documentación de OpenCV¶
https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
'''
# Tenga en cuenta que esta es una operación en el lugar, lo que significa que cambia la imagen de entrada
# A diferencia de muchas otras funciones de OpenCV que devuelven una nueva imagen sin afectar la entrada
# Recuerda que nuestra imagen era el lienzo negro
imageLine = image.copy()  # COPIAR UNA IMAGEN

# La línea comienza en (200,100) y termina en (400,100)
# El color de la línea es AMARILLO (Recordemos que OpenCV usa formato BGR)
# El grosor de la línea es 5px
# El tipo de línea es cv2.LINE_AA'''
cv2.line(imageLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA);
cv2.line(imageLine, (0,0), (511,511), (255,127,0), 5)

imshow("Black Canvas With Diagonal Line", imageLine)


# ******  Dibujar un rectángulo
''''
Usaremos la función cv2.rectangle para dibujar un rectángulo en una imagen. 

sintaxis 
img = cv2.rectangle(img, pt1, pt2, color[, grosor[, lineType[, shift]]])
cv2.rectangle(imagen, vértice inicial (sup izq), vértice opuesto (inf der), color, espesor)

La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que se va a dibujar el rectángulo.
- pt1: Vértice del rectángulo. Usualmente usamos el vértice superior izquierdo aquí.
- pt2: Vértice del rectángulo opuesto a pt1. Usualmente usamos el vértice inferior derecho aquí.
- color: color del rectángulo

A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, 
dará como resultado un rectángulo relleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Enlaces de documentación de OpenCV
**rectángulo:**https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
'''
# Vuelva a crear nuestro lienzo negro porque ahora tiene una línea
image = np.zeros((512,512,3), np.uint8)

# Espesor (ultimo parámetro) - si es positivo. Espesor -1 rellena el objeto
cv2.rectangle(image, (100,100), (300,250), (127,50,127), 10)
imshow("Black Canvas With Pink Rectangle", image)


# ### **Dibujemos algunos círculos**
'''círculo en una imagen. Usaremos la función cv2.circle para esto.
sintaxis funcional
img = cv2.circle(img, centro, radio, color[, grosor[, tipo de línea[, desplazamiento]]])
img: La imagen de salida que ha sido anotada.
La función tiene 4 argumentos requeridos:
- img: Imagen sobre la que dibujaremos una línea
- centro: Centro del círculo
- radio: Radio del círculo
- color: Color del círculo que se dibujará

A continuación, echemos un vistazo a los argumentos (opcionales) que vamos a utilizar bastante.
- grosor: Grosor del contorno del círculo (si es positivo). Si se proporciona un valor negativo para este argumento, 
dará como resultado un círculo lleno.
- lineType: Tipo del límite del círculo. Esto es exactamente lo mismo que el argumento lineType en cv2.line
Documentación de OpenCV¶
círculo: https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
'''
# cv2.circle(imagen, centro, radio, color, relleno)
# de nuevo la imagen negra ...
image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,150,50), -1) 
imshow("Black Canvas With Green Circle", image)


# ### **Polígonos**
# ```cv2.polylines(imagen, puntos, ¿Cerrado?, color, grosor)```
# si Cerrado = Verdadero, unimos el primer y último punto.
# De nuevo reseteamos la imagen negra ...
image = np.zeros((512, 512, 3), np.uint8)

# Definamos cuatro puntos mediante un array, una matriz con subpuntos dentro
pts = np.array([[10,50], [400,50], [90,200], [50,500]], np.int32)
pts.shape   # (4,2)
print(pts)
'''[[ 10  50]
 [400  50]
 [ 90 200]
 [ 50 500]]'''

# **Nota** cv2.polylines requiere que nuestros datos tengan la siguiente forma, por lo que hay que remodelarlos:
# estás agregando un 1, en una dimensión adicional en medio por como funciona polylines internamente, como decodifica
# los puntos
pts = pts.reshape((-1, 1, 2))
pts.shape  # (4, 1 ,2)
'''[[[ 10  50]]
 [[400  50]]
 [[ 90 200]]
 [[ 50 500]]]'''
print(pts)

cv2.polylines(image, [pts], True, (0,0,255), 3)
imshow("Black Canvas with Red Polygon", image)

# ### **Y ahora para agregar texto con cv2.putText**
'''
Para escribir texto en una imagen usando la función cv2.putText.
img = cv2.putText(img, text, org, fontFace, fontScale, color[, thick[, lineType[, bottomLeftOrigin]]])
# cv2.putText(imagen, 'Texto para mostrar', punto de inicio inferior izquierdo, Fuente, Tamaño de fuente, Color, Grosor)

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

# **Fuentes disponibles**
# - FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
# - FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX 
# - FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
# - FONT_HERSHEY_SCRIPT_SIMPLEX
# - FONT_HERSHEY_SCRIPT_COMPLEX

image = np.zeros((1000,1000,3), np.uint8)
ourString = 'Hello World!'
cv2.putText(image, ourString, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 4)
imshow("Messing with some text", image)
