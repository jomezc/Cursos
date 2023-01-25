# *******************************************************************
# ***** Estimacion de la pose humana mediante el aprendizaje profundo
# *******************************************************************
'''
La estimación de la pose humana puede ser difícil:
 - Los contornos es no siempre son muy visibles
 - la ropa u otra los objetos pueden oscurecer aún más la imagen.
 - la complejidad añadida de no solo identificar los puntos clave, sino también asociarlos con las personas adecuadas

Usaremos el modelo Open Pose Cafe que se entrenó en el multipropósito conjunto de datos de imagen, y lo haremos usando
una sola imagen, señalando antes que la estimación de la pose humana a menudo se aplica a las transmisiones de video
para varias aplicaciones, como entrenadores inteligentes, por ejemplo.
'''
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython import get_ipython
#
# # Cargamos el Modelo  si no está en el directorio
# if not os.path.isdir('model'):
#   os.mkdir("model")
#
protoFile = "model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"
#
# # Descargamos el modelo si no se encuentra en el directorio
# if not os.path.isfile(protoFile):
#   # Descargamos el archivo del prototipo
#   get_ipython().system('wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt -O $protoFile')
#
# if not os.path.isfile(weightsFile):
#   # Descargamso el modelo con el archivo de lso pesos de la red
#   get_ipython().system('wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -O $weightsFile')

# Especificamos el número de puntos en el modelo y el asociado de pares de ligamiento por sus índices
'''
cada uno  de estos bloques aquí se refiere a un vínculo en la anatomía humana:
- 0 -> cabeza.
- 1 -> cuello
- 2 -> hombro derecho 
- 3 -> codo derecho
... y así sucesivamente

Es un mapeo que el modelo usa durante el entrenamiento, y vamos a necesitar este mapeo para procesar la salida de la red
'''
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
# leemos el modelo pasamos el archivo del prototipo y los pesos y nos devolverá una instancia de la red que usaremos
# en la inferencia
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# leemos la imagen
im = cv2.imread("Tiger_Woods.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # intercambiamos los canales de color rojo y azúl
# recuperamos el tamaño de la imagen
inWidth = im.shape[1]
inHeight = im.shape[0]

'''cuando preparamos una imagen para la inferencia, necesitamos realizar cualquier preprocesamiento en esa archivo 
    que se realizó en el conjunto de entrenamiento. Esta función contiene varios argumentos relacionados con el 
    preprocesamiento requerido.
    - La imagen, 
    - Factor de escala, que es el mismo factor de escala que se aplicó a las imágenes de entrenamiento. Así que 
      necesitamos realizar esa misma transformación aquí en la imagen de entrada.
    - tamaño de las imágenes de entrenamiento, (netInputSize) por lo que la imagen de prueba, 
      deberán ser remodelados de acuerdo con este tamaño.
    - valor medio, Si a las imágenes de entrenamiento se les hubiera aplicado un valor medio sustraído, entonces esto 
     habría sido otro vector, estas imágenes no requieren ninguna resta de medios, simplemente estamos indicando 0.
    - swapRB por si queremos o no cambiar  loa canales de colores rojo y azul. EN este ejemplo queremos hacer eso, ya 
      que las imágenes de entrenamiento usan una convención diferente que lo que usa OpenCV.
    - Flag de recorte, que se establece como predeterminada, es decir, las imágenes simplemente cambiarán de tamaño en 
    lugar de recortarlas a la derecha.
    Esta función nos devuelve una representación de blob de esa imagen que ha sido preprocesada, con lo que hay un paso 
    de procesamiento previo, y luego también hay un paso de conversión de formato
'''
netInputSize = (368, 368)
# Convertimos la imagen a blob
inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
net.setInput(inpBlob)


# realiza la predicción, se realiza la inferencia en la imagen mediante el método net.forward(), devuelve es la salida
# de la red, que consta de mapas de confianza y afinidad.
output = net.forward()


# Mostrar mapas de probabilidad
'''
solo usaremos los mapas de confianza para realizar la clave detección de puntos en esta demostración. para cada punto, 
vamos a recibir un mapa de probabilidad '''
plt.figure(figsize=(20,10))
plt.title('Probability Maps of Keypoints')
for i in range(nPoints):
    probMap = output[0, i, :, :]  # recibimos ese mapa de probabilidad
    '''y luego simplemente vamos a trazar cada uno de estos mapas de probabilidad y se podrá observar que están 
    codificados por colores, sus mapas de calor que indican la probabilidad, de la ubicación del punto clave detectado.
    El rojo es una probabilidad muy alta. en cada uno de estos mapas de probabilidad, la ubicación probable para un 
    punto clave (punto cero, cabeza,  uno cuello  y así sucesivamente.
    '''
    displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
    plt.subplot(3, 5, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')

'''Podemos usar estos mapas de probabilidad para superponer esos puntos clave en la imagen original. Y para hacer eso,
vamos a tener que escalarlos en la misma escala que la imagen de entrada. Estamos usando la forma de salida de la red, 
es decir, la forma de los mapas de probabilidad y también la forma de entrada de la imagen de prueba para calcular a 
escala los factores X e Y que terminaremos usando a continuación para determinar la ubicación de los puntos clave en 
la imagen de prueba real.
Antes, vamos a necesitar determinar la ubicación de los puntos clave en el mapa de probabilidad
'''

# ***** Extraemos los puntos

# X and Y Scale
scaleX = float(inWidth) / output.shape[3]
scaleY = float(inHeight) / output.shape[2]

# Lista vacía para almacenar los puntos clave detectados
points = []

# Umbral de confianza
threshold = 0.1
# Recorre todos los puntos clave, y para cada punto clave, vamos a recuperar el mapa de probabilidad de la matriz de
# salida de la red.
for i in range(nPoints):
    # Obtener mapa de probabilidad
    probMap = output[0, i, :, :]
    
    # Encuentra los máximos globales del probMap.
    '''llamamos a la función de OpenCV cv2.minMaxLoc pasándole el mapa de probabilidad. Y esto va devolver La ubicación
    del punto asociado con la máxima probabilidad.
    * En point se encuentran las coordenadas del punto
    '''
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    '''Una vez que tengamos esa ubicación en las coordenadas del mapa de probabilidad, la multiplicaremos por los 
    factores de escala X e Y que calculamos arriba para obtener la ubicación del punto clave en la imagen de prueba 
    original.'''
    # Escale el punto para que encaje en la imagen original
    x = scaleX * point[0]
    y = scaleY * point[1]

    if prob > threshold:  # Si la probabilidad devuelta es mayor que el umbral
        # Tomamos ese punto, agregándolo a la lista.
        points.append((int(x), int(y)))
    else:
        points.append(None)

# Y ahora estamos listos para renderizar esos puntos en la imagen de prueba.

# **** Puntos de visualización y esqueleto
# Estamos haciendo una copia de la imagen de entrada, en uno la llamamos punto y en otro esqueleto
imPoints = im.copy()
imSkeleton = im.copy()

# **** Dibujamos puntos
'''vamos a recorrer todos los puntos que fueron los que acabamos de crear en los bucles anteriores. Y esas son las 
coordenadas de los puntos clave en el cuadro de coordenadas de la imagen de prueba.
'''
for i, p in enumerate(points):
    # vamos a usar el círculo y el texto para dibujar y etiquetar esos puntos en la imagen de los puntos finales (izq)
    cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, lineType=cv2.LINE_AA)

# dibujar esqueleto
'''
vamos a renderizar la vista de esqueleto (derecha del resultado). Con este ciclo for, estamos recorriendo todos los 
pares de publicaciones, que definimos antes, y luego estamos recuperando esos pares y vamos a configurar esas dos 
partes A y parte B aquí y luego utilizarlas como índices.
'''
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    '''
    Ingresamos la lista de puntos que creamos anteriormente, que contiene la lista de ubicaciones de puntos clave en la 
    imagen de prueba Y ahora simplemente vamos a usar las funciones de círculo y línea CV abiertas para dibujar una 
    línea desde un punto hasta el siguiente codificado por colores, además de dibujar un círculo en el primer punto 
    clave en ese enlace.'''
    if points[partA] and points[partB]:
        cv2.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
        cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

plt.figure(figsize=(20,10))
plt.subplot(121); plt.axis('off'); plt.imshow(imPoints);  # Usamos plt.imshow para mostrar ambas imágenes
#plt.title('Displaying Points')
plt.subplot(122); plt.axis('off'); plt.imshow(imSkeleton);
#plt.title('Displaying Skeleton')
plt.show()


