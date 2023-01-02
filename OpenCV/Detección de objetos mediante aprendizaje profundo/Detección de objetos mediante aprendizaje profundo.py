# ********************************************************
# ***** Detección de Objetos mediante aprendizaje profundo
# ********************************************************
# 1.Arquitectura: Multi-Box (SSD) basado en Mobilenet
# 2.Marco: Tensorflow
'''
SSD significa detección de caja múltiple de un solo disparo. "un solo disparo" se refiere a que vamos a hacer un único
pase hacia adelante por la red neuronal para realizar inferencias y, sin embargo, detectar múltiples objetos dentro de
una imagen. Al igual que otros tipos de redes, los modelos SSD se pueden entrenar con diferentes estructuras troncales
arquitectónicas, lo que esencialmente significa que puede modelar un solo concepto pero usar diferentes columnas
dependiendo de la solicitud.

Entonces, en este caso, estamos usando una arquitectura de red móvil, que es un modelo más pequeño diseñado para
dispositivos móviles.

# Descargar archivos del repositorio oficial de TensorFlow, con numerosos modelos disponibles
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

**The cell given below downloads a mobilenet model**
## Download mobilenet model file
The code below will run on Linux / MacOS systems.
Please download the file http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

Uncompress it and put it in models folder.
'''
import os
import cv2
import numpy as np
import urllib
import matplotlib.pyplot as plt

# **** Crear un archivo de configuración a partir de un gráfico congelado
''' NO funciona con tensorflow macosx 2.11 tf_text_graph_ssd.py por el módulo tensorflow.tools.graph_transforms
1. Extrae los archivos
2. Ejecute el archivo tf_text_graph_ssd.py con la entrada como ruta al archivo frozen_graph.pb y la salida como desee.
Se ha incluido un archivo de configuración de muestra en la carpeta de modelos'''


# frozen_inference_graph.pb, que es el archivo de pesos para el modelo.
modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
# Archivo de configuración para la red
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# etiquetas de clase para el conjunto de datos que se usó para entrenar este modelo
classFile = "coco_class_labels.txt"

if not os.path.isdir('models'):
    os.mkdir("models")

if not os.path.isfile(modelFile):
    os.chdir("models")
    # Download the tensorflow Model
    urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

    # Se descomprime el fichero

    # Se borra el comprimido
    os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

    # volvemos al directorio anterior
    os.chdir("..")

'''Hay una gran diferencia entre un detector de objetos de aprendizaje profundo y un objeto de visión artificial 
tradicional ( los revisados hasta ahora) Solíamos tener un detector para cada clase, por ejemplo, teníamos un detector 
de rostros, un detector de personas y así sucesivamente, todos modelos separados. Pero con los modelos de aprendizaje 
profundo, tenemos una enorme capacidad para aprender. Por lo tanto, un solo modelo puede detectar múltiples objetos en 
una amplia gama de ángulos de aspecto y escalas, lo que es la verdadera belleza del aprendizaje profundo'''
# ***** Comprobar las etiquetas de la clase
with open(classFile) as fp:
    labels = fp.read().split("\n")
print(labels)


# *****  leer el modelo de Tensorflow
# Toma como entrada, un archivo de modelo y el archivo de configuración y nos devolverá una instancia de la red
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


# ***** Detectar Objetos
# Definimos una función para detectar archivos, Para cada archivo en el directorio
def detect_objects(net, im):  # toma como entrada la instancia de la red y la imagen
    dim = 300

    # Crea un blob a partir de la imagen,
    '''cuando preparamos una imagen para la inferencia, necesitamos realizar cualquier preprocesamiento en esa archivo 
    que se realizó en el conjunto de entrenamiento. Esta función contiene varios argumentos relacionados con el 
    preprocesamiento requerido.
    - La imagen, 
    - Factor de escala, establecido en uno que indica que el conjunto de entrenamiento no se le realizó ninguna 
      escala especial.
    - tamaño de las imágenes de entrenamiento, (dim=300) por lo que la imagen de prueba, 
      deberán ser remodelados de acuerdo con este tamaño.
    - valor medio, Si a las imágenes de entrenamiento se les hubiera aplicado un valor medio sustraído, entonces esto 
     habría sido otro vector, estas imágenes no requieren ninguna resta de medios, simplemente estamos indicando 0.
    - swapRB por si queremos o no cambiar  loa canales de colores rojo y azul. EN este ejemplo queremos hacer eso, ya 
      que las imágenes de entrenamiento usan una convención diferente que lo que usa OpenCV.
    - Flag de recorte, que se establece como predeterminada, es decir, las imágenes simplemente cambiarán de tamaño en 
    lugar de recortarlas a la derecha.
    Esta función nos devuelve una representación de blob de esa imagen que ha sido preprocesada, con lo que hay un paso 
    de procesamiento previo, y luego también hay un paso de conversión de formato'''
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    # pasa el blob a la red neuronal como entrada
    net.setInput(blob)

    # realiza la predicción, se realiza la inferencia en la imagen mediante el método net.forward()
    objects = net.forward()
    return objects


def display_text(im, text, x, y):  # toma le fotograma, el texto y coordenadas
    '''anotará un cuadro delimitador con la etiqueta de clase dibujando un rectángulo negro y lo  mete en el fotograma
    con algún texto que indique la etiqueta de clase dentro del negro'''
    # Obtener el tamaño del texto
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Usa el tamaño del texto para crear un rectángulo negro
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


# **** Mostrar Objetos
# configuración del texto
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


# toma el fotograma, una lista de objetos detectados y el umbral de detección
def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0];
    cols = im.shape[1]

    # Para cada objeto detectado
    for i in range(objects.shape[2]):
        # Encuentra la clase y la confianza
        classId = int(objects[0, 0, i, 1])  # recupera su ID de clase
        score = float(objects[0, 0, i, 2])  # y sus puntuaciones

        # Recuperar las coordenadas originales de las coordenadas normalizadas para el cuadro delimitador
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Comprueba si la detección es de buena calidad
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)  # llama a la función arriba definida
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)  # introduce en la imagen un rectángulo blanco

    # Convertir imagen a RGB, ya que estamos usando Matplotlib para mostrar la imagen
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10));
    plt.imshow(mp_img);
    plt.show();

# **** Resultados
'''estamos leyendo en una imagen de prueba, y ahora vamos a usar la función que creamos para detectar los objetos que 
pasan en la instancia de red y la imagen leída, retornando la lista de objetos detectados'''
im = cv2.imread('images/street.jpg')
# llamamos a la función de visualización de objetos pasando la imagen de prueba y la matriz de objetos.
objects = detect_objects(net, im)
# este es un algoritmo de detección de objetos muy robusto que tiene alrededor de 80 clases.
display_objects(im, objects)

im = cv2.imread('images/baseball.jpg')
objects = detect_objects(net, im)
display_objects(im, objects, 0.2)

im = cv2.imread('images/soccer.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)