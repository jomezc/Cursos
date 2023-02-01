# ***************************************************************
# ***** 42 Detectores de disparo único (SSD) con OpenCV Caffemodel
# *****************************************************************
# ####**En esta lección aprenderemos a usar modelos pre-entrenados para implementar un SSD en OpenCV**
# Fuente - https://github.com/datitran/object_detector_app/tree/master/object_detection
'''
SSD significa detección de caja múltiple de un solo disparo. "un solo disparo" se refiere a que vamos a hacer un único
pase hacia adelante por la red neuronal para realizar inferencias y, sin embargo, detectar múltiples objetos dentro de
una imagen. Al igual que otros tipos de redes, los modelos SSD se pueden entrenar con diferentes estructuras troncales
arquitectónicas, lo que esencialmente significa que puede modelar un solo concepto pero usar diferentes columnas
dependiendo de la solicitud.

Entonces, en este caso, estamos usando una arquitectura de red móvil, que es un modelo más pequeño diseñado para
dispositivos móviles.
'''
# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
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
    
'''# Descargar y descomprimir nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/SSDs.zip')
get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq SSDs.zip')
'''

# Descargar archivos del repositorio oficial de TensorFlow, con numerosos modelos disponibles
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# Utilizamos un modelo TensorFlow de TensorFlow modelo de detección de objetos zoo se puede utilizar para detectar
# objetos de 90 clases:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
#
# La definición del grafo de texto debe tomarse de opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
## **Encuentra otros modelos preentrenados aquí**
# https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs


# Cargar nuestras imágenes

#frame = cv2.imread('./images/elephant.jpg')
#frame = cv2.imread('./images/Volleyball.jpeg')
#frame = cv2.imread('./images/coffee.jpg')
#frame = cv2.imread('./images/hilton.jpeg')
frame = cv2.imread('./images/tommys_beers.jpeg')
imshow("original", frame)

print("Running our Single Shot Detector on our image...")
# Hacer una copia de nuestra imagen cargada
image = frame.copy()

# Establecer las anchuras y alturas que se necesitan para la entrada en nuestro modelo
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)

# Estos son necesarios para el preprocesamiento de nuestra imagen
inScaleFactor = 0.007843
meanVal = 127.5

# Apuntar a las rutas de nuestros pesos y la arquitectura del modelo en un búfer de protocolo
prototxt = "modelos/SSDs/ssd_mobilenet_v1_coco.pbtxt"  # Esta es la definición del modelo,solo la descripción del modelo
weights = "modelos/SSDs/frozen_inference_graph2.pb"

# Número de clases
num_classes = 90

# Umbral de probabilidad
thr = 0.5


# *****  leer el modelo de Tensorflow
# Toma como entrada, un archivo de modelo y el archivo de configuración y nos devolverá una instancia de la red
net = cv2.dnn.readNetFromTensorflow(weights, prototxt)


'''Hay una gran diferencia entre un detector de objetos de aprendizaje profundo y un objeto de visión artificial 
tradicional ( los revisados hasta ahora) Solíamos tener un detector para cada clase, por ejemplo, teníamos un detector 
de rostros, un detector de personas y así sucesivamente, todos modelos separados. Pero con los modelos de aprendizaje 
profundo, tenemos una enorme capacidad para aprender. Por lo tanto, un solo modelo puede detectar múltiples objetos en 
una amplia gama de ángulos de aspecto y escalas, lo que es la verdadera belleza del aprendizaje profundo'''
# ***** Comprobar las etiquetas de la clase
swapRB = True
classNames = { 0: 'background',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }

# Crear nuestra imagen de entrada blob necesaria para la entrada en nuestra red
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
blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
net.setInput(blob)

# Pasar nuestra imagen/blob de entrada a la red
detections = net.forward()

# Recorta el marco si es necesario, ya que no redimensionamos la entrada sino que tomamos una entrada cuadrada
cols = frame.shape[1]
rows = frame.shape[0]

if cols / float(rows) > WHRatio:
    cropSize = (int(rows * WHRatio), rows)
else:
    cropSize = (cols, int(cols / WHRatio))


y1 = int((rows - cropSize[1]) / 2)
y2 = y1 + cropSize[1]
x1 = int((cols - cropSize[0]) / 2)
x2 = x1 + cropSize[0]
frame = frame[y1:y2, x1:x2]

cols = frame.shape[1]
rows = frame.shape[0]

# Iterar sobre cada detección
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]  # y sus puntuaciones
    # Una vez que la confianza es mayor que el umbral obtenemos nuestra caja delimitadora es decir, Comprueba si la
    # detección es de buena calidad
    if confidence > thr:
        class_id = int(detections[0, 0, i, 1])  # recupera su ID de clase

        # Recuperar las coordenadas originales de las coordenadas normalizadas para el cuadro delimitador
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)

        # Dibujar nuestro cuadro delimitador sobre nuestra imagen
        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                      (0, 255, 0), 3)
        # Obtenemos los nombres de nuestras clases y los ponemos sobre nuestra imagen (usando un fondo blanco)
        if class_id in classNames:
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Mostrar nuestras detecciones
imshow("detections", frame)






