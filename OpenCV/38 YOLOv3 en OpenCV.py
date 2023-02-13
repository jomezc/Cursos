#########################################################################
# 38 **YOLOv3 usando cv2.dnn.readNetFrom()**#####
#########################################################################
# tutorial oficial en https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
# https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
# https://neptune.ai/blog/object-detection-with-yolo-hands-on-tutorial
# YOLO es uno de los algoritmos de detección de objetos en tiempo real más rápidos (45 cuadros por segundo) en
# comparación con la familia R-CNN (R-CNN, Fast R-CNN, Faster R-CNN, etc.)
# La familia de algoritmos R-CNN utiliza regiones para localizar los objetos en las imágenes, lo que significa que el
# modelo se aplica a varias regiones y las regiones de la imagen con una puntuación alta se consideran objetos
# detectados. Pero YOLO sigue un enfoque completamente diferente. En lugar de seleccionar algunas regiones, aplica una
# red neuronal a toda la imagen para predecir los cuadros delimitadores y sus probabilidades.

# ####**En esta lección aprenderemos a cargar un Modelo YOLOV3 pre-entrenado y usar OpenCV para ejecutar inferencias
# sobre algunas imágenes**
# YOLOV -> detector de objetos
# importar los paquetes necesarios
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



# ## **Detección de Objetos YOLO**
# ![](https://opencv-tutorial.readthedocs.io/en/latest/_images/yolo1_net.png)
# Pasos necesarios
# 1. Usar pesos YOLOV3 preentrenados (237MB)- https://pjreddie.com/media/files/yolov3.weights
# 2. Crear nuestro objeto blob que es nuestro modelo cargado
# 3. Establecer el backend que ejecuta el modelo

# Cargar las etiquetas de clase COCO con las que se ha entrenado nuestro modelo YOLO
# coco es un tipo de conjunto de datos de objetos comunes
# ImageNet es un conjunto de datos clasificado que ha demostrado ser invaluable para la investigación de computer vision
# contiene los nombres de los diferentes objetos que nuestro modelo ha sido entrenado para identificar.

labelsPath = "modelos/YOLO3/yolo/coco.names"
'''
person
bicycle
car
motorbike'''
LABELS = open(labelsPath).read().strip().split("\n")
 
# Ahora necesitamos inicializar una lista de colores para representar cada posible etiqueta de clase
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("Loading YOLO weights...") 

# almacena todos los pesos para el modelo
weights_path = "modelos/YOLO3/yolo/yolov3.weights"

# Es la estructura del modelo, define toda la estructura del modelo YOLOV que está codificada en su campo de cisión
cfg_path = "modelos/YOLO3/yolo/yolov3.cfg"

# Crear nuestro objeto blob
'''OpenCV tiene varias funciones de conveniencia que nos permiten leer y pre-entrenar modelos que fueron entrenados 
usando marcos de trabajo como NetFromDarknet y pytorch que son marcos de aprendizaje profundo que permiten diseñar y 
entrenar redes neuronales. Además OpenCV tiene una funcionalidad  integrada para usar redes pre-entrenadas para realizar
inferencias ( es decir, no podemos usare OpenCV  para entrenar una red neuronal, pero puede usarlo para realizar 
inferencias en una red entrenada)

la función cv2.dnn.readNetFromDarknet es una función diseñada específicamente para cargar un modelo . necesita dos
argumentos:
- configuración (modelo)
- pesos
'''
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Establece nuestro backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Our YOLO Layers")
# nombres de las capas, La red neuronal YOLO tiene 254 componentes
ln = net.getLayerNames()

# Hay 254 capas, Los 524 elementos consisten en capas convolucionales ( conv), unidades lineales rectificadoras ( relu),
print(len(ln), ln)  # 254 ('conv_0', 'bn_0', 'leaky_1', 'conv_1', 'bn_1', 'leaky_2', 'conv_2', 'bn_2', 'leaky_3',
# 'conv_3', 'bn_3', 'leaky_4', 'shortcut_4', 'conv_5', 'bn_5', 'leaky_6', 'conv_6', 'bn_6', 'leaky_7', 'conv_7', '
# bn_7', 'leaky_8', 'shortcut_8', 'conv_9', 'bn_9', 'leaky_10', 'conv_10', 'bn_10', 'leaky_11', 'shortcut_11', ...)


# Necesitamos pasar los nombres de las capas para las cuales se calculará la salida. net.getUnconnectedOutLayers()
# devuelve los índices de las capas de salida de la red.
ln_unconnected = net.getUnconnectedOutLayers()

print(len(ln_unconnected), ln_unconnected)  # 3 [200 227 254]

#  sólo queremos los nombres de las capas *de salida* que necesitamos de YOLO
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Hay 3 capas
print(len(ln), ln)  # 3 ['yolo_82', 'yolo_94', 'yolo_106']


# La entrada a la red es un objeto llamado blob.
# Estamos haciendo un preprocesamiento en el fotograma, llamando a este método blobFromImage. lo que
# realiza es un preprocesamiento en la imagen de entrada y ponerla en el formato adecuado para que luego podamos
# realizar inferencias en esa imagen.
# Un blob es un objeto de matriz numpy 4D (imágenes, canales, ancho, alto) y la siguiente función lo transforma a ese
# formato (blob)

# ***    cv.dnn.blobFromImage(img, scale,    size,       mean) ejemplo:
# blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
# Tiene los siguientes parámetros:**
#
# 1. la imagen a transformar
# 2. el factor de escala (1/255 para escalar los valores de los píxeles a [0..1]) no tiene que ser el mismo siempre
# 3. el tamaño, aquí una imagen cuadrada de 416x416 (ancho y alto del fotograma)
# 4. el valor medio que se va a restar de todos los fotogramas (por defecto=0)
# 5. la opción swapBR=True (ya que OpenCV usa BGR) para cambiar el orden de los canales de color en la imagen
# 6. Recorte de argumento de entrada, indica que puede recortar su imagen de entrada para que tenga el tamaño correcto
#   o puede cambiar su tamaño, al ponerlo a False, significa que simplemente vamos a cambiar el tamaño de la imagen para
#    300x300

# La llamada a la función devuelve una representación del blob del fotograma con el pre-procesamiento realizado

# **Nota** Un blob es un objeto 4D numpy array (imágenes, canales, ancho, alto). La imagen de abajo muestra el canal
# rojo del blob. Observa el brillo de la chaqueta roja en el fondo.
#
#


print("Starting Detections...")
# Obtener imágenes ubicadas en la carpeta ./images
mypath = "modelos/YOLO3/images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(file_names)
# Recorre las imágenes y pásalas por nuestro clasificador
for file in file_names:
    # cargar nuestra imagen de entrada y tomar sus dimensiones espaciales
    print(mypath+file)
    image = cv2.imread(mypath+file)
    (H, W) = image.shape[:2]


    #  Ahora construimos nuestro blob a partir de nuestras imágenes de entrada
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Estas dos instrucciones calculan la respuesta de la red:
    # *** 1: Establecemos nuestra entrada a nuestro blob de imagen
    net.setInput(blob)
    # **** 2 A continuación, ejecutamos un pase hacia adelante a través de la red
    # Los outputs por defecto objetos son vectores de longitud 85
    # 4x el cuadro delimitador (centerx, centery, ancho, alto)
    # 1x caja de confianza
    # 80x confianza de clase

    # Pasamos en ln sólo de los componentes de salida que necesitamos
    # La función forward() del módulo cv2.dnn devuelve una lista anidada que contiene información sobre todos los
    # objetos detectados, que incluye las coordenadas x e y del centro del objeto detectado, la altura y el ancho del
    # cuadro delimitador, la confianza y las puntuaciones de todos. las clases de objetos enumerados en coco.names. La
    # clase con la puntuación más alta se considera la clase predicha.
    layerOutputs = net.forward(ln)

    # inicializamos nuestras listas para nuestras cajas delimitadoras, confidencias y clases detectadas
    boxes = []
    confidences = []
    IDs = []

    # Recorremos cada una de las salidas de las capas
    """
    se crea una lista llamada scores que almacena la confianza correspondiente a cada objeto. Luego identificamos 
    el índice de clase con la mayor confianza/puntuación mediante np.argmax() . Podemos obtener el nombre de la clase 
    correspondiente al índice de la lista de clases que creamos en ln .
    """
    for output in layerOutputs:

        # Recorrer cada detección
        for detection in output:
            # [4.4197343e-02 4.8798084e-02 3.2957375e-01 1.4272095e-01 1.1992421e-06
            #  0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
            #  0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00

            # Obtener ID de clase y probabilidad de detección
            scores = detection[5:]
            # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  0. 0. 0. 0. 0. 0. 0. 0.]

            classID = np.argmax(scores)  # 0 en este ejemplo, índice de clase con la mayor confianza/puntuación
            confidence = scores[classID]
            """
            He seleccionado todos los cuadros delimitadores previstos con una confianza de más del 75 %. 
            Puedes jugar con este valor.
            """
            # Nos quedamos sólo con las predicciones más probables
            if confidence > 0.75:

                # Escalamos las coordenadas del cuadro delimitador respecto a la imagen
                # Nota: YOLO en realidad devuelve el centro (x, y) de la caja # delimitadora seguido de la anchura y
                # la altura.
                # caja delimitadora seguido de la anchura y la altura de la caja
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Obtener la esquina superior e izquierda de la caja delimitadora
                # Recuerda que ya tenemos la anchura y la altura
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Añade nuestra lista de coordenadas de la caja delimitadora, confidencias e IDs de clase
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)

    """
    Ahora que tenemos los vértices del cuadro delimitador predicho y class_id (índice de la clase de objeto predicha), 
    necesitamos dibujar el cuadro delimitador y agregarle una etiqueta de objeto. Lo haremos con la ayuda de la función 
    draw_labels().
    """
    # NMSBoxes
    # Aunque eliminamos los cuadros delimitadores de baja confianza, existe la posibilidad de que todavía tengamos
    # detecciones duplicadas alrededor de un objeto. Para solucionar esta situación, necesitaremos aplicar la supresión
    # no máxima (NMS). Pasamos el valor de umbral de confianza y el valor de
    # umbral de NMS como parámetros para seleccionar un cuadro delimitador. Del rango de 0 a 1, debemos seleccionar
    # un valor intermedio como 0.4 o 0.5 para asegurarnos de que detectamos los objetos superpuestos, pero no terminamos
    # obteniendo múltiples cuadros delimitadores para el mismo objeto.

    # Ahora aplicamos la supresión de no-máximos para reducir el solapamiento de las cajas delimitadoras
    # ## **NOTA:** **Cómo realizar la supresión no máxima dadas las cajas y las puntuaciones correspondientes.**
    #
    # ```indices = cv.dnn.NMSBoxes( bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]```
    # Parámetros
    # - bboxes un conjunto de cuadros delimitadores para aplicar NMS.
    # - scores un conjunto de confidencias correspondientes.
    # - score_threshold un umbral usado para filtrar cajas por puntuación.
    # - nms_threshold un umbral utilizado en la supresión no máxima.
    # - índices los índices mantenidos de bboxes después de NMS.
    # - eta un coeficiente en la fórmula del umbral adaptativo: nms_thresholdi+1=eta⋅nms_thresholdi.
    # - top_k if >0, keep at most top_k picked índices.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)


    # Procedemos una vez encontrada una detección
    if len(idxs) > 0:
        # iteramos sobre los índices que vamos conservando
        for i in idxs.flatten():
            # Obtenemos las coordenadas de la caja delimitadora
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Dibujar nuestras cajas delimitadoras y poner nuestra etiqueta de clase en la imagen
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # mostrar la imagen de salida
    imshow("YOLO Detections", image, size = 12)



