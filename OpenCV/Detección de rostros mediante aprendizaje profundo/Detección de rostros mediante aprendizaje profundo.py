# ********************************************************
# ***** Detección de rostros mediante aprendizaje profundo
# ********************************************************
'''Para detectar los rostros, podemos utilizar OpenCV que nos permitirá leer en un modelo previamente entrenado y
realizar inferencias usando ese modelo'''
import cv2
import sys

# Establece el índice para la cámara si no se introduce otro por parámetro.
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# crea un objeto de captura de vídeo
source = cv2.VideoCapture(s)

# crea una ventana de salida para enviar todos los resultados a la pantalla
win_name = 'Detección de cámara'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

'''OpenCV tiene varias funciones de conveniencia que nos permiten leer y pre-entrenar modelos que fueron entrenados 
usando marcos de trabajo como NetFromCaffe y pytorch que son marcos de aprendizaje profundo que permiten diseñar y 
entrenar redes neuronales. Además OpenCV tiene una funcionalidad  integrada para usar redes pre-entrenadas para realizar
inferencias ( es decir, no podemos usare OpenCV  para entrenar una red neuronal, pero puede usarlo para realizar 
inferencias en una red entrenada)

la función cv2.dnn.readNetFromCaffe es una función diseñada específicamente para leer un modelo caffemodel. necesita dos
argumentos:
- El primer argumento aquí es el archivo deploy.prototxt, que contiene la información de la arquitectura de la red,
- El segundo archivo es el archivo res10_300x300_ssd_iter_140000_fp16.caffemodel, un archivo mucho más grande que 
contiene los pesos del modelo que ha sido entrenado.

en https://github.com/opencv/opencv/tree/4.x/samples/dnn tenemos varios ejemplos de modelos pre entrenados para diversas
utilidades. Hay un archivo Léame que contiene una descripción e instrucciones sobre cómo usar el script para descargar 
varios modelos. El script hace referencia a un archivo de un modelo con una referencia en el bloque de la parte superior 
al modelo que va a utilizar y la URL para descargar el archivo de pesos, así como otros parámetros relacionados con como
se entrenó ese modelo como el factor de escala, alto, ancho y rgb.

Cuando llamamos a este método readNetFromCaffe, regresa para una instancia de la red neuronal, cuyo objeto se usará a 
continuación para realizar inferencias en nuestras imágenes de prueba de la transmisión de video'
'''
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

'''Identifica los parámetros del modelo que se asociaron con la forma en que se realizó el modelo entrenado siendo 
importante porque cualquier imagen que pasemos a través del modelo para realizar la inferencia también deben procesarse
de la misma manera que se procesaron las imágenes de entrenamiento.'''
in_width = 300  # se usaron imágenes de 300x300 para entrenar este modelo
in_height = 300
mean = [104, 117, 123]  # lista de valores medios de los canales de color de las imágenes usadas en el entrenamiento
conf_threshold = 0.7  # Umbral de competencia, es un valor que determinará la sensibilidad de las detecciones

while cv2.waitKey(1) != 27:  # mientras no pulsemos la tecla con ord 27 (esc)
    has_frame, frame = source.read()  # leemos un fotograma del vídeo
    if not has_frame:  # lo comprobamos
        break
    frame = cv2.flip(frame, 1)  # giramos horizontalmente el fotograma para mejor interpretación visual de las señales
    # se recupera el tamaño del fotograma
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Cree un blob 4D a partir de un fotograma.
    '''Estamos haciendo un preprocesamiento en el fotograma, llamando a este método blobFromImage. lo que 
    realiza es un preprocesamiento en la imagen de entrada y ponerla en el formato adecuado para que luego podamos 
    realizar inferencias en esa imagen. argumentos:
    - fotograma de la imagen
    - factor de escala (1.0) no tiene que ser el mismo siempre
    - ancho y alto del fotograma (300x300)
    - valor medio que se va a restar de todos los fotogramas
    - cambio de flag swapRB (rojo azul), en este caso no es necesario porque caffemodel y OpenCV usan la misma conveción
     para los 3 canales de color
    - Recorte de argumento de entrada, indica que puede recortar su imagen de entrada para que tenga el tamaño correcto 
    o puede cambiar su tamaño, al ponerlo a False, significa que simplemente vamos a cambiar el tamaño de la imagen para
     300x300
     La llamada a la función devuelve una representación del blob del fotograma con el pre-procesamiento realizado'''
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Corremos el modelo
    net.setInput(blob)  # pasamos el blob a esta función, establecemos la entrada, prepara para la inferencia
    detections = net.forward()  # Avanza a través de la red, realiza la inferencia sobre la representación del fotograma

    for i in range(detections.shape[2]):  # para las detecciones devueltas por la inferencia las recorre
        confidence = detections[0, 0, i, 2]
        # Determina si la competencia de una detección particular excede el umbral de detección establecido
        if confidence > conf_threshold:
            '''si lo hace profundiza y consulta en la lista de detecciones las coordenadas del fotograma de esa  
            detección en particular.'''
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)
            '''Genera un cuadro delimitador ( rectángulo) con los puntos de coordenadas obtenidos, así como un texto con
             el % de confiaza de la detección'''
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # y lo dibuja en el fotograma
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    '''Una vez que ha terminado de procesar todas las detecciones, llama a getPerfProfile, que devuelve el tiempo 
    necesitado para realizar la inferencia, lo convertimos a milisegundos y lo introduce en la imagen'''
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)  # muestra el fotograma relleno en la ventana de salida

source.release()
cv2.destroyWindow(win_name)
