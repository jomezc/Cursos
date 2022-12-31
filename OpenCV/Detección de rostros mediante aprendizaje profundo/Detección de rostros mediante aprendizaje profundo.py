# ********************************************************
# ***** Detección de rostros mediante aprendizaje profundo
# ********************************************************
'''Para detectar los rostros, utilizar OpenCV que nos permitirá leer en un modelo previamente entrenado y realizar
inferencias usando ese modelo'''
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
entrenar redes neuronales. Además OpenCV tiene una funcionalidad funcionalidad integrada para usar redes preentrenadas 
para realizar inferencias ( es decir, no podemos usare OpenCV  para entrenar una red neuronal, pero puede usarlo para 
realizar inferencia en una red entrenada)

la función cv2.dnn.readNetFromCaffe es una función diseñada específicamente para leer un modelo caffemodel 
'''
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
