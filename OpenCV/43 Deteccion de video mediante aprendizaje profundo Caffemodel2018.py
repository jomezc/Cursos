# ********************************************************
# ***** 43 Detección de video mediante aprendizaje profundo
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
import cv2
import sys

# Establece el índice para la cámara si no se introduce otro por parámetro.
s = 0
# s = 'video/pr.mp4'
# s = 'rtsp://10.9.0.31/videodevice'
if len(sys.argv) > 1:
    s = sys.argv[1]

# crea un objeto de captura de vídeo
source = cv2.VideoCapture(s)

# crea una ventana de salida para enviar todos los resultados a la pantalla
win_name = 'Prueba_de_concepto'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# **** Escribir el vídeo usando OpenCV ( ojo con no haber ya recorrido el objeto de video)
'''
Para escribir el video, debe crear un objeto de videowriter con los parámetros correctos.

Sintaxis de la función
VideoWriter objeto = cv.VideoWriter (nombre de archivo, fourcc, fps, frameSize)
Parámetros
-filename: Nombre del archivo de vídeo de salida.
-fourcc: código de códec de 4 caracteres que se utiliza para comprimir los fotogramas.
 Por ejemplo, VideoWriter::fourcc('P','I','M','1') es un códec MPEG-1, VideoWriter::fourcc('M','J','P','G ') es un códec
 jpeg de movimiento, etc. La lista de códigos se puede obtener en la página Video Codecs by FOURCC. El backend FFMPEG 
 con contenedor MP4 usa de forma nativa otros valores como código fourcc: consulte ObjectType, por lo que puede recibir 
 un mensaje de advertencia de OpenCV sobre la conversión del código fourcc.
- fps: velocidad de fotogramas de la transmisión de video creada.
- frameSize: Tamaño de los fotogramas de vídeo tupla (ancho,alto).

*El tamaño del marco es importante porque deben ser las dimensiones de los marcos que tiene en la memoria que desea 
 escribir en el disco


Lo primero que vamos a hacer es usar el objeto de captura de video para llamar a este método de get(), que
nos va a recuperar las dimensiones del cuadro de video que tenemos en memoria.'''
# Se obtienen las resoluciones predeterminadas del cuadro, int() Convierte las resoluciones de float a entero
frame_width = int(source.get(3))  # en 3 guarda el ancho
frame_height = int(source.get(4))  # en 4 guarda el alto

# Define el códec y crea el objeto VideoWriter.
out_mp4 = cv2.VideoWriter('Video_camara.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

# frozen_inference_graph2.pb, que es el archivo de pesos para el modelo.
modelFile = "modelos/SSDs/frozen_inference_graph2.pb"
# Archivo de configuración para la red (Ultimo encontrado)
configFile = "modelos/SSDs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
# etiquetas de clase para el conjunto de datos que se usó para entrenar este modelo
classFile = "modelos/SSDs/coco_class_labels.txt"

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
    rows = im.shape[0]
    cols = im.shape[1]

    # NUEVO probar para fallo 
    WHRatio = 330/300
    # Recorta el marco si es necesario, ya que no redimensionamos la entrada sino que tomamos una entrada cuadrada
    cols = im.shape[1]
    rows = im.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    im = im[y1:y2, x1:x2]

    cols = frame.shape[1]
    rows = frame.shape[0]
    # fin nuevo

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

    return im


while cv2.waitKey(1) != 27:  # mientras no pulsemos la tecla con ord 27 (esc)
    try:
        has_frame, frame = source.read()  # leemos un fotograma del vídeo
        if not has_frame:  # lo comprobamos
            break
        # frame = cv2.flip(frame, 1)  # giramos horizontalmente el fotograma para mejor interpretación visual de las señales
        # se recupera el tamaño del fotograma
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        objects = detect_objects(net, frame)



        frame = display_objects(frame, objects, 0.2)

        '''Una vez que ha terminado de procesar todas las detecciones, llama a getPerfProfile, que devuelve el tiempo 
        necesitado para realizar la inferencia, lo convertimos a milisegundos y lo introduce en la imagen'''
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow(win_name, frame)  # muestra el fotograma relleno en la ventana de salida

        # **** Escribe cada frame en el fichero
        out_mp4.write(frame)

    except Exception as e:
        print(e)


'''Cuando todo esté listo, liberamos los objetos VideoCapture y VideoWriter'''
source.release()
out_mp4.release()
cv2.destroyWindow(win_name)