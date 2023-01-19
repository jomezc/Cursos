import zipfile
import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import urllib
# ****************************
# ***** Seguimiento de Objetos
# ****************************
'''Objetivo: dada la ubicación inicial de un objeto, realizar un seguimiento de la ubicación en fotogramas posteriores.

El seguimiento generalmente se refiere a estimar la ubicación de un objeto y predecir su ubicación en algún momento
futuro en el tiempo, y en el contexto de la visión por computadora, generalmente equivale a detectar un objeto de
interés en un video para posteriormente predecir la ubicación de ese objeto en cuadros de video subsiguientes Y logramos
esto mediante el desarrollo de un modelo de movimiento y un modelo de apariencia, usando esa información para predecir
su ubicación y futuros cuadros de video.

También podemos usar un modelo de apariencia que codifica el aspecto del objeto y buscar la región alrededor de la
ubicación predicha del modelo de movimiento para ajustar la ubicación del objeto. El modelo de movimiento es una
aproximación a la ubicación del objeto en un cuadro de video futuro, y  se usa el modelo de apariencia para afinar esa
estimación.

Como un ejemplo concreto, supongamos que estamos interesados en rastrear un objeto específico como el coche de carreras
identificado en el primer fotograma de un videoclip. Para iniciar el algoritmo de seguimiento, necesitamos especificar 
la ubicación inicial del objeto y para hacer esto, definimos un cuadro delimitador que se muestra aquí en azul, que 
consta de dos conjuntos de coordenadas de píxeles que definen las esquinas superior izquierda e inferior derecha del 
cuadro delimitador. uUna vez que el algoritmo de seguimiento se inicializa con esta información, el objetivo es realizar
un seguimiento del objeto y los cuadros de video subsiguientes al producir un cuadro delimitador en cada nuevo cuadro de
video.

En OpenCV tenemos 8 algoritmos de seguimiento disponibles:
1. BOOSTING
2. MIL
3. KCF
4. CRST
5. TLD -> Tiende a recuperarse de las oclusiones.
6. MEDIANFLOW -> Bueno para cámara lenta predecible
7. GOTRUN -> Basado en aprendizaje profundo, Más preciso
8. MOSSE -> El más rápido
'''
video_input_file_name = "race_car.mp4"

# *** Definición de funciones


def drawRectangle(frame, bbox):  # Cuadro delimitador, dibujar
    p1 = (int(bbox[0]), int(bbox[1]))  # punto izquierdo superior
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # punto inferior derecho
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    # imagen, vértice sup izq, vértice inf der, color(R,G,B), grosor, tipo de línea


def displayRectangle(frame, bbox):  # Cuadro delimitador, mostrar
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()  # copiamos el fotograma
    drawRectangle(frameCopy, bbox)  # Llamamos al de arriba para dibujar el rectángulo en el fotograma
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)  # cambio de color
    plt.imshow(frameCopy); plt.axis('off')  # mostramos el fotograma


def drawText(frame, txt, location, color = (50,170,50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)  # dibujamos texto en el fotograma

'''Uno de los algoritmos es el modelo GOTURN que requiere un modelo de inferencia, que se entrena teniendo como entrada
el fotograma previo el actual, pasa por el modelo de red neuronal entrenado ( conocido como modelo de inferencia) 
Utiliza el cuadro delimitador del cuadro anterior para recortar ambas imágenes y, por lo tanto, el objeto de interés se 
encuentra en el centro de este cuadro anterior. Y obviamente, si el objeto se ha movido en el marco actual, entonces no 
estará centrado en este recorte de fotograma porque estamos usando el cuadro delimitador del cuadro anterior para 
recortar ambos fotogramas. Y luego es el trabajo del modelo de inferencia predecir cuál es el cuadro delimitador en la 
salida y da como salida el fotograma de seguimiento actual.
'''
# Descargar modelo de seguimiento (solo  GOTURN)
if not os.path.isfile('goturn.prototxt') or not os.path.isfile('goturn.caffemodel'):
    print("Downloading GOTURN model zip file")
    urllib.request.urlretrieve('https://www.dropbox.com/sh/77frbrkmf9ojfm6/AACgY7-wSfj-LIyYcOgUSZ0Ua?dl=1',
                               'GOTURN.zip')

    # descomprimir el fichero
    '''
    El método extractall() se usa para extraer paratodos los archivos presentes en el archivo zip al directorio de trabajo 
    actual. Los archivos también se pueden extraer a una ubicación diferente sin pasar por el parámetro de ruta.
    sintaxis: ZipFile.extractall(ruta_archivo, miembros=Ninguno, pwd=Ninguno)
    Parámetros:
    - file_path: ubicación donde se debe extraer el archivo comprimido, si file_path es None, el contenido del archivo zip se extraerá al directorio de trabajo actual
    - miembros: Especifica la lista de archivos a extraer, si no se especifica, se extraerán todos los archivos del zip. los miembros deben ser un subconjunto de la lista devuelta por namelist()
    - pwd: la contraseña utilizada para los archivos cifrados. Por defecto, pwd es Ninguno.
    '''
    with zipfile.ZipFile("GOTURN.zip", 'r') as zObject:
        # Extracting all the members of the zip
        # into a specific location.
        zObject.extractall(
            path=None)
    # Delete the zip file
    os.remove('GOTURN.zip')

# **** Crear la instancia de Tracker
# Configurar rastreador definiendo una lista de tracker ( "rastreadores") disponibles en la API
tracker_types = ['BOOSTING', 'MIL','KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN','MOSSE']

# Cambiar el índice para cambiar el tipo de rastreador
tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy_TrackerBoosting.create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy_TrackerCSRT.create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy_TrackerTLD.create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy_TrackerMedianFlow.create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
else:
    tracker = cv2.legacy_TrackerMOSSE.create()

# ***** Leer video de entrada y configuración de salida de video

# Leer video
'''# Estamos configurando las transmisiones de video de salida de entrada, por lo que pasamos la entrada de vídeo (el 
nombre de archivo) y creando un objeto de entrada de vídeo'''
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()  # leemos el primer fotograma del archivo
# plt.imshow(frame[..., ::-1])
# plt.show()
# Salir si no se puede abrir el video
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # capturamos del fotograma el ancho
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # capturamos del fotograma el alto

video_output_file_name = 'race_car-' + tracker_type + '.mp4'  # nombre parametrizado del archivo
'''Para escribir el vídeo, creamos un objeto de salida de vídeo que escriba los resultados del algoritmo de seguimiento 
escogido 
* explicado en "Escribir video en el disco"
'''
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'avc1'), 10, (width, height))

# ****** Definir cuadro delimitador
'''Necesitábamos encontrar un cuadro delimitador alrededor del objeto que nos interesa rastrear, y lo estamos logrando 
aquí de forma manual, Pero en la práctica, seleccionaría eso con una interfaz de usuario o tal vez usaría un algoritmo 
de detección para detectar objetos de interés para el seguimiento '''
bbox = (1300, 405, 160, 120)  # Dos conjuntos de pixeles, esquina sup izq y esquina inf der
#bbox = cv2.selectROI(frame, False)
#print(bbox)
displayRectangle(frame,bbox)

# ****** Inicializar rastreador
'''
Inicializamos el rastreador y para ello llamamos a tracker.init pásandole el primer fotograma y el cuadro delimitador'''
ok = tracker.init(frame, bbox)

# ***** Marco de lectura y objeto de seguimiento
while True:
    '''Comprobamos que existe el objeto inicializado de tracker (ok) y el fotograma, además Está leyendo el siguiente 
    fotograma del vídeo'''
    ok, frame = video.read()
    if not ok:
        break

    # Start empieza el contador
    timer = cv2.getTickCount()

    '''vamos a pasar el fotograma a la función de seguimiento o actualización que nos devolverá un cuadro delimitado 
    para el objeto detectado ( en caso de encontrarlo)'''
    ok, bbox = tracker.update(frame)

    # calcular los frames por segundo (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # dibujar la caja de seguimiento si hemos detectado el objeto
    if ok:
        drawRectangle(frame, bbox)
    else:
        # si no escribiríamos el texto de fallo en el seguimiento
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # mostrar la información calculada (en 175)
    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))

    # escribir el fotograma del vídeo
    video_out.write(frame)
'''El bucle Recorre cada cuadro en el clip de video y llama a la función de actualización del rastreador y luego anota
los fotogramas y los envía al flujo de vídeo de salida.'''

video.release()
video_out.release()