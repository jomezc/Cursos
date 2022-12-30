# ****************************
# ***** Seguimiento de Objetos
# ****************************
# Objetivo: dada la ubicación inicial de un objeto, realizar un seguimiento de la ubicación en fotogramas posteriores


import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import urllib

video_input_file_name = "race_car.mp4"

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

def displayRectangle(frame, bbox):
    plt.figure(figsize=(20,10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy); plt.axis('off')

def drawText(frame, txt, location, color = (50,170,50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Descargar modelo de seguimiento (solo para GOTURN)
if not os.path.isfile('goturn.prototxt') or not os.path.isfile('goturn.caffemodel'):
    print("Downloading GOTURN model zip file")
    urllib.request.urlretrieve('https://www.dropbox.com/sh/77frbrkmf9ojfm6/AACgY7-wSfj-LIyYcOgUSZ0Ua?dl=1',
                               'GOTURN.zip')

    # descomprimir el fichero
    '''
    El método extractall() se usa para extraer todos los archivos presentes en el archivo zip al directorio de trabajo 
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
# Configurar rastreador
tracker_types = ['BOOSTING', 'MIL','KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN','MOSSE']

# Cambiar el índice para cambiar el tipo de rastreador
tracker_type = tracker_types[2]

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
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()
# plt.imshow(frame[..., ::-1])
# plt.show()
# Salir si no se puede abrir el video
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = 'race_car-' + tracker_type + '.mp4'
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'avc1'), 10, (width, height))

# ****** Definir cuadro delimitador
# Definir un cuadro delimitador
bbox = (1300, 405, 160, 120)
#bbox = cv2.selectROI(frame, False)
#print(bbox)
displayRectangle(frame,bbox)

# ****** Inicializar rastreador
'''
1. Un marco
2. Un cuadro delimitador'''
ok = tracker.init(frame, bbox)

# ***** Marco de lectura y objeto de seguimiento
while True:
    ok, frame = video.read()
    if not ok:
        break

    # Start empieza el contador
    timer = cv2.getTickCount()

    ok, bbox = tracker.update(frame)

    # calcular los frames por segundo (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # dibujar la caja de seguimiento
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # mostrar la información
    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))

    # escribir el fotograma del vídeo
    video_out.write(frame)

video.release()
video_out.release()