
# ******************************************
# ***** 16 Usando la camara en OpenCV
# ******************************************
import cv2
import sys

# especificamos un índice de dispositivo de cámara predeterminado de cero.
s = 0
print(sys.argv)  # contiene los argumentos de la librería sys, por ejemplo 0 es la ruta
# ['C:\\Users\\jgomcano\\PycharmProjects\\guiapython\\OpenCV\\Usando la camara en openCV\\16 Usando_camara_OpenCV e importar videos youtube.py']
# y simplemente estamos verificando si hubo una especificación de línea de comando para anular ese valor predeterminado.
if len(sys.argv) > 1:
    s = sys.argv[1]
print(s)  # 0
source = cv2.VideoCapture(s)  # llamamos a la clase de captura de video para crear un objeto de captura de video,
#  Con el índice 0 accederá a la cámara predeterminada en su sistema, si no hay que indicarlo
win_name = 'Vista de camara'
# estamos creando una ventana con nombre, que eventualmente vamos a enviar la salida transmitida
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

'''ciclo while nos permitirá transmitir continuamente video desde la cámara y enviarlo a la salida a menos que el 
usuario pulse la tecla de escape.'''
while cv2.waitKey(1) != 27:  # Escape
    '''usa esa fuente de objeto de captura de vídeo  de captura de video para llamar al método read, que  devolverá un 
    solo cuadro de la transmisión de video, así como una variable lógica has_frame.
    Entonces, si hay algún tipo de problema con la lectura de la transmisión de video o el acceso a la cámara, entonces 
    has_frame sería falso y saldríamos del bucle.
    De lo contrario, continuaríamos y llamaríamos a la función de visualización de mensajes instantáneos y abriríamos
     kbps para enviar el video (frame) a la ventana de salida'''
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)

### generar un boceto

# Nuestra función generadora de bocetos
def sketch(image):
    # Convierte la imagen a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Limpia la imagen usando Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Extraer bordes
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # Invertir y binarizar la imagen
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


# Inicializar webcam, cap es el objeto proporcionado por VideoCapture
cap = cv2.VideoCapture(0)

while True:
    # Contiene un booleano indicando si tuvo éxito (ret)
    # También contiene las imágenes recogidas de la webcam (frame)
    ret, frame = cap.read()
    # Pasamos nuestro frame a nuestra función sketch directamente dentro de cv2.imshow()
    cv2.imshow('Nuestro dibujante en vivo', sketch(frame))
    if cv2.waitKey(1) == 13:  # 13 es la tecla Enter
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()



## ```cap.get(id)```
# Es un método donde id es un número del 0 al 18. Cada número denota una propiedad del vídeo (si es aplicable a ese vídeo)
# **Puedes ver que los índices 3 y 4 corresponden a las dimensiones del vídeo**
import cv2

cap = cv2.VideoCapture('./videos/drummer.mp4')

for i in range(0,18):
    print(cap.get(i))

# **Capturando Video Usando Capturas de Pantalla**
#
# #### **En esta lección aprenderemos a utilizar una clase para permitir la reconexión automática a un flujo de vídeo**

# #### **Instrucciones de instalación:**
#
# MacOS o Linux
# 1. pip install Pillow
# 2. sudo -H pip install pyscreenshot
#
# **Windows**
# 1. pip install Pillow
# 2. pip install pyscreenshot

# # **Capturar una sola imagen de pantalla**

import pyscreenshot as ImageGrab

# grab fullscreen
im = ImageGrab.grab()

# save image file
im.save('fullscreen.png')

# ## **Capture Video from Screen**
import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()

while (True):

    # frame = np.array(ImageGrab.grab(bbox=(0,0,300,300)))
    frame = np.array(ImageGrab.grab())
    # Obtener pantalla completa
    # frame = np.array(ImageGrab.grab())

    # Mostrar tasa de FPS
    FPS = 1.0 / (time.time() - last_time)
    print('FPS = {}'.format(FPS))
    last_time = time.time()

    # Mostrar pantalla
    cv2.imshow('window', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        print("Exited...")
        break

cv2.destroyAllWindows()

# ## **¿Necesitas una velocidad de fotogramas súper rápida? Usa MSS**
#
# ```pip install mss```


import numpy as np
import cv2
from mss import mss
from PIL import Image
import time

bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}

sct = mss()
frame_count = 0
last_time = time.time()

while True:
    frame_count += 1
    sct_img = sct.grab()
    cv2.imshow('screen', np.array(sct_img))

    # Mostrar tasa de FPS
    if frame_count % 30 == 0:
        FPS = 1.0 / (time.time() - last_time)
        print('FPS = {}'.format(FPS))
    last_time = time.time()

    if cv2.waitKey(1) == 13:  # 13 es la tecla Enter
        print("Exited...")
        break

cv2.destroyAllWindows()

# # **Importar videos de YouTube (incluyendo transmisiones en vivo) a OpenCV**
#
# #### **En esta lección aprenderemos:**
# 1. Cómo usar la Biblioteca Pafy para importar videos de YouTube en Opencv
# 2. Cómo sacar datos META de videos de youtube
# 3. Descarga el video o audio desde un enlace de youtube

# **Necesitarás instalar:**
# 1. pip install pafy
# 2. pip install youtube-dl
#
# https://pypi.org/project/pafy/

# ## Mostrar un video de YouTube en OpenCV


import cv2
import pafy

url = 'https://youtu.be/QC8iQqtG0hg'
video = pafy.new(url)

best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture()
capture.open(best.url)

while (True):
    ret, frame = capture.read()
    if ret == True:
        cv2.imshow('src', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

# Suelte la cámara y cierre las ventanas
capture.release()
cv2.destroyAllWindows()

# ### Obtener datos META de vídeo

import pafy

url = 'https://youtu.be/QC8iQqtG0hg'
video = pafy.new(url)

print("Title: {}".format(video.title))
print("Rating: {}".format(video.rating))
print("Viewcount: {}".format(video.viewcount))
print("Author: {}".format(video.author))
print("Length: {}".format(video.length))
print("Duration: {}".format(video.duration))

# ### Ver los flujos disponibles


# In[ ]:


streams = video.streams

for s in streams:
    print(s.resolution)
    print(s.extension)
    print(s.get_filesize())
    print(s.url)

# ### Obtenga la transmisión de la más alta calidad

best = video.getbest()
best.resolution, best.extension

best.url

# ### Download Videos

best.download(quiet=False)

# ### Obtener y descargar audio

audiostreams = video.audiostreams
for a in audiostreams:
    print(a.bitrate, a.extension, a.get_filesize())

audiostreams[1].download()

