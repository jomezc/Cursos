#!/usr/bin/env python
# coding: utf-8
#################################################
# 17 **Detección de vehículos y peatones** ######
#################################################

# # **Detección de vehículos y peatones**
#
# ####**En esta lección aprenderemos:**
# 1. Usar un clasificador Haarcascade para detectar Peatones
# 2. Usar nuestros clasificadores Haarcascade en vídeos
# 3. Usar un clasificador Haarcascade para detectar Vehículos o coches


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
import IPython

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Descarga y descomprime nuestros vídeos y clasificadores Haarcascade
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/videos.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip')
get_ipython().system('unzip -qq haarcascades.zip')
get_ipython().system('unzip -qq videos.zip')
'''

# #### **Pruebas con un solo fotograma de nuestro vídeo**
# Creamos nuestro objeto capturador de vídeo
cap = cv2.VideoCapture('videos/walking.mp4')

# Lectura del primer fotograma
body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# Lectura del primer fotograma
ret, frame = cap.read()

# Ret es True si se ha leído correctamente
if ret: 

  # Escala de grises de nuestra imagen para un procesamiento más rápido
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Pasa la imagen a nuestro clasificador de cuerpos
  bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

  # Extraer las cajas delimitadoras de los cuerpos identificados
  for (x,y,w,h) in bodies:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
  
# Liberar nuestra captura de vídeo
cap.release()   
imshow("Pedestrian Detector", frame)


# #### **Prueba en nuestro clip de 15 segundos**
# **NOTA**: Tarda alrededor de 1 minuto en ejecutarse.
# Usamos cv2.VideoWriter para guardar la salida como un archivo AVI. #
# ```cv2.VideoWriter(video_output.avi, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width, height))````
# Los formatos pueden ser:
# - 'M','J','P','G' o MJPG
# - MP4V
# - X264
# - avc1


# Creamos nuestro objeto capturador de vídeo
cap = cv2.VideoCapture('videos/walking.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'walking_output.avi'.
# out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
out = cv2.VideoWriter('walking_output.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
body_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# Bucle una vez que el vídeo se ha cargado correctamente
while(True):

  ret, frame = cap.read()
  if ret: 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasar frame a nuestro clasificador de cuerpos
    bodies = body_detector.detectMultiScale(gray, 1.2, 3)

    # Extraer las cajas delimitadoras de los cuerpos identificados
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Escribe el fotograma en el archivo 'output.avi ( + mp4)
    out.write(frame)
  else:
      break

cap.release()
out.release()


# ## **Reproducir Video dentro de Colab**
# Pasos
# 1. Convertir el archivo AVI a MP4 usando FFMPEG
# 2. Cargar los plugins HTML en IPython
# 3. Mostrar nuestro reproductor de vídeo HTML


# Convertir el vídeo y mostrarlo en HTML
# IPython.get_ipython().system('ffmpeg -i /walking_output.avi walking_output.mp4 -y')



from IPython.display import HTML
from base64 import b64encode

mp4 = open('walking_output.mp4' , 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()



HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# #### **Detección de vehículos en una sola imagen**

# In[ ]:
# Creamos nuestro objeto de captura de vídeo
cap = cv2.VideoCapture('videos/cars.mp4')

# Cargar nuestro clasificador de vehículos
vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Leer primer fotograma
ret, frame = cap.read()

# Ret es True si se ha leído correctamente
if ret: 

  # Escala de grises de nuestra imagen para un procesamiento más rápido
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #Pasa la imagen a nuestro clasificador de carrocerías
  vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

  # Extraer las cajas delimitadoras de los cuerpos identificados
  for (x,y,w,h) in vehicles:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
  
# Liberar nuestra captura de vídeo
cap.release()   
imshow("Vehicle Detector", frame)


# #### **Prueba en nuestro clip de 15 segundos**

# In[ ]:


# Crear nuestro objeto de captura de vídeo
cap = cv2.VideoCapture('videos/cars.mp4')

#  Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Define el codec y crea el objeto VideoWriter.La salida se almacena en el archivo 'outpy.avi'.
# out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
out = cv2.VideoWriter('cars_output.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))

vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Bucle una vez que el vídeo se ha cargado correctamente
while(True):

  ret, frame = cap.read()
  if ret: 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasar frame a nuestro clasificador de carrocerías
    vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

    # Extraer las cajas delimitadoras de los cuerpos identificados
    for (x,y,w,h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Escribe el fotograma en el archivo 'output.avi
    out.write(frame)
  else:
      break

cap.release()
out.release()

# Convertir el vídeo y mostrarlo en HTML
# no funione en ubuntu la conversión añadida salida en
# IPython.get_ipython().system('ffmpeg -i /content/cars_output.avi cars_output.mp4 -y')
#

from IPython.display import HTML
from base64 import b64encode

mp4 = open('cars_output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# no estamos mostrando la salida pero no falla
HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)





