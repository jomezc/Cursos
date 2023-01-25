#!/usr/bin/env python
# coding: utf-8
########################################################################
# Detección de caras y ojos con clasificadores en cascada Haar ######
########################################################################
# ####**En esta lección aprenderemos:**
# 1. A utilizar un clasificador en cascada de Haar para detectar caras
# 2. utilizar un clasificador Haarcascade para detectar ojos.
# 3. Usar un clasificador Haarcascade para detectar caras y ojos desde su webcam en Colab.


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Descargar y descomprimir nuestras imágenes y clasificadores Haarcascade
'''get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip')

get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq haarcascades.zip')
'''

# ### **Primero, ¿Qué es la Detección de Objetos?**
# ![](https://miro.medium.com/max/739/1*zlWrCk1hBBFRXa5t84lmHQ.jpeg)
#
# **Detección de Objetos** es la capacidad de detectar y clasificar objetos individuales en una imagen y dibujar un
# cuadro delimitador sobre el área del objeto.


# # **Clasificadores en cascada HAAR**
# Desarrollados por Viola y Jones en 2001.
# Método de detección de objetos que utiliza una serie de clasificadores (en cascada) para identificar objetos en una
# imagen. Están entrenados para identificar un tipo de objeto, sin embargo, podemos utilizar varios de ellos en
# paralelo, por ejemplo, detectar ojos y caras juntos.
# Los clasificadores HAAR se entrenan utilizando muchas imágenes
# positivas (es decir, imágenes con el objeto presente) e imágenes negativas (es decir, imágenes sin el objeto
# presente). Estos clasificadores son modelos pre entrenados.
# Fueron los primeros detectores de texturas ópticas de trabajo real que funcionaron bastante bien y muy
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/haar.png)

# utiliza un concepto de ventanas correderas para básicamente deslizar estas imágenes y hace una convolución en la parte
# superior de esta imagen y extrae esas características. Tenemos muchas características de bordes, líneas, rectángulos
# y muchas otras. La combinación de esas características corresponde a un rostro, y esos clasificadores son entrenados
# para identificar las diferentes secuencias.
#
# Probablemente puedo describirlo como que la secuencia de valores que corresponden a la cara de una persona, al
# menos ...lo que sea que esté entrenado. Y para entrenar esto, básicamente sólo necesitas un montón de imágenes
# positivas. Son imágenes donde el objeto está presente e imágenes negativas. Así es como aprende a diferenciar cuando
# una cara está allí y cuando una cara no está allí.
# No va a prendiendo


# Apuntamos la función CascadeClassifier de OpenCV a donde nuestro clasificador (formato de archivo XML) se almacena

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Cargamos nuestra imagen y la convertimos a escala de grises
image = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nuestro clasificador devuelve el ROI de la cara detectada como una tupla
# Almacena la coordenada superior izquierda y la coordenada inferior derecha
"""Así que hemos creado nuestro primer objeto clasificador aquí y ahora que tiene una función llamada CascadeClassifier.
Aquí es donde nos alimentamos en la imagen de entrada. El primer parámetro que podemos establecer scaleFactor, 
así como un minNeighbors. Son parámetros de configuración OPCIONALES que  ajustan la sensibilidad. con ellos se puede 
conseguir más cajas en la cara y el factor de habilidad también. Depende del tipo de imagen y el tipo de cara o el t
amaño de las caras en la imagen. Agarra la cara y extrae en una matriz."""
faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No faces found")

# Recorremos la matriz de caras y dibujamos un rectángulo
# sobre cada cara en faces
for (x,y,w,h) in faces:
    # Puntos x e y, asi como el ancho (hacia la izq) y el alto (hacia abajo), para poder calcular el rectágulo
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2) # el último es el grosor
imshow('Face Detection', image)


# ## **Detección simple de ojos y caras usando clasificadores Haarcascade**
import numpy as np
import cv2
 
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
 
img = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    # Está recortando la cara y luego lo está haciendo de manera similar para la imagen en color también.
    # Así que podemos probar ya sea en el color o gris.
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray, 1.2, 3)  # detectamos los ojos
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

imshow('Eye & Face Detection', img)

"""
# ## **Usando los fragmentos de código de Colab accedamos a la webcam para una entrada**
# Nota: Requiere que tu ordenador tenga webcam
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


# In[ ]:


from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Mostrar la imagen que se acaba de tomar.
  display(Image(filename))
except Exception as err:
    # Se lanzarán errores si el usuario no tiene webcam o si no
    # concedido permiso a la página para acceder a ella.
  print(str(err))


# In[ ]:


import numpy as np
import cv2
 
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
 
img = cv2.imread('photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Si no se detectan caras, face_classifier devuelve una tupla vacía
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

imshow('Eye & Face Detection',img)
"""

#Use su webcam para hacer la detección de caras y ojos en directo
# Esto sólo funciona en una máquina local, no funcionará en Colab

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    
    for (x,y,w,h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
            
    roi_color = cv2.flip(roi_color,1)
    return roi_color

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      

