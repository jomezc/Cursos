#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **DeepFace: edad, sexo, expresión, posición de la cabeza y reconocimiento**
#
# ---
#
#
# En esta lección, usamos **DeepFace API para edad, género, expresión facial y reconocimiento. Incluso usamos la biblioteca headpose para obtener la dirección/inclinación de la cabeza**. DeepFace es un módulo de Python fácil de usar que proporciona acceso a varios modelos de detección y reconocimiento facial. Es muy simple de usar para sumergirnos.
#
# 1. Instala los módulos necesarios y descarga nuestros archivos
# 2. Demuestra puntos de referencia faciales
# 3. Obtenga edad, género, expresión emocional y etnicidad usando DeepFace
# 4. Realice la similitud facial
# 5. Realizar reconocimiento facial
#
#
# **NOTA** Cambie a la configuración de RAM alta.
#

### **1. Instale los módulos necesarios y descargue nuestros archivos**


''''pip install deepface'
'pip install dlib'''


# #### **Definir nuestra función imshow**

# Algunas importaciones y nuestra función de visualización de imágenes
import dlib
import tarfile
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


'''# Descargar puntos de referencia faciales
'wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/shape_predictor_68_face_landmarks.dat')


# Descargue nuestras imágenes de prueba y una foto de prueba en otro apartado anterior de puede descargar face_recognition

'wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/me.jpg'''


### **2. Demostrar puntos de referencia faciales**


from imutils import face_utils

p = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('images/me.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# Obtener caras
rects = detector(gray, 0)

# Para cada rostro detectado, encuentre el punto de referencia.
for (i, rect) in enumerate(rects):
    # Hacer la predicción y transformarla en una matriz numpy
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Dibuja en nuestra imagen, todos los puntos de coordenadas encontrados (x,y)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Mostrar la imagen
imshow("Output", image)


### **3. Obtenga edad, género, expresión emocional y etnicidad usando DeepFace**
#
# **Descarga nuestros modelos**

# En[ ]:

'''DESCARGAR
 https://drive.google.com/uc?id=1Id32-d-nS9BooBLLkw1PQhvLWWAukCsq
	 https://drive.google.com/uc?id=1txWignSWdELl8cWdZHYqIlSE2ZRjI8WI
	 https://drive.google.com/uc?id=1d_tQRWjvQ5i4lZyUfFEfRj7LzXWXseBY 
	 https://drive.google.com/uc?id=1kWp2CVg_xTIFqdZAwfN_86A3grim9NyI '''


from deepface import DeepFace

obj = DeepFace.analyze(img_path =  "images/face_recognition/training_faces/Nidia_1.jpg", actions = ['age', 'gender', 'race', 'emotion'])
print(obj) # estaba mal el código, seguramente por deprecado, no era un diccionario sino una lista con un diccionario dentro



from deepface import DeepFace
import pprint

img_path = "images/face_recognition/training_faces/Nidia_1.jpg"
image = cv2.imread(img_path)

obj = DeepFace.analyze(img_path = img_path,
                       actions = ['age', 'gender', 'race', 'emotion'])
imshow("Face Analysis", image)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Crea una función simple para mostrar nuestros resultados en la imagen**

import cv2

def drawFace(img_path, obj):
  image = cv2.imread(img_path)
  # por lo comentado antes con print(obj)
  # estaba mal el código, seguramente por deprecado, no era un diccionario sino una lista con un diccionario dentro
  # se ppne [0]
  x = obj[0]['region']['x']
  y = obj[0]['region']['y']
  h = obj[0]['region']['h']
  w = obj[0]['region']['w']
  age = obj[0]['age']
  gender = obj[0]['gender']
  gender = 'F' if gender == 'Woman' else 'M'
  dominant_emotion = obj[0]['dominant_emotion']
  dominant_race = obj[0]['dominant_race']
  dominant_emotion = obj[0]['dominant_emotion']
  description = f'{age}{gender} - {dominant_emotion}'
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(image, description, (x,y-10) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  cv2.putText(image, dominant_race, (x,y+h+30) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  imshow("Face Analysis", image)


# #### **Prueba en otra imagen**



from deepface import DeepFace
import pprint

img_path = "images/face_recognition/training_faces/Nidia_4.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'])
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Cambiar backends de detección de rostros**



from deepface import DeepFace

# backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

img_path = "images/me.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = "images/me.jpg", actions = ['age', 'gender', 'race', 'emotion'], detector_backend = 'mtcnn')
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


### **4. Realizar similitud facial**

result  = DeepFace.verify("images/face_recognition/training_faces/Nidia_1.jpg", "images/face_recognition/training_faces/Nidia_3.jpg")
print("Is verified: ", result["verified"])
result


# ### **Incluso podemos usar diferentes métricas de distancia**

#métricas = ["coseno", "euclidiana", "euclidiana_l2"]

result  = DeepFace.verify("images/face_recognition/training_faces/Nidia_1.jpg", "images/face_recognition/training_faces/Nidia_3.jpg", distance_metric = 'euclidean')
print("Is verified: ", result["verified"])
result


#métricas = ["coseno", "euclidiana", "euclidiana_l2"]

result  = DeepFace.verify("images/face_recognition/training_faces/Nidia_1.jpg", "images/face_recognition/training_faces/Nidia_3.jpg", distance_metric = 'euclidean_l2')
print("Is verified: ", result["verified"])
result


# ### **Descargue modelos ya que el descargador DeepFace existente dejó de funcionar**
''' Descargar pesos
https://drive.google.com/uc?id=1OdJNKL85CCYStVi9XtJRpHhXo2FU6Gf1 
https://drive.google.com/uc?id=1GWIuvW3Vm3wMpGGEyTT7sU-c1cVWZIEc
'''
### **5. Realizar reconocimiento facial**


from deepface import DeepFace
import pandas as pd

df = DeepFace.find(img_path = "images/face_recognition/training_faces/Nidia_1.jpg", db_path = 'images/face_recognition/training_faces/', detector_backend = 'ssd')
df


# ## **Incluso podemos probar algunos modelos diferentes**


from deepface import DeepFace
import pandas as pd

'''dfs = []
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

for model in models:
   df = DeepFace.find(img_path = "images/face_recognition/training_faces//Nidia_1.jpg", db_path = 'images/face_recognition/training_faces//', model_name = model,  detector_backend = 'ssd')
   df['model'] = model
   dfs.append(df)

pd.concat(dfs)


imshow('1', cv2.imread('images/face_recognition/training_faces/training_faces/Nidia_1.jpg'))
imshow('1', cv2.imread('images/face_recognition/training_faces/training_faces/Nidia_5.jpg'))
'''

# Aquí hay un excelente tutorial sobre cómo construir un sistema basado en MongoDB para el reconocimiento
# facial https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/
