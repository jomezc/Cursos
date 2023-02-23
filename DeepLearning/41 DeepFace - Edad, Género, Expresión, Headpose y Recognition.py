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

# En[ ]:


get_ipython().system('pip install deepface')
get_ipython().system('pip install dlib')


# #### **Definir nuestra función imshow**

# En[ ]:


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


# En[ ]:


# Descargar puntos de referencia faciales
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/shape_predictor_68_face_landmarks.dat')


# En[ ]:


# Descargue nuestras imágenes de prueba y una foto de prueba
get_ipython().system('gdown --id 1RDw1BqsuZv4auJNv3iJ4ToIOnBq9WsVZ')
get_ipython().system('unzip -q face_recognition.zip')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/me.jpg')


### **2. Demostrar puntos de referencia faciales**

# En[ ]:


from imutils import face_utils

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('me.jpg')
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


get_ipython().system('gdown --id 1Id32-d-nS9BooBLLkw1PQhvLWWAukCsq')
get_ipython().system('gdown --id 1txWignSWdELl8cWdZHYqIlSE2ZRjI8WI')
get_ipython().system('gdown --id 1d_tQRWjvQ5i4lZyUfFEfRj7LzXWXseBY')
get_ipython().system('gdown --id 1kWp2CVg_xTIFqdZAwfN_86A3grim9NyI')

get_ipython().system('mv facial_expression_model_weights.zip /root/.deepface/weights/facial_expression_model_weights.zip')
get_ipython().system('mv age_model_weights.h5 /root/.deepface/weights/age_model_weights.h5')
get_ipython().system('mv gender_model_weights.h5 /root/.deepface/weights/gender_model_weights.h5')
get_ipython().system('mv race_model_single_batch.zip /root/.deepface/weights/race_model_single_batch.zip')


# En[ ]:


from deepface import DeepFace

obj = DeepFace.analyze(img_path =  "./training_faces/Nidia_1.jpg", actions = ['age', 'gender', 'race', 'emotion'])
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])


# En[ ]:


from deepface import DeepFace
import pprint

img_path = "./training_faces/Nidia_1.jpg"
image = cv2.imread(img_path)

obj = DeepFace.analyze(img_path = img_path,
                       actions = ['age', 'gender', 'race', 'emotion'])
imshow("Face Analysis", image)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Crea una función simple para mostrar nuestros resultados en la imagen**

# En[ ]:


import cv2

def drawFace(img_path, obj):
  image = cv2.imread(img_path)
  x = obj['region']['x'] 
  y = obj['region']['y'] 
  h = obj['region']['h'] 
  w = obj['region']['w'] 
  age = obj['age']
  gender = obj['gender']
  gender = 'F' if gender == 'Woman' else 'M'
  dominant_emotion = obj['dominant_emotion']
  dominant_race = obj['dominant_race']
  dominant_emotion = obj['dominant_emotion']
  description = f'{age}{gender} - {dominant_emotion}'
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(image, description, (x,y-10) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  cv2.putText(image, dominant_race, (x,y+h+30) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  imshow("Face Analysis", image)


# #### **Prueba en otra imagen**

# En[ ]:


from deepface import DeepFace
import pprint

img_path = "training_faces/Nidia_4.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'])
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Cambiar backends de detección de rostros**

# En[ ]:


from deepface import DeepFace

# backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

img_path = "me.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = "me.jpg", actions = ['age', 'gender', 'race', 'emotion'], detector_backend = 'mtcnn')
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


### **4. Realizar similitud facial**

# En[ ]:


result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg")
print("Is verified: ", result["verified"])
result


# ### **Incluso podemos usar diferentes métricas de distancia**

# En[ ]:


#métricas = ["coseno", "euclidiana", "euclidiana_l2"]

result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg", distance_metric = 'euclidean')
print("Is verified: ", result["verified"])
result


# En[ ]:


#métricas = ["coseno", "euclidiana", "euclidiana_l2"]

result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg", distance_metric = 'euclidean_l2')
print("Is verified: ", result["verified"])
result


# ### **Descargue modelos ya que el descargador DeepFace existente dejó de funcionar**

# En[ ]:


get_ipython().system('gdown --id 1OdJNKL85CCYStVi9XtJRpHhXo2FU6Gf1')
get_ipython().system('gdown --id 1GWIuvW3Vm3wMpGGEyTT7sU-c1cVWZIEc')
get_ipython().system('mv vgg_face_weights.h5 /root/.deepface/weights/vgg_face_weights.h5')
get_ipython().system('mv facenet_weights.h5 /root/.deepface/weights/facenet_weights.h5')


### **5. Realizar reconocimiento facial**

# En[ ]:


from deepface import DeepFace
import pandas as pd

df = DeepFace.find(img_path = "./training_faces/Nidia_1.jpg", db_path = './training_faces/', detector_backend = 'ssd')
df


# ## **Incluso podemos probar algunos modelos diferentes**

# En[ ]:


from deepface import DeepFace
import pandas as pd

dfs = []
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

for model in models:
   df = DeepFace.find(img_path = "./training_faces/Nidia_1.jpg", db_path = './training_faces/', model_name = model,  detector_backend = 'ssd')
   df['model'] = model
   dfs.append(df)

pd.concat(dfs)


# En[ ]:


imshow('1', cv2.imread('./training_faces/Nidia_1.jpg'))
imshow('1', cv2.imread('./training_faces/Nidia_5.jpg'))


# Aquí hay un excelente tutorial sobre cómo construir un sistema basado en MongoDB para el reconocimiento facial https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/
