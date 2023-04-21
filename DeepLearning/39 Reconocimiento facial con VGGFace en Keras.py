#!/usr/bin/env python
# codificación: utf-8


# # **Reconocimiento facial con VGGFace en Keras**
#

# En esta lección, usamos **Reconocimiento facial con VGGFace en Keras** para comparar la similitud facial. Cargamos un
# modelo previamente entrenado de VGGFace (entrenado en miles de caras) y lo usamos, junto con una métrica de similitud,
# para definir si dos caras son de la misma persona.
#
# 1. Descarga nuestros datos e importa nuestros módulos
#2. Definir nuestro Modelo VGGFace y cargar nuestros pesos
# 3. Crea nuestra función de distancia coseno
# 4. Verifica la similitud facial
# 5. Reconocimiento facial con One Shot Learning
# 6. Modelo de prueba usando su cámara web
# 7. Prueba en video del programa Friends TV
#
# Documento relacionado - https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
#

### **1. Descarga nuestros datos e importa nuestros módulos**

# En[ ]:
'''wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/face_recognition.zip'''
'''unzip -q face_recognition.zip'''


# Importar nuestras bibliotecas
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


### **2. Definir nuestro modelo VGGFace y cargar nuestros pesos**

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))


# ### **Cargue nuestros pesos faciales VGG**
# No necesitamos entrenar a nuestro modelo si podemos obtener los 'pesos' ya entrenados.

#puedes descargar los pesos preentrenados desde un enlace el siguiente enlace ... :(
#de la documentación detallada https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

model.load_weights('models/vgg_face_weights.h5')


### **3. Crea nuestra función de distancia coseno**
# ![Imagen de similitud de coseno](https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/cosine.JPG)
# ![Imagen de la fórmula de similitud del coseno](https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/cosinesim.JPG)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Nuestro modelo que genera el vector de incrustación 2622
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


### **4. Verificar similitud facial**
#
# #### **Defina nuestra función de verificación de caras donde cargamos imágenes de caras y las comparamos.**
#
# Configuramos **epsilon** para que sea el umbral de si nuestras dos caras son la misma persona. Establecer un valor
# más bajo lo hace más estricto con nuestra coincidencia de rostros.


epsilon = 0.40

def verifyFace(img1, img2):
    # Obtener incrustación/codificación para face1 y face2
    img1_representation = vgg_face_descriptor.predict(preprocess_image((img1)))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image((img2)))[0,:]
    
    # Calcular la similitud del coseno entre las dos incrustaciones
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img((img1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img( (img2)))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    
    print("Cosine similarity: ",cosine_similarity)
    
    # Si la puntuación de similitud es menor que el umbral épsilon
    if(cosine_similarity < epsilon):
        print("They are same person")
    else:
        print("They are not same person!")


# ### **Hagamos algunas SOCKET**


# Comparemos dos caras de la misma persona
verifyFace("images/face_recognition/group_of_faces/face_nidia2.jpg", "images/face_recognition/people/nidia2.jpg")

# Ahora comparemos su rostro con el de Jennifer Lopez
verifyFace("images/face_recognition/group_of_faces/face_nidia2.jpg", "images/face_recognition/people/jlo.jpg")

# Y ahora Jennifer Lopez con Lady Gaga
verifyFace("images/face_recognition/people/jlo.jpg", "images/face_recognition/people/ladygaga.jpg")


# # **5. Reconocimiento facial con One Shot Learning**
# ### **Extrae caras de fotos de personas**
#
# #### **Instrucciones:**
# 1. Coloque fotos de personas (una cara visible) en la carpeta llamada "./personas"
# 2. Reemplace mi foto titulada "Rajeev.jpg" con una imagen de su rostro para probar en una cámara web
# 3. Las caras se extraen usando el modelo de detector haarcascade_frontalface_default
# 4. Las caras extraídas se colocan en la carpeta llamada "./group_of_faces"
# 5. Estamos extrayendo las caras necesarias para nuestro modelo de aprendizaje de una sola vez, cargará 5 caras extraídas

'''gdown --id 1_X-V1Lp6qMAl_-9opsseieprD3Lhdq8U''' # desde colab


# Nuestra configuración, importar bibliotecas, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 8):
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()


# El siguiente código extrae caras de las imágenes y las coloca en la carpeta
import os
from os import listdir
from os.path import isfile, join

# Crear una función para configurar los directorios en los que almacenaremos nuestras imágenes
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        pass

# Cargando el detector de rostros HAARCascade
face_detector = cv2.CascadeClassifier('models/Haarcascades/haarcascade_frontalface_default.xml')

# Directorio de imágenes de personas de las que extraeremos rostros
mypath = "images/face_recognition/people/"
image_file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("Collected " + str(len(image_file_names)) + " images")

for image_name in image_file_names:
    person_image = cv2.imread(mypath+image_name)
    face_info = face_detector.detectMultiScale(person_image, 1.3, 5)
    for (x,y,w,h) in face_info:
        face = person_image[y:y+h, x:x+w]
        roi = cv2.resize(face, (128, 128), interpolation = cv2.INTER_CUBIC)
    
    path = "images/face_recognition/group_of_faces/" + "face_" + image_name
    imshow("face", roi)


# ### **Cargar nuestro VGGFaceModel**
# - Este bloque de código define el modelo VGGFace (que usaremos más adelante) y carga el modelo


#autor Sefik Ilkin Serengil
#puede encontrar la documentación de este código en el siguiente enlace:
# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

import numpy as np
import cv2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from os import listdir

def preprocess_image(image_path):
    """ Carga la imagen desde la ruta y la redimensiona"""
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

#puedes descargar pesas preentrenadas desde https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
from tensorflow.keras.models import model_from_json
model.load_weights('models/vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

model = vgg_face_descriptor

print("Model Loaded")


### **6. Modelo de prueba usando su cámara web**
# Este código busca las caras que extrajo en la carpeta "group_of_faces" y usa la similitud (similitud del coseno)
# para detectar qué caras son más similares a la que se extrae con su cámara web.

# **se ha quitado por motivos de deprecación y rapidez la webcam por una imagen fija**

from IPython.display import display, Javascript

from IPython.display import Image
try:
  filename = cv2.imread('images/face_recognition/people/ladygaga.jpg')
  print('Saved to {}'.format(filename))

  # Muestra la imagen que se acaba de tomar.
  display(Image(filename))
  plt.imshow(filename)
  plt.show()
except Exception as err:
  # Se arrojarán errores si el usuario no tiene cámara web o si no la tiene
  # otorgar permiso a la página para acceder a ella.
  print(str(err))


#apunta a tus caras extraídas
people_pictures = "images/face_recognition/group_of_faces/"

all_people_faces = dict()

for file in listdir(people_pictures):
    person_face, extension = file.split(".")
    try:
      all_people_faces[person_face] = model.predict(preprocess_image('images/face_recognition/group_of_faces/%s.jpg' % (person_face)))[0,:]
    except:
      pass

print("Face representations retrieved successfully")

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

img = cv2.imread('images/face_recognition/people/ladygaga.jpg')
faces = face_detector.detectMultiScale(img, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # dibuja un rectángulo en la imagen principal
    detected_face = img[int(y):int(y+h), int(x):int(x+w)] # recortar cara detectada
    detected_face = cv2.resize(detected_face, (224, 224)) # cambiar el tamaño a 224x224

    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255

    captured_representation = model.predict(img_pixels)[0,:]

    found = 0
    for i in all_people_faces:
        person_name = i
        representation = all_people_faces[i]

        similarity = findCosineSimilarity(representation, captured_representation)
        if(similarity < 0.35):
            cv2.putText(img, person_name[5:], (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            found = 1
            break

    # conectar cara y texto
    cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255, 0, 0),1)
    cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255, 0, 0),1)

    if(found == 0): # si la imagen encontrada no está en nuestra base de datos de personas
        cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

imshow('img',img)
