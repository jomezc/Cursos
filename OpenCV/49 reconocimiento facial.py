#!/usr/bin/env python
# coding: utf-8

# ******************************
# ***** 40 Reconocimiento facial
# ******************************

# En esta lección, implementaremos **simples Reconocimientos Faciales usando la librería de python face-recognition**.
#
# 1. Instalar `face-recognition` #
# 2. Comprobar similitud facial
# 3. Reconocer caras en una imagen

'''
get_ipython().system('pip install face-recognition')
'''
# ## **2. Comprobar la similitud facial entre dos caras**


# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



import cv2
from matplotlib import pyplot as plt

biden = cv2.imread('images/biden.jpg')
biden2 = cv2.imread('images/biden2.jpg')
trump = cv2.imread('images/trump2.jpeg')

imshow('Trump', trump)
imshow('Biden', biden)
imshow('Biden', biden2)


# ### **Ahora probemos con las dos imágenes anteriores**


import face_recognition

known_image = face_recognition.load_image_file("images/biden.jpg")
unknown_image = face_recognition.load_image_file("images/trump2.jpeg")

# ponemos la imagen en la primera posición
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# [biden_encoding] con lo que podríamos coger muchas imágenes diferentes para tomar la salida
result = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# indexamos el primer resultado porque solo queremos comparar con la primera cara

print(f'Face Match is {result[0]}')  # Face Match is False


# ### **Ahora probemos con las dos imágenes de Biden**
import face_recognition

known_image = face_recognition.load_image_file("images/biden.jpg")
unknown_image = face_recognition.load_image_file("images/biden2.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

result = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(f'Face Match is {result[0]}')  # Face Match is True


# ## **3. Reconocer caras en una imagen**
import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carga una imagen de ejemplo y aprende a reconocerla.
trump_image = face_recognition.load_image_file("images/trump2.jpeg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

# Carga una segunda imagen de ejemplo y aprende a reconocerla.
biden_image = face_recognition.load_image_file("images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Crear matrices de codificaciones de caras conocidas y sus nombres
known_face_encodings = [
    trump_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Donald Trump",
    "Joe Biden"
]

# Inicializar algunas variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# Obtener un único fotograma de vídeo
#frame = cv2.imread('images/biden2.jpg')
frame = cv2.imread('images/Trump.jpg')
# Redimensiona el fotograma de vídeo a 1/4 de tamaño para un procesamiento más rápido del reconocimiento facial
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convertir la imagen de color BGR (que utiliza OpenCV) a color RGB (que utiliza face_recognition)
rgb_small_frame = small_frame[:, :, ::-1]

# Sólo procesa cada dos fotogramas de vídeo para ahorrar tiempo
if process_this_frame:
    # Encuentra todas las caras y codificaciones de caras en el fotograma actual del vídeo
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Ver si la cara coincide con la(s) cara(s) conocida(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Si se ha encontrado una coincidencia en codificaciones_cara_conocidas, utiliza sólo la primera.
        # if True in coincidencias:
        # first_match_index = matches.index(True)
        # nombre = nombres_cara_conocidos[indice_primera_pareja]

        # O en su lugar, utilizar la cara conocida con la menor distancia a la nueva cara
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)


# Mostrar los resultados
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Vuelve a escalar las localizaciones de caras ya que el fotograma en el que detectamos se escaló a 1/4 de tamaño
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Dibuja una caja alrededor de la cara
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Dibuja una etiqueta con un nombre debajo de la cara
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Mostrar la imagen resultante
imshow('Face Recognition', frame)

