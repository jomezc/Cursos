#!/usr/bin/env python
# coding: utf-8

############################################
# 25 Object Tracking with Optical Flow######
############################################
# 1. Cómo usar Optical Flow en OpenCV
# 2. Luego usar Dense Optical Flow

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking_short_clip.mp4')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/walking.avi')
'''

# ## **El algoritmo de flujo óptico Lucas-Kanade**
#
# El flujo óptico es el patrón de movimiento aparente de los objetos de la imagen entre dos fotogramas consecutivos
# causado por el movimiento del objeto o de la cámara. Se trata de un campo vectorial 2D en el que cada vector es un
# vector de desplazamiento que muestra el movimiento de los puntos del primer fotograma al segundo. Considere la s
# Siguiente imagen (Imagen cortesía: Wikipedia article on Optical Flow).
#
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Optical_flow_example_v2.png/440px-Optical_flow_example_v2.png)
#
# Muestra una bola moviéndose en 5 fotogramas consecutivos. La flecha muestra su vector de desplazamiento. El flujo
# óptico tiene muchas aplicaciones en áreas como:
#
# - Estructura a partir del movimiento
# - Compresión de vídeo
# - Estabilización de vídeo
#
# El flujo óptico funciona en varios supuestos:
#
# - Las intensidades de los píxeles de un objeto no cambian entre fotogramas consecutivos.
# - Los píxeles vecinos tienen un movimiento similar.
#
# Más información - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

# ES DECIR Busca el flujo aparente, el movimiento o la dirección de un objeto que se mueve en una imagen y entre
# fotogramas consecutivos. luego, rastrea eso con el campo vectorial 2D, donde el vector de características representa
# el desplazamiento el movimiento de puntos de fotograma a fotograma.


# Cargar flujo de vídeo, clip corto
cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Cargar flujo de vídeo, clip largo
# cap = cv2.VideoCapture('videos/walking.avi')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('optical_flow_walking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# Establecer parámetros para la detección de esquinas ShiTomasi
# ES uno de los métodos que podemos usar en el flujo óptico para identificar los puntos que necesitamos rastrear.
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parámetros para el flujo óptico lucas kanade
lucas_kanade_params = dict( winSize  = (15,15),  # tamaño de la ventana
                  maxLevel = 2, # indica la cantidad de pirámides
                            #  una herramienta de escala que se abre al usuario para que podamos ver dos, podemos hacer
                            #  que se vean diferentes habilidades y más robusto a los objetos más pequeños o más grandes
                            #  que queremos rastrear.
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Crear algunos colores aleatorios
# Usados para crear nuestras estelas para el movimiento del objeto en la imagen
color = np.random.randint(0,255,(100,3))

# Toma el primer fotograma y encuentra las esquinas en él
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Encontrar las esquinas iniciales para establecer nuestro movimiento
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

# Crear una imagen de máscara para dibujar con las dimensiones del frame
mask = np.zeros_like(prev_frame)

while(1):
    ret, frame = cap.read()

    if ret == True:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calcular el flujo óptico
        # le pasamos frame en escala de grises anterior, el actual, las esquinas previamente calculadas
        # y los parámetros establecidos anteriormente
        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                            frame_gray, 
                                                            prev_corners, 
                                                            None, 
                                                            **lucas_kanade_params)

        # Seleccionar y almacenar los puntos buenos que queremos usar ( los que tienen el estado 1 o correcto)
        good_new = new_corners[status==1]
        good_old = prev_corners[status==1]

      # Dibuja las pistas
        try:
            # Zip para hacerlo con una lista o un conjunto como este.
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel() # los aplanamos para que solo nos de 2 valores
                c, d = old.ravel()
                a, b, c, d= int(a), int(b),int(c),int(d)  # Jesus fix
                # dibujamos las líneas
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a,b), 5, color[i].tolist(),-1)

        except Exception as e:
            print(e)
        img = cv2.add(frame,mask)

        # Guardar vídeo
        out.write(img)
        # Mostrar flujo óptico
        #imshow('Optical Flow - Lucas-Kanade',img)

        # Ahora actualiza el fotograma anterior y los puntos anteriores
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1,1,2)

    else:
        break
    
cap.release()
out.release()


# **NOTE** No muestra este ejemplo el vídeo, sino el movimiento sobre un fondo negro
# 
# Este código no comprueba cómo de correctos son los siguientes puntos clave. Por lo tanto, incluso si un punto
# desaparece en la imagen, existe la posibilidad de que el flujo óptico encuentre el siguiente punto que se le parezca.
# Así que para un seguimiento robusto, los puntos de esquina deben ser detectados en intervalos particulares.

# Flujo óptico denso
# El método Lucas-Kanade calcula el flujo óptico para un conjunto de características dispersas (en nuestro ejemplo,
# esquinas detectadas usando el algoritmo Shi-Tomasi). OpenCV proporciona otro algoritmo para encontrar el flujo óptico
# denso. Calcula el flujo óptico para todos los puntos del fotograma. Se basa en el algoritmo de Gunner Farneback que
# se explica en "[Two-Frame Motion Estimation Based on Polynomial Expansion]
# (https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)"
# por Gunner Farneback en 2003.


# A continuación se muestra cómo encontrar el flujo óptico denso utilizando el algoritmo anterior.
# Obtenemos una matriz de 2 canales con vectores de flujo óptico, (u,v).
# Encontramos su magnitud y dirección.
# Coloreamos el resultado para una mejor visualización.
#
# - Dirección corresponde al valor Hue de la imagen.
# - Magnitud corresponde al plano Valor. Ver el código a continuación:


# Cargar flujo de vídeo, clip corto
# cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# # Cargar flujo de vídeo, clip largo
cap = cv2.VideoCapture('videos/walking.mp4')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('dense_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# Obtener primer fotograma
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:
    
    # Lectura del archivo de vídeo
    ret, frame2 = cap.read()

    if ret == True:
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

      # Calcula el flujo óptico denso usando el algoritmo de Gunnar Farneback
      flow = cv2.calcOpticalFlowFarneback(previous_gray, next, 
                                          None, 0.5, 3, 15, 3, 5, 1.2, 0)

      # usa el flujo para calcular la magnitud (velocidad) y el ángulo de movimiento
      # usa estos valores para calcular el color que refleje la velocidad y el ángulo
      magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = angle * (180 / (np.pi/2))
      hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

      # Guardar vídeo
      out.write(final)
      # Mostrar nuestra demo de Dense Optical Flow
      #imshow('Dense Optical Flow', final)
      
      # Guardar la imagen actual como imagen anterior
      previous_gray = next

    else:
      break
    
cap.release()
out.release()




