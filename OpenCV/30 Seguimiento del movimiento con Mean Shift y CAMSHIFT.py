#!/usr/bin/env python
# coding: utf-8

################################################################
# 24 Seguimiento del movimiento con Mean Shift y CAMSHIFT ######
################################################################
# Seguimiento: Imagina que tienes una persona en movimiento o un vehículo en movimiento en un video de CCTV y quieres
# enfocarte en esa persona. Dibujas una caja y la mueves sobre la persona mientras él, ella, el coche, etc se mueve en
# el video. Eso es lo que es el seguimiento

# ####**En esta lección aprenderemos dos Algoritmos de Seguimiento de Objetos:**
# 1. Cómo usar el algoritmo Mean Shift en OpenCV
# 2. Usar CAMSHIFT en OpenCV

# In[1]:


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
      
'''get_ipython().system('wget https://github.com/makelove/OpenCV-Python-Tutorial/raw/master/data/slow.flv')
'''

# ## **Rastreo de Objetos Meanshif**
#
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_basics.jpg)
#
# La intuición detrás del meanhift es simple. Considera que tienes un conjunto de puntos. (Puede ser una distribución
# de píxeles como la retroproyección del histograma). Se le da una pequeña ventana (puede ser un círculo) y usted tiene
# que mover esa ventana a la zona de máxima densidad de píxeles (o el número máximo de puntos). Se ilustra en la imagen
# simple dada a continuación:
#
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_face.gif)
#
# El desplazamiento medio es un algoritmo de escalada que consiste en desplazar iterativamente este núcleo a una región
# de mayor densidad hasta la convergencia. Cada desplazamiento se define por un vector de desplazamiento medio. El
# vector de desplazamiento medio siempre apunta hacia la dirección del máximo incremento en la densidad.
# ![](https://upload.wikimedia.org/wikipedia/commons/b/bd/Meanshiftred.gif)
#
# Lea el artículo aquí - https://ieeexplore.ieee.org/document/732882
#
# Fuente de la animación - https://fr.wikipedia.org/wiki/Camshift

#  Es decir, estableces una ventana y la mueves iterativamente a la parte mś intensa de la trama, considerando
#  el histograma, esto es, intensidades de color en el cuadro delimitador inicial que establecimos. Acabamos de
#  establecer algunos criterios para mirar, moverse y buscar el siguiente punto más brillante alrededor de esa imagen.
# Y lo mueves iterativamente hacia el área más densa de la intensidad pudiendo utilizar intensidad de rojo,  azul, verde
# .... de saturación y espacio de color HSV.


cap = cv2.VideoCapture('videos/data_slow.flv')

# toma el primer fotograma del video
ret, frame = cap.read()

# Obtener la altura y anchura del fotograma (se requiere que sea un entero)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('car_tracking_mean_shift.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# configurar la ubicación inicial de la ventana
r,h,c,w = 250,90,400,125  # simplemente codificar los valores
track_window = (c,r,w,h)

# establecer el ROI para el seguimiento
roi = frame[r:r+h, c:c+w]  # establecemos en las coordenadas de la imagen el roi como un rectángulo con los valores conf

hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # pasamos el frame a HSV

# Usar inRange (visto) para capturar sólo los valores entre inferior y superior, es decir5 crear una máscara, un
# umbral binario en la imagen, el blanco sería un SI, entra en la máscara y el negro un NO
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# calcula el histograma del roi con la mascara establecida
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

# normaliza el resultado, para asegurarse de que, de cuadro a cuadro, sea consistente en el mismo rango.
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Establecer los criterios de terminación, ya sea 10 iteración o mover por lo menos 1 pt,
# para dejar de rastrear en ese punto. Entonces, dejamos de atender donde no está el movimiento, al menos por un
# punto, eso significa que dejamos de rastrear en ese punto.
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # pasamos el frame a HSV

        # calculamos la retroproyección para el cálculo del histograma.
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # aplicar desplazamiento medio para obtener la nueva ubicación con la imagen, la ubicación de la imagen actual
        # y los criterios de terminación establecidos
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Dibújalo en la imagen
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255),2)
        out.write(img2)
        # El cuadrado 'pequeño' es el establecido y el otro va buscando las áreas más brillantes de la imagen
        #imshow('Tracking', img2)

    else:
        break

cap.release()
out.release()




# ## **Camshift en OpenCV**
# Es casi igual que meanshift, pero devuelve un rectángulo rotado (que es nuestro resultado) y parámetros de caja
# (que se pasan como ventana de búsqueda en la siguiente iteración).
# Por lo tanto, es una forma más efectiva de seguimiento.
# ![](https://upload.wikimedia.org/wikipedia/commons/8/86/CamshiftStillImage.gif)
#
# Lea el artículo aquí - https://ieeexplore.ieee.org/document/732882
#
# Fuente de animación - https://fr.wikipedia.org/wiki/Camshift


cap = cv2.VideoCapture('videos/data_slow.flv')

# toma el primer fotograma del video
ret,frame = cap.read()

# Obtener la altura y anchura del fotograma (se requiere que sea un entero)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('car_tracking_cam_shift.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# configurar la ubicación inicial de la ventana
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# establecer el ROI para el seguimiento
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Establecer los criterios de terminación, ya sea 10 iteración o mover por lo menos 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # aplicar desplazamiento medio para obtener la nueva ubicación
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Dibújalo en la imagen diferente al anterior porque en lugar de dibujar usando el rectángulo, tenemos que
        # obtener dos puntos y dibuje el polígono de línea para el rectángulo rotado.
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        out.write(img2)
        #imshow('img2',img2)

    else:
        break

cap.release()
out.release()

