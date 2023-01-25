#!/usr/bin/env python
# coding: utf-8

############################################
# 26 Simple Rastreo de Objetos por Color######
############################################
# 1. Cómo usar un Filtro de Color HSV para Crear una Máscara y luego Rastrear nuestro Objeto Deseado

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/bmwm4.mp4')
'''

# Rastreo de objetos
import cv2
import numpy as np

# Initalizar cámara
# cap = cv2.VideoCapture(0)

# definir rango de color en HSV, establecemos (visto en 21) un filtro para el color amarillo
lower = np.array([20,50,90])
upper = np.array([40,255,255])

# Crear matriz de puntos vacía, son los puntos que se van a rastrear para que pueda ver una línea.
# Hay una línea histórica de puntos de seguimiento.
points = []

# Obtener el tamaño por defecto de la ventana de la cámara

# Cargar flujo de video, clip largo
cap = cv2.VideoCapture('videos/bmwm4.mp4')

# Obtener la altura y anchura del fotograma (se requiere que sea un interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('bmwm4_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0
radius = 0


# primero filtra y luego introduce los contronos encontrados a raiz de ello en la salida ( visto en 11  y en 12)
# específicamente el controno más grande ( linea 82), es decir El cuadro más grande alrededor de uno de los objetos
# amarillos en la pantalla.
while True:
  
    # Capturar fotograma webcame
    ret, frame = cap.read()
    if ret:
      hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # Umbral de la imagen HSV para obtener sólo los colores verdes
      mask = cv2.inRange(hsv_img, lower, upper)
      #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

      #
      contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      # Crea una matriz de centros vacía para almacenar el centro de masa del centroide
      center =   int(Height/2), int(Width/2)

      if len(contours) > 0:

          # Obtener el contorno más grande y su centro
          # obtenga el área, el radio, para un círculo de cierre mínimo para el contorno.
          c = max(contours, key=cv2.contourArea)
          (x, y), radius = cv2.minEnclosingCircle(c) # radius obtentiene el punto y el radio
          M = cv2.moments(c)
          
          # A veces los contornos pequeños de un punto provocan un error de división por cero
          try:
              center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

          except:
              center =   int(Height/2), int(Width/2)

          # Permitir sólo los contadores que tengan un radio superior a 25 píxeles
          if radius > 25:
              
              # Dibuja un circulo y deja el ultimo centro creando un rastro
              cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
              cv2.circle(frame, center, 5, (0, 255, 0), -1)
              
          # Registrar los puntos del centro
          points.append(center)
      
      # Si el radio es suficientemente grande, usamos 25 píxeles
      # almacenamos todos los puntos aquí y luego dibujamos una línea.
      # Así que esa línea es básicamente el seguimiento histórico.
      if radius > 25:
          
          # bucle sobre el conjunto de puntos rastreados
          for i in range(1, len(points)):
              try:
                  cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
              except:
                  pass
              
          # Hacer cero el recuento de fotogramas
          frame_count = 0
              
      out.write(frame)
      # en el vídeo de ejemplo empieza rasteando una zona que tendra algo de amarillo y cuando
      # encuentra el coche lo sigue, lo pierde y lo vuelve a seguir
    else:
      break

# Libera la cámara y cierra las ventanas abiertas
cap.release()
out.release()





