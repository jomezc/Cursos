import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# ******************************************
# ***** Escribir video en el disco
# ******************************************


source = './race_car.mp4'  # source = 0 for webcam
cap = cv2.VideoCapture(source)  # llamamos a la clase de captura de video para crear un objeto de captura de video,

# Comprobamos si se creó correctamente el objeto y está abierto
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# ****** Leer y mostrar un frame
'''
Los 3 puntos ... son una Ellipsis en Pyhton significan que puedes recibir los que sea y ya el ultimo valor (en este caso
 el ultimo array) , es iterar dándole la vuelta empezando por el ultimo para darle la vuelta a los canales,  sería igual
que [:,:, ::-1]
https://realpython.com/python-ellipsis/
'''
ret, frame = cap.read()
plt.imshow(frame[..., ::-1])
plt.show()

# Mostrar el video del archivo en jupyter
# from IPython.display import HTML
# HTML("""
# <video width=1024 controls>
#   <source src="race_car.mp4" type="video/mp4">
# </video>
# """)

# **** Mostrar el video del archivo desde consola mediante ventana
#
# win_name = 'video'
# # estamos creando una ventana con nombre, que eventualmente vamos a enviar la salida transmitida
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#
# '''ciclo while nos permitirá transmitir continuamente video desde la cámara y enviarlo a la salida a menos que el
# usuario pulse la tecla de escape.'''
# while cv2.waitKey(1) != 27:  # Escape
#     '''usa esa fuente de objeto de captura de vídeo  de captura de video para llamar al método read, que  devolverá un
#     solo cuadro de la transmisión de video, así como una variable lógica has_frame.
#     Entonces, si hay algún tipo de problema con la lectura de la transmisión de video o el acceso a la cámara,
#     has_frame sería falso y saldríamos del bucle.
#     De lo contrario, continuaríamos y llamaríamos a la función de visualización de mensajes instantáneos y abriríamos
#      kbps para enviar el video (frame) a la ventana de salida'''
#     has_frame, frame = cap.read()
#     if not has_frame:
#         break
#     cv2.imshow(win_name, frame)
#
# cap.release()
# cv2.destroyWindow(win_name)


# **** Escribir el vídeo usando OpenCV ( ojo con no haber ya recorrido el objeto de video )
'''
Para escribir el video, debe crear un objeto de videowriter con los parámetros correctos.

Sintaxis de la función
VideoWriter objeto = cv.VideoWriter (nombre de archivo, fourcc, fps, frameSize)
Parámetros
-filename: Nombre del archivo de vídeo de salida.
-fourcc: código de códec de 4 caracteres que se utiliza para comprimir los fotogramas.
 Por ejemplo, VideoWriter::fourcc('P','I','M','1') es un códec MPEG-1, VideoWriter::fourcc('M','J','P','G ') es un códec
 jpeg de movimiento, etc. La lista de códigos se puede obtener en la página Video Codecs by FOURCC. El backend FFMPEG 
 con contenedor MP4 usa de forma nativa otros valores como código fourcc: consulte ObjectType, por lo que puede recibir 
 un mensaje de advertencia de OpenCV sobre la conversión del código fourcc.
- fps: velocidad de fotogramas de la transmisión de video creada.
- frameSize: Tamaño de los fotogramas de vídeo tupla (ancho,alto).

*El tamaño del marco es importante porque deben ser las dimensiones de los marcos que tiene en la memoria que desea 
 escribir en el disco


Lo primero que vamos a hacer es usar el objeto de captura de video para llamar a este método de get(), que
nos va a recuperar las dimensiones del cuadro de video que tenemos en memoria.'''
# Se obtienen las resoluciones predeterminadas del cuadro, int() Convierte las resoluciones de float a entero
frame_width = int(cap.get(3))  # en 3 guarda el ancho
frame_height = int(cap.get(4))  # en 4 guarda el alto

# Define el códec y crea el objeto VideoWriter.
out_avi = cv2.VideoWriter('race_car_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out_mp4 = cv2.VideoWriter('race_car_out.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

# Leer fotogramas y escribir en el archivo
'''Leeremos los cuadros del video del auto de carreras y escribiremos lo mismo en los dos objetos que creamos en el paso
 anterior. Deberíamos liberar los objetos después de completar la tarea.'''

# leer mientras el video se completa
while (cap.isOpened()):
  # Va capturando frame a frame
  ret, frame = cap.read()

  if ret == True:

    # Escribe cada frame en los ficheros
    out_avi.write(frame)
    out_mp4.write(frame)

  # rompe el bucle
  else:
    break

# Cuando todo esté listo, liberamos los objetos VideoCapture y VideoWriter
cap.release()
out_avi.release()
out_mp4.release()