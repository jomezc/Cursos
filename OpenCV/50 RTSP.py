# ******************************
# ***** 50 RTSP
# ******************************


'''### **Protocolo de transmisión en tiempo real RTSP**

RSPT es un protocolo de control de presentación multimedia cliente-servidor, diseñado para satisfacer las necesidades
de entrega eficiente de streaming multimedia a través de redes IP. El protocolo subyacente utilizado para RTSP es el
protocolo RTP.

RTSP fue desarrollado por RealNetworks, Netscape y la Universidad de Columbia alrededor de 1996. Es un protocolo que
se utiliza para transferir datos multimedia en tiempo real (por ejemplo, audio/vídeo) entre un cliente y un servidor.
Normalmente, un cliente solicita y el servidor responde a la solicitud con los datos a través de este protocolo.
Se trata de un protocolo de transmisión en tiempo real, lo que significa que los datos se transfieren y representan
simultáneamente en tiempo real. Aquí los datos multimedia se encapsulan en paquetes del Protocolo de Transporte en
Tiempo Real (RTP). Así que no es el RTSP el que hace el trabajo, sino el RTP. '''

import cv2

# Nuestro Enlace RSTP de Prueba Gratuito
# Puede configurar sus Cámaras IPTV CCTV para emitir un Stream RSTP
cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

while (1):
    ret, frame = cap.read()

    cv2.imshow('RTSP Stream', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()



### **¿Interesado en Ingerir Múltiples Flujos IP Fácilmente? Mira ImageZMG**
'''
- https://github.com/jeffbass/imagezmq#introduction
- https://www.pyimageconf.com/static/talks/jeff_bass.pdf'''

# Necesitará instalar ImageZMQ primero
#!pip install imagezmq
# ejecute este programa en el Mac para visualizar secuencias de imágenes de varias RPis import cv2
import imagezmq

image_hub = imagezmq.ImageHub()

while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    cv2.imshow(rpi_name, image) # 1 ventana por cada RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')


# Reconexión automática a un flujo RSTP

#### **En esta lección aprenderemos a usar una clase para habilitar la reconexión automática a un stream de video**

import cv2
import requests
import time


class VideoCapture:
    def __init__(self, cam_address, cam_force_address=None, blocking=False):
        """
        cam_address: dirección ip de la cámara de vídeo
        cam_force_address: dirección ip para desconectar otros clientes (tomar el control a la fuerza)
        blocking: si es true los métodos read() y connect_camera() se bloquean hasta que se reconecta la cámara ip
        """
        self.cam_address = cam_address
        self.cam_force_address = cam_force_address
        self.blocking = blocking
        self.capture = None

        # NOTA: Puede aumentarse para reducir la impresión
        self.RECONNECTION_PERIOD = 0.5
        # Llama al método connect
        self.connect_camera()

    def connect_camera(self):
        print("Connecting...")
        while True:
            try:
                if self.cam_force_address is not None:
                    requests.get(self.cam_force_address)

                self.capture = cv2.VideoCapture(self.cam_address)

                if not self.capture.isOpened():
                    time.sleep(self.RECONNECTION_PERIOD)
                    raise Exception("Could not connect to a camera: {0}".format(self.cam_address))

                print("Connected to a camera: {}".format(self.cam_address))

                break
            except Exception as e:
                print(e)

                if self.blocking is False:
                    break

                time.sleep(self.RECONNECTION_PERIOD)

    def getStream(self):
        """
        Lee la imagen y si no se recibe intenta reconectar la cámara
        :return: ret - bool que especifica si la imagen se ha leído correctamente
                 frame - imagen opencv de la cámara
        """

        ret, frame = self.capture.read()

        # Si se cae la señal intentamos reconectar
        if ret is False:
            self.connect_camera()

        return ret, frame

cap = VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

while (1):
    ret, frame = cap.getStream()
    # Note this will keep the loop running until you force the program to exit
    try:
        cv2.imshow('RTSP Stream', frame)
    except:
        print("Feed has gone down...")

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        print("Exited...")
        break

# Release camera and close windows
cv2.destroyAllWindows()