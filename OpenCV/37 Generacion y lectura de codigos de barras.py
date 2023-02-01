#########################################################################
# 37 **Generación y lectura de códigos de barras**#####
#########################################################################

# - In this lesson we'll to create barcodes of various standards as well reading what's on them.

# In[1]:


'''# Our Setup, Import Libaries, Create our Imshow Function and Download our Images

get_ipython().system('pip install python-barcode[images]')
get_ipython().system('pip install qrcode')
get_ipython().system('apt install libzbar0')
get_ipython().system('pip install pyzbar')
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from barcode import EAN13
from barcode.writer import ImageWriter

# Define nuestra función imshow

def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ## **Generación de códigos de barras**
# los códigos de barras son solo representaciones lineales del símbolo de la información codificada que podemos
# decodificar de un texto, dígitos o cualquier tipo de información relativa a los códigos de barras de las tiendas.
#
# O podemos almacenar información de manera efectiva.
# Vamos a generar códigos de barras usando nuestro paquete python-barcode.
# Formatos soportados
# En el momento de escribir esto, este paquete soporta los siguientes formatos:
# - EAN-8
# - EAN-13
# - EAN-14
# UPC-A
# - JAN
# - ISBN-10
# - ISBN-13
# - ISSN
# Código 39
# - Código 128
# - PZN

with open('images/barcode.png', 'wb') as f:
    # el número es una entidad para guardar el resultado, podemos introducir el que queramos
    EAN13('123456789102', writer=ImageWriter()).write(f)

barcode = cv2.imread("images/barcode.png")
imshow("Barcode", barcode)


# ## **Generación de Códigos QR**
# Vamos a generar Códigos QR usando nuestro paquete qrcode.

# Un código QR (abreviatura de Quick Response code) es un tipo de código de barras matricial (o código de barras
# bidimensional) diseñado por primera vez en 1994 para la industria del automóvil en Japón. Un código de barras es una
# etiqueta óptica legible por máquina que contiene información sobre el artículo al que está adherido. En la práctica,
# los códigos QR suelen contener datos para un localizador, identificador o rastreador que apunta a un sitio web o una
# aplicación. Un código QR utiliza cuatro modos de codificación estandarizados (numérico, alfanumérico, byte/binario y
# kanji) para almacenar datos de forma eficiente; también se pueden utilizar extensiones.
#
# Un código QR consiste en cuadrados negros dispuestos en una cuadrícula cuadrada sobre un fondo blanco, que pueden ser
# leídos por un dispositivo de imagen como una cámara, y procesados utilizando la corrección de errores Reed-Solomon
# hasta que la imagen puede ser interpretada adecuadamente. A continuación, se extraen los datos necesarios de los
# patrones presentes en los componentes horizontal y vertical de la imagen.
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/QR_Code_Structure_Example_3.svg/800px-QR_Code_Structure_Example_3.svg.png)

import qrcode
from PIL import Image  # librería similar a OpenCV pero no tan extensa

# **Configuración de los códigos QR**:
#
# - version - Controla el tamaño del Código QR. Acepta un número entero de 1 a 40. La versión 1 consiste en una matriz
#             de 21 x 21.
# - error_correction - Controla la corrección de errores utilizada para el Código QR.
# - box_size - Controla el número de píxeles de cada caja del código QR.
# - border - Controla el grosor del borde de las cajas. El valor por defecto es 4, que es también el valor mínimo según
#            la especificación.
#
# Hay 4 constantes disponibles para error_correction. Cuanto mayor sea la corrección de errores, mejor será.
# pero a mayor corrección de errores menos información puedes almacenar en el codigo.
# - ERROR_CORRECT_L - Alrededor del 7% o menos errores pueden ser corregidos.
# - ERROR_CORRECT_M - Alrededor del 15% o menos errores pueden ser corregidos. Este es el valor por defecto.
# ERROR_CORRECT_Q - Se pueden corregir un 25% o menos de errores.
# ERROR_CORRECT_H - Alrededor del 30% o menos de errores pueden ser corregidos.

# creando el objeto QR code con la configuración
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

# Vamos a generar un código QR con la web de openCV
qr.add_data("https://wwww.opencv.org")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("images/qrcode.png")

qrcode = cv2.imread("images/qrcode.png")
imshow("QR Code", qrcode, size = 8)




# ## **Descifrar códigos QR**
from pyzbar.pyzbar import decode
from PIL import Image
img = Image.open('images/qrcode.png')
# decodifica la imagen con la función decode del módulo pyzbar de la librería pyzbar
result = decode(img)
for i in result:
    print(i.data.decode("utf-8"))  # https://wwww.opencv.org


# ### **Detección de códigos QR**
from pyzbar.pyzbar import decode

image = cv2.imread("images/1DwED.jpg")

# Detectar y decodificar el qrcode
codes = decode(image)

# bucle sobre los códigos de barras detectados
for bc in codes:
  # Obtener los rectángulos coordiantes para la colocación del texto
  (x, y, w, h) = bc.rect
  print(bc.polygon)
  pt1,pt2,pt3,pt4 = bc.polygon

  # Dibuja una caja delimitadora sobre nuestro código QR detectado
  pts = np.array( [[pt1.x,pt1.y], [pt2.x,pt2.y], [pt3.x,pt3.y], [pt4.x,pt4.y]], np.int32)
  pts = pts.reshape((-1,1,2))
  cv2.polylines(image, [pts], True, (0,0,255), 3)

  # extraer los datos de información de la cadena y el tipo de nuestro objeto
  barcode_text = bc.data.decode()
  barcode_type = bc.type

  # mostrar nuestro
  text = "{} ({})".format(barcode_text, barcode_type)
  cv2.putText(image, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  cv2.putText(image, barcode_type, (x+w, y+h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  print("QR Code revealed: {}".format(text))

# mostrar nuestra salida
imshow("QR Scanner", image, size = 12)



image = cv2.imread("images/1024px-ISBN.jpg")

# Detectar y decodificar el qrcode
barcodes = decode(image)

# bucle sobre los códigos de barras detectados
for bc in barcodes:
  # Obtener los rectángulos coordiantes para la colocación del texto
  (x, y, w, h) = bc.rect
  cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

  # extraer los datos de información de la cadena y el tipo de nuestro objeto
  barcode_text = bc.data.decode()
  barcode_type = bc.type

  # Mostrar nuestro
  text = "{} ({})".format(barcode_text, barcode_type)
  cv2.putText(image, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  cv2.putText(image, barcode_type, (x+w, y+h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
  print("Barcode revealed: {}".format(barcode_text))
  print("Barcode revealed: {}".format(barcode_text))

# mostrar nuestra salida
imshow("QR Scanner", image, size = 16)






