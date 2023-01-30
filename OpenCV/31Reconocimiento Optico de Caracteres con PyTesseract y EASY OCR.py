#!/usr/bin/env python
# coding: utf-8
#########################################################################
# 30 **Reconocimiento Óptico de Caracteres con PyTesseract & EASY OCR**#####
#########################################################################
# - En esta lección implementaremos OCR en algunas imágenes usando PyTesseract
#
# ![](https://miro.medium.com/max/1400/1*X7RfC5wOZ-Gsoo95Ez1FvQ.png)
# Fuente - https://medium.com/@balaajip/optical-character-recognition-99aba2dad314

# #### **Install PyTesseract **
#  librería de código abierto, a alto nivel toma una entrada de imagen, reconoce el texto de la misma, lo detecta ,
#  trata y limpia devolviendo un texto como una cadena
'''# Install PyTesseract and setup on Colab
get_ipython().system('sudo apt install tesseract-ocr')
get_ipython().system('pip install pytesseract')
get_ipython().system('pip install easyocr')'''


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from pytesseract import Output
from easyocr import Reader
import pandas as pd
import time

# importamos las librerías instaladas en sistema
pytesseract.pytesseract.tesseract_cmd = (
    r'/usr/bin/tesseract'
)

# Definir nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Descargar y descomprimir nuestras imágenes
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/OCRSamples.zip')
get_ipython().system('unzip -qq OCRSamples.zip')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/Receipt-woolworth.jpg')

get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/whatsapp_conv.jpeg')
'''

# ## **Nuestra primera prueba de OCR**
img = cv2.imread('images/OCR Samples/OCR1.png')
imshow("Input Image", img)

# Ejecutar nuestra imagen a través de PyTesseract, esta línea es la que realiza la detección, procesamiento y devuelve
# el texto de la imagen
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # PyTesseract Extracted: Welcome to OCR


# ## **¿Funciona el texto blanco sobre fondo negro?**
img = cv2.imread('images/OCR Samples/OCR2.png')
imshow("Input Image", img)
# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # PyTesseract Extracted: Welcome to OCR


# ## **¿Qué pasa con los fondos más desordenados?**
img = cv2.imread('images/OCR Samples/OCR3.png')
imshow("Input Image", img)

# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)

print("PyTesseract Extracted: {}".format(output_txt))  # no funciona


# ## **¿Qué tal un escaneo de la vida real?**
img = cv2.imread('images/OCR Samples/scan2.jpeg')
imshow("Input Image", img, size = 48)

# Ejecutar nuestra imagen a través de PyTesseract
output_txt = pytesseract.image_to_string(img)
print("PyTesseract Extracted: {}".format(output_txt))  # funciona hasta cierto punto


# **Necesitamos limpiar nuestras imágenes**
image = cv2.imread('images/OCR Samples/scan2.jpeg')
imshow("Input Image", image, size = 48)

# Obtenemos el componente Valor del espacio de color HSV
# luego aplicamos umbralización adaptativa a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Aplicar la operación umbral
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh, size = 48)

output_txt = pytesseract.image_to_string(thresh)
print("PyTesseract Extracted: {}".format(output_txt))  # se ve mucho mejor


# ### **Umbralizar ayuda mucho**
# Típicamente un buen pipeline de preprocesamiento para reconocimiento OCR contendrá algunos o más de los siguientes
# procesos:
# 1. Desenfoque
# 2. Umbralización
# 3. Desenfoque
# 4. Dilatación/Erosión/Apertura/Cierre
# 5. 5. Eliminación de ruido

### **Dibujemos sobre regiones reconocidas por PyTesseract**


image = cv2.imread('images/Receipt-woolworth.jpg')

# Obtenemos el componente Valor del espacio de color HSV
# luego aplicamos umbralización adaptativa a
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Aplicar la operación umbral
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)

output_txt = pytesseract.image_to_string(thresh)
print("PyTesseract Extracted: {}".format(output_txt))




d = pytesseract.image_to_data(thresh, output_type = Output.DICT)
print(d.keys())  # dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top',
#                             'width', 'height', 'conf', 'text'])
# Usando este diccionario, podemos obtener de cada palabra detectada en la imagen, la información (con coordenadas) de
# su cuadro delimitador, el texto que contiene y las puntuaciones de confianza de cada una.
#
n_boxes = len(d['text'])

# recorre los elementos (etiquetas del diccionario)
for i in range(n_boxes):
    # si la confianza obtenida es mayor de 60
    if int(d['conf'][i]) > 60:
        # extrae de esa etiqueta  las coordenadas y el tamaño para poder dibujar su cuadro delimitador
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # dibuja el rectángula verde
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# muestra la imagen
imshow('Output', image, size = 12)


# ## **EASY OCR**
# esta librería funciona mejor que la anterior, pero necesita más capacidad de procesamiento, siendo lento en CPU pero
# pudiendo aprovechar la GPU
# ### **Instalar OpenCV antiguo (EasyOCR no es compatible con el último OpenCV aquí en Colab)**
'''get_ipython().system('pip uninstall opencv-python -y')
get_ipython().system('pip install opencv-python-headless==4.1.2.30')
'''

# ## **Detectar Texto en Imagen y Mostrar nuestra Imagen de Entrada**

# cargar la imagen de entrada desde el disco
image = cv2.imread("images/whatsapp_conv.jpeg")
imshow("Original Image", image, size = 12)

# OCR de la imagen de entrada utilizando EasyOCR
print("Detecting and OCR'ing text from input image...")
# reader importado de EasyOcr, va a buscar texto en inglés ['en'] y vamos a usarlo sin GPU
# se descarga y utiliza automáticamente el modelo
reader = Reader(['en'], gpu = False)


ts = time.time()
results = reader.readtext(image) # introducimos en results todo el texto detectado ( en "cajas")
# [ ([[24, 12], [192, 12], [192, 38], [24, 38]], 'bmobile _ill < 82', 0.12457801531641248)
#   ([[396, 12], [510, 12], [510, 38], [396, 38]], '"\'0 ^ (50%', 0.33694383567965347) ...

te = time.time()  # usamos time y para comprobar cuanto tiempo ha tardado en realizar el procesamiento de la imagen
td = te - ts
print(f'Completed in {td} seconds')  # 7.667776107788086 seconds


# ## **Mostrar texto superpuesto a nuestra imagen**
all_text = []

# iterar sobre el texto extraído
for (bbox, text, prob) in results:
    # mostrar el texto OCR y la probabilidad asociada de que sea texto
    print(f" Probability of Text: {prob*100:.3f}% OCR'd Text: {text}")

    # obtener las coordenadas del cuadro delimitador
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # Elimina los caracteres no ASCII del texto para que, uniendo el no eliminado con join y usando el codepoint de
    # los caracteres detectados
    # podamos dibujar el recuadro que rodea el texto superpuesto a la imagen original
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    all_text.append(text)
    cv2.rectangle(image, tl, br, (255, 0, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# mostrar la imagen de salida
imshow("OCR'd Image", image, size = 12)


# ## **Ejecutar en nuestro WoolWorth Reciept**

def clean_text(text):
    # elimina el texto no ASCII para que podamos dibujar el texto en la imagen
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

image = cv2.imread('images/Receipt-woolworth.jpg')

reader = Reader(["en","ar"], gpu=False)
results = reader.readtext(image)

# bucle sobre los resultados
for (bbox, text, prob) in results:
    # mostrar el texto OCR y la probabilidad asociada
    print("[INFO] {:.4f}: {}".format(prob, text))

    # descomprimir el cuadro delimitador
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # limpia el texto y dibuja el recuadro que lo rodea a lo largo de
    text = clean_text(text)
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


#Aplicar la operación umbral
#thresh = (V > T).astype("uint8") * 255
imshow("EASY OCR", image)
print("EASY OCR Extracted: {}".format(text))
'''
...
[INFO] 0.0220: 900
[INFO] 0.9213: Woolworths
[INFO] 0.2355: The fregh food
[INFO] 0.6164: VICIURIA HARBOUR PH:  0383476527
[INFO] 0.7185: Store Hanager
[INFO] 0.6325: i$ ٥avid
[INFO] 0.3334: WUULWURIHS TAX INVOICE
[INFO] 0.6674: ABN 88 000 014 675
...EASY OCR Extracted: .50/k9
'''


