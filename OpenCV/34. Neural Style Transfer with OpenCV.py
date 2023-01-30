#!/usr/bin/env python
# coding: utf-8

# *******************************************************
# ***** 34 Transferencia de Estilos Neuronales con OpenCV
# *******************************************************
# ####**En esta lección aprenderemos a usar Modelos pre-entrenados para implementar la Transferencia Neuronal de
# Estilos en OpenCV**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/NSTdemo.png)
#
# **Acerca de la Transferencia Neuronal de Estilos**
#
# Introducido por Leon Gatys et al. en 2015, en su artículo titulado "[A Neural Algorithm for Artistic Style]
# (https://arxiv.org/abs/1508.06576)", el algoritmo Neural Style Transfer se hizo viral dando lugar a una explosión de
# trabajos posteriores y aplicaciones móviles.
#
# ¡Neural Style Transfer permite aplicar el estilo artístico de una imagen a otra! Copia los patrones de color, las
# combinaciones y las pinceladas de la imagen de origen y lo aplica a la imagen de entrada. Y es una de las
# implementaciones más impresionantes de Redes Neuronales en mi opinión.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/NST.png)



# importamos los paquetes necesarios
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Descargar y descomprimir nuestras imágenes y archivos YOLO
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/NeuralStyleTransfer.zip')
get_ipython().system('unzip -qq NeuralStyleTransfer.zip')
'''

# In[ ]:

'''
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/city.jpg')'''


### **Implementar la Transferencia Neuronal de Estilos usando Modelos preentrenados**
#
# Usamos modelos PyTorch t7 preentrenados que pueden ser importados usando ``cv2.dnn.readNetFromTouch()```
#
# Estos modelos que utilizamos provienen del artículo *Perceptual Losses for Real-Time Style Transfer and
# Super-Resolution* de Johnson et al.
#
# Mejoraron proponiendo un algoritmo Neural de Transferencia de Estilo que funcionaba 3 veces más rápido utilizando un
# problema similar a la super-resolución basado en la función de pérdida perceptual.


# Cargar nuestros modelos de transferencia neural t7
model_file_path = "modelos/NeuralStyleTransfer/models/"
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

# Cargar nuestra imagen de prueba
img = cv2.imread("images/city.jpg")

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
for (i,model) in enumerate(model_file_paths):
    # imprimir el modelo utilizado
    print(str(i+1) + ". Using Model: " + str(model)[:-3])    
    style = cv2.imread("modelos/NeuralStyleTransfer/art/"+str(model)[:-3]+".jpg")
    # cargar nuestro modelo neural style transfer
    neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path+ model)

    # Vamos a redimensionar a una altura fija de 640 (siéntete libre de cambiar)
    height, width = int(img.shape[0]), int(img.shape[1])
    newWidth = int((640 / height) * width)
    resizedImg = cv2.resize(img, (newWidth, 640), interpolation = cv2.INTER_AREA)

    # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
    inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False, crop=False)

    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    # Remodelar el tensor de salida, añadiendo de nuevo la sustracción de la media y reordenando los canales
    # Eso se suma debido a los datos en los que se entrenó el modelo.
    # estos valores preestablecidos que están codificados establecen un valor específico para este
    # modelo y los datos con los que fueron entrenados.

    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)
    
    # Mostrar nuestra imagen original, el estilo que se está aplicando y la Transposición Neural de Estilos final
    imshow("Original", img)
    imshow("Style", style)
    imshow("Neural Style Transfers", output)


# ## **Utilizando el algoritmo NST actualizado de ECCV16**
#
# En la publicación de Ulyanov et al. de 2017, *Instance Normalization: The Missing Ingredient for Fast Stylization*,
# se descubrió que cambiar la normalización de lotes por la normalización de instancias (y aplicar la normalización de
# instancias tanto en el entrenamiento como en la prueba), conduce a un rendimiento en tiempo real aún más rápido y
# podría decirse que también a resultados estéticamente más agradables.
#
# Usemos ahora los modelos utilizados por Johnson et al. en su documento ECCV.
#
#

# In[ ]:


# Cargar nuestros modelos de transferencia neural t7
model_file_path = "modelos/NeuralStyleTransfer/models/ECCV16/"
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

# Cargar nuestra imagen de prueba
img = cv2.imread("images/city.jpg")

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
for (i,model) in enumerate(model_file_paths):
    # imprimir el modelo utilizado
    print(str(i+1) + ". Using Model: " + str(model)[:-3])    
    style = cv2.imread("modelos/NeuralStyleTransfer/art/"+str(model)[:-3]+".jpg")
    # cargar nuestro modelo neural style transfer
    neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path+ model)

    # Vamos a cambiar el tamaño a una altura fija de 640 (siéntase libre de cambiar)
    height, width = int(img.shape[0]), int(img.shape[1])
    newWidth = int((640 / height) * width)
    resizedImg = cv2.resize(img, (newWidth, 640), interpolation = cv2.INTER_AREA)

    # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
    inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False, crop=False)

    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    # Remodelar el tensor de salida, añadiendo de nuevo la sustracción de la media y reordenando los canales
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)
    
    # Mostrar nuestra imagen original, el estilo que se está aplicando y la Transposición Neural de Estilos final
    imshow("Original", img)
    imshow("Style", style)
    imshow("Neural Style Transfers", output)




'''get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/dj.mp4')
'''



# Cargar nuestros modelos de transferencia neuronal t7
model_file_path = "modelos/NeuralStyleTransfer/models/ECCV16/starry_night.t7"

# Cargar flujo de vídeo, clip largo
cap = cv2.VideoCapture('modelos/NeuralStyleTransfer/dj.mp4')

# Obtener la altura y la anchura del fotograma (se requiere que sea un interger)
w = int(cap.get(3)) 
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('NST_Starry_Night.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Recorrer y aplicar cada estilo de modelo a nuestra imagen de entrada
# for (i,model) in enumerate(model_file_paths):
style = cv2.imread("modelos/NeuralStyleTransfer/art/starry_night.jpg")
i = 0
while(1):

    ret, img = cap.read()

    if ret == True:  
      i += 1
      print("Completed {} Frame(s)".format(i))
      # cargar nuestro modelo de transferencia de estilo neural
      neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path)

      # Vamos a cambiar el tamaño a una altura fija de 640 (siéntase libre de cambiar)
      height, width = int(img.shape[0]), int(img.shape[1])
      newWidth = int((640 / height) * width)
      resizedImg = cv2.resize(img, (newWidth, 640), interpolation = cv2.INTER_AREA)

      # Creamos nuestro blob a partir de la imagen y a continuación realizamos una pasada hacia delante de la red
      inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640),
                                (103.939, 116.779, 123.68), swapRB=False, crop=False)

      neuralStyleModel.setInput(inpBlob)
      output = neuralStyleModel.forward()

      # Remodelar el tensor de salida, añadiendo la resta de medias
      # y reordenando los canales
      output = output.reshape(3, output.shape[2], output.shape[3])
      output[0] += 103.939
      output[1] += 116.779
      output[2] += 123.68
      output /= 255
      output = output.transpose(1, 2, 0)

      # Mostrar nuestra imagen original, el estilo aplicado y la Transposición Neural final
      #imshow("Original", img)
      #imshow("Style", style)
      #imshow("Neural Style Transfers", output)
      vid_output = (output * 255).astype(np.uint8)
      vid_output = cv2.resize(vid_output, (w, h), interpolation = cv2.INTER_AREA)
      out.write(vid_output)
    else:
      break

cap.release()
out.release()



# ## **¿Quieres entrenar tu propio modelo NST?**
#
# ## **Mira secciones posteriores del curso donde echaremos un vistazo a la Implementación de nuestro propio
# Algoritmo NST de Aprendizaje Profundo**
#
# Alternativamente, dale una oportunidad a este repositorio de github y pruébalo tú mismo -
# https://github.com/jcjohnson/fast-neural-style

