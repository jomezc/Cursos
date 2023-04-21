#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Similitud facial usando PyTorch**
# ### **Uso de MTCNN para detección facial**
#
# ![](https://github.com/timesler/facenet-pytorch/raw/master/data/facenet-pytorch-banner.png)
#
# ---
#
#
#
#
# En esta lección, aprenderemos a usar el módulo **FaceNet-PyTorch** para realizar detección, similitud y reconocimiento de rostros simples.
# 1. Clone el repositorio e instale `facenet-pytorch`
#2. Cargar nuestros módulos y datos
# 3. Reconocimiento facial MTCNN de Perfom (Redes convolucionales en cascada multitarea)
#
#
# **Notas:**
#
# Este es un repositorio para modelos Inception Resnet (V1) en pytorch, entrenados previamente en VGGFace2 y CASIA-Webface.
#
# Los pesos del modelo de Pytorch se inicializaron usando parámetros transferidos del repositorio facenet de tensorflow
# de David Sandberg.
#
# También se incluye en este repositorio una implementación pytorch eficiente de MTCNN para la detección de rostros
# antes de la inferencia. Estos modelos también están preentrenados. Hasta donde sabemos, esta es la implementación
# de MTCNN más rápida disponible.
#
# https://github.com/timesler/facenet-pytorch#guide-to-mtcnn-in-facenet-pytorch


# IDEA
# Detector rápido de MTCNN - https://www.kaggle.com/code/timesler/fast-mtcnn-detector-55-fps-at-full-resolution/notebook

# otro ejemplo de detección de rostros + clsasificación
# https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
### **1. Clone el repositorio e instale facenet-pytorch**


'''git clone https://github.com/timesler/facenet-pytorch.git'''

'''pip install facenet-pytorch'''


### **2. Cargue nuestros módulos y datos**
#
# El siguiente ejemplo ilustra cómo usar el paquete python facenet_pytorch para realizar la detección y el
# reconocimiento de rostros en un conjunto de datos de imágenes usando un Inception Resnet V1 entrenado previamente en
# el conjunto de datos VGGFace2.
#
# Se incluyen los siguientes métodos de Pytorch:
#
# conjuntos de datos
# cargadores de datos
# Procesamiento GPU/CPU


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4


# Determinar si una GPU nvidia está disponible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# **Definir módulo MTCNN**
#
# Los parámetros predeterminados se muestran a modo de ilustración, pero no son necesarios. Tenga en cuenta que, dado
# que MTCNN es una colección de redes neuronales y otro código, el dispositivo debe pasarse de la siguiente manera para
# permitir la copia de objetos cuando sea necesario internamente.
#
# Consulte la ayuda (MTCNN) para obtener más detalles.


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)


# **Definir el módulo Inception Resnet V1**
#
# Establecer classify=True para el clasificador preentrenado. Para este ejemplo, usaremos el modelo para generar
# incrustaciones/características de CNN. Tenga en cuenta que para la inferencia, es importante establecer el modelo en
# modo eval.
#
# Consulte la ayuda (InceptionResnetV1) para obtener más detalles.


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# **Definir un conjunto de datos y un cargador de datos**
#
# Agregamos el atributo idx_to_class al conjunto de datos para permitir una fácil recodificación de índices de
# etiquetas para identificar nombres más adelante.


def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('models/facenet-pytorch/data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


# ### **Ver nuestras imágenes**

import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imgshow(title="", image = None, size = 6):
    if image.any():
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()
    else:
      print("Image not found")


for f in glob.glob('models/facenet-pytorch/data/test_images/**/*.jpg', recursive=True):
    image = cv2.imread(f)
    imgshow(f, image)


### **3. Realizar detección facial MTCNN**
#
# Iterar a través del objeto DataLoader y detectar rostros y probabilidades de detección asociadas para cada uno. El
# método de reenvío MTCNN devuelve imágenes recortadas al rostro detectado, si se detectó un rostro. De manera
# predeterminada, solo se devuelve una sola cara detectada: para que MTCNN devuelva todas las caras detectadas,
# establezca keep_all=True al crear el objeto MTCNN anterior.
#
# Para obtener cuadros delimitadores en lugar de imágenes de caras recortadas, puede llamar a la función mtcnn.detect()
# de nivel inferior. Consulte la ayuda (mtcnn.detect) para obtener más información.


aligned = []
names = []

for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])


# **Calcular incrustaciones de imágenes**
#
# MTCNN devolverá imágenes de rostros del mismo tamaño, lo que permite un fácil procesamiento por lotes con el módulo de
# reconocimiento Resnet. Aquí, dado que solo tenemos unas pocas imágenes, construimos un solo lote y realizamos
# inferencias sobre él.
#
# Para conjuntos de datos reales, el código debe modificarse para controlar los tamaños de los lotes que se pasan a
# Resnet, especialmente si se procesan en una GPU. Para SOCKET repetidas, es mejor separar la detección de rostros
# (usando MTCNN) de la incrustación o clasificación (usando InceptionResnetV1), ya que el cálculo de rostros recortados
# o cuadros delimitadores se puede realizar una sola vez y los rostros detectados se pueden guardar para uso futuro.

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()


# **Imprimir matriz de distancia para clases**

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
pd.DataFrame(dists, columns=names, index=names)



