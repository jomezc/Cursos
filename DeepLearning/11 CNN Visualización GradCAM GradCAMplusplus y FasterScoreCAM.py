#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Visualizaciones de GradCAM, GradCAM++ y Faster-ScoreCAM**
#
# ---
#
#
# En esta lección, usamos **Keras con TensorFlow 2.0** para visualizar lo siguiente (ver más abajo). Esto lo ayuda a
# obtener una mejor comprensión de lo que sucede debajo del capó y desmitifica algunos de los aspectos del aprendizaje
# profundo.
#
# 1. Aprenda a usar GradCAM, GradCAM++, ScoreCAM y Faster-ScoreCAM para ver dónde 'mira' nuestra CNN
#
# **Referencias:**
#
# https://github.com/keisen/tf-keras-vis
#

# #### **Instalar bibliotecas**
#
# Primero, necesitamos instalar tf-keras-vis.
'''
pip install --upgrade tf-keras-vis tensorflow
'''

# #### **Cargar nuestras bibliotecas**
#
#


import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))


# ### **Cargar un modelo VGG16 preentrenado.**

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# Cargar modelo
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Cargar imágenes**
#
# tf-keras-vis admite la evaluación por lotes que incluye varias imágenes. Aquí, cargamos tres imágenes de peces
# dorados, osos y rifles de asalto como datos de entrada.

'''
wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/goldfish.jpg
wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/bear.jpg'
wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/soldiers.jpg'
'''
from tensorflow.keras.preprocessing.image import load_img

# Títulos de imágenes
image_titles = ['Goldfish', 'Bear', 'Assault rifle']

# Cargar imágenes
img1 = load_img('images/goldfish.jpg', target_size=(224, 224))
img2 = load_img('images/bear.jpg', target_size=(224, 224))
img3 = load_img('images/soldiers.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparando datos de entrada
X = preprocess_input(images)

# Representación
subplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.show()


#
# Las clases de Imagenet - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# ## **Definir funciones necesarias**
#
# #### **Definir funciones de pérdida**
#
# DEBE definir la función de pérdida que devuelve puntajes objetivo. Aquí, devuelve las puntuaciones correspondientes
# Goldfish, Bear, Assault Rifle.


# La variable `salida` se refiere a la salida del modelo,
# entonces, en este caso, la forma de `salida` es `(3, 1000)` es decir, (muestras, clases).
def loss(output):
    #1 es el índice imagenet correspondiente a Goldfish, el 294 a Bear y el 413 a Assault Rifle.
    return (output[0][1], output[1][294], output[2][413])


# #### **Definir función de modificador de modelo**
#
# Luego, cuando la función de activación softmax se aplica a la última capa del modelo, puede obstruir la generación
# de imágenes de atención, por lo que debe reemplazar la función por una función lineal. Aquí, lo hacemos usando
# model_modifier.

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


# ## **GradCAM**
# GradCAM es otra forma de visualizar la atención sobre la entrada. En lugar de usar gradientes con respecto a las
# salidas del modelo, usa la penúltima salida de la capa Conv (antes de la capa Densa).


from tensorflow.keras import backend as K
from tf_keras_vis.utils import normalize
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Create Gradcam object
gradcam = Gradcam(model,
                  model_modifier=model_modifier,
                  clone=False)

# Generate heatmap with GradCAM
cam = gradcam(loss,
              X,
              penultimate_layer=-1,  # model.layers number
             )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
plt.tight_layout()
plt.show()
# ## **GradCAM++**
#
# GradCAM++ puede proporcionar mejores explicaciones visuales de las predicciones del modelo CNN. En tf-keras-vis,
# la clase GradcamPlusPlus (GradCAM++) tiene la mayor parte de la compatibilidad con Gradcam. Entonces puede usar
# GradcamPlusPlus si simplemente reemplaza el nombre de clase de Gradcam a GradcamPlusPlus.
#
#

# En[14]:
from tf_keras_vis.gradcam import GradcamPlusPlus

# Create GradCAM++ object, Just only repalce class name to "GradcamPlusPlus"
# gradcam = Gradcam(model, model_modifier, clone=False)
gradcam = GradcamPlusPlus(model,
                          model_modifier,
                          clone=False)

# Generate heatmap with GradCAM++
cam = gradcam(loss,
              X,
              penultimate_layer=-1,  # model.layers number
             )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
plt.tight_layout()
plt.show()


# Como puede ver arriba, ahora, ¡las atenciones visualizadas cubren casi por completo los objetos de destino!
#
# ## **ScoreCAM**
#
# Por último, aquí le mostramos ScoreCAM. SocreCAM es otro método que genera Class Activation Map. La característica
# es que es el método CAM sin gradiente a diferencia de GradCAM/GradCAM++.
#
# De forma predeterminada, este método lleva demasiado tiempo, por lo que en la celda debajo de ScoreCAM NO se ejecuta
# con CPU.
#
#

from tf_keras_vis.scorecam import ScoreCAM

# Create ScoreCAM object
scorecam = ScoreCAM(model, model_modifier, clone=False)

# This cell takes toooooooo much time, so only doing with GPU.
if gpus > 0:
    # Generate heatmap with ScoreCAM
    cam = scorecam(loss,
                   X,
                   penultimate_layer=-1, # model.layers number
                  )
    cam = normalize(cam)

    f, ax = plt.subplots(**subplot_args)
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("NOTE: Change to GPU to see visual output\n")


# ##**ScoreCAM más rápido**
#
# Como puede ver arriba, ScoreCAM necesita una gran potencia de procesamiento, pero hay buenas noticias para nosotros.
# Faster-ScorecAM que hace que ScoreCAM sea más eficiente fue ideado por @tabayashi0117.
#
# https://github.com/tabayashi0117/Score-CAM/blob/master/README.md#faster-score-cam
#
# > Pensamos que varios canales eran dominantes en la generación del mapa de calor final. Faster-Score-CAM agrega el
# procesamiento de "usar solo canales con grandes variaciones como imágenes de máscara" a Score-CAM. (max_N = -1 es el
# Score-CAM original).

# Create ScoreCAM object
scorecam = ScoreCAM(model, model_modifier, clone=False)

# Generate heatmap with Faster-ScoreCAM
cam = scorecam(loss,
               X,
               penultimate_layer=-1, # model.layers number
               max_N=10
              )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
plt.tight_layout()
plt.show()




