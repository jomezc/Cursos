#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Maximización de filtros y clases**
#
# ---
#
#
# En esta lección, usamos **Keras con TensorFlow 2.0** para visualizar lo siguiente (ver más abajo). Esto lo ayuda a
# obtener una mejor comprensión de lo que sucede debajo del capó y desmitifica algunos de los aspectos del aprendizaje
# profundo.**
# 1. Maximización de filtros
# 2. Maximización de clase
#
# **Referencias:**
#
# https://github.com/keisen/tf-keras-vis
#

#
# ## **Maximizar las activaciones de filtros**
#
# El proceso es relativamente simple en principio.
# 1. Construirás una función de pérdida que maximiza el valor de un filtro dado en una capa de convolución dada
# 2. Usarás Stochastic Gradient Descent para ajustar los valores de la imagen de entrada para maximizar este valor de
#    activación.

# # **Visualización de maximizaciones de filtros de conversión**
#
# Primero, necesitamos instalar tf-keras-vis. https://github.com/keisen/tf-keras-vis

'''
pip install --upgrade tf-keras-vis tensorflow
'''

# **Importar nuestras bibliotecas**


import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))


# ### **Cargar un modelo VGG16 preentrenado.**

from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Cargar modelo
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **En primer lugar, definimos una función para modificar el modelo**
# Defina el modificador para reemplazar la salida del modelo por la salida de la capa de destino que tiene los filtros
# que desea visualizar.



layer_name = 'block5_conv3' # La capa de destino que es la última capa de VGG16.

def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


# ### **Crear instancia de maximización de activación**
# Si el argumento de clonación es Verdadero (predeterminado), el modelo se clonará, por lo que la instancia del modelo
# NO se modificará, pero requiere recursos de la máquina.


from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model, model_modifier, clone=False)


# ### **Definir la función Pérdida**
# DEBE definir la función de pérdida que devuelve valores de filtro arbitrarios. Aquí, devuelve el valor del tercer
# filtro correspondiente en la capa block5_conv3. ActivationMaximization maximizará el valor del filtro.


filter_number = 7
def loss(output):
    return output[..., filter_number]


# ### **Visualizar**
# ActivationMaximization maximizará el valor de salida del modelo calculado por la función de pérdida. Aquí, tratamos
# de visualizar un filtro convolucional.


from tf_keras_vis.utils.callbacks import Print

# la línea image = activation[0].astype(np.uint8) fallaba arreglado activando numpy (2 líneas siguientes)
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# Generate max activation
activation = activation_maximization(loss, callbacks=[Print(interval=50)])
image = activation[0].astype(np.uint8)

# Render
subplot_args = {'nrows': 1, 'ncols': 1, 'figsize': (3, 3),
                'subplot_kw': {'xticks': [], 'yticks': []}}

f, ax = plt.subplots(**subplot_args)
ax.imshow(image)
ax.set_title('filter[{:03d}]'.format(filter_number), fontsize=14)
plt.tight_layout()
plt.show()

# ## **Ahora visualicemos múltiples filtros convolucionales**
# #### **Definir función de pérdida**
# Al visualizar múltiples filtros convolucionales, DEBE definir la función de pérdida que devuelve valores de filtro
# arbitrarios para cada capa.


filter_numbers = [63, 132, 320]

# Definir la función de pérdida que devuelve múltiples salidas de filtro.
def loss(output):
    return (output[0, ..., 63], output[1, ..., 132], output[2, ..., 320])


# #### **Crear valores SeedInput**
#
# Y luego, DEBE preparar el valor de entrada inicial. De forma predeterminada, al visualizar un filtro de conversión,
# tf-keras-vis genera automáticamente una entrada inicial para generar una imagen. Al visualizar múltiples filtros de
# conv, DEBE generar manualmente una entrada semilla cuya muestra-dim sea tanto como el número de filtros que desea
# generar.



# Definir entradas de semillas cuya forma es (muestras, altura, ancho, canales).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualizar**
#
# Aquí, visualizaremos 3 imágenes mientras que la opción de pasos es 512 para obtener imágenes claras.



# Generate max activation
activations = activation_maximization(loss,
                                      seed_input=seed_input,  # To generate multiple images
                                      callbacks=[Print(interval=50)])
images = [activation.astype(np.uint8) for activation in activations]

# Render
subplot_args = {'nrows': 1, 'ncols': 3, 'figsize': (9, 3),
                'subplot_kw': {'xticks': [], 'yticks': []}}
f, ax = plt.subplots(**subplot_args)
for i, filter_number in enumerate(filter_numbers):
    ax[i].set_title('filter[{:03d}]'.format(filter_number), fontsize=14)
    ax[i].imshow(images[i])

plt.tight_layout()
plt.show()

# # **Maximización de clases**
#
# Encontrar una entrada que maximice una clase específica de VGGNet.
# #### **Cargar bibliotecas y cargar su modelo VGG16 preentrenado**
# Cargar tf.keras.Model¶
# Este tutorial usa el modelo VGG16 en tf.keras, pero si desea usar otros tf.keras.Models, puede hacerlo modificando la
# sección a continuación.


import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))

from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Cargar modelo
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Definir una función para modificar el modelo**
# Definir modificador para reemplazar una función softmax de la última capa a una función lineal.

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


# #### **Crear instancia de maximización de activación**
# Si el argumento de clonación es Verdadero (predeterminado), el modelo se clonará, por lo que la instancia del modelo
# NO se modificará, pero requiere recursos de la máquina.

from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)


# #### **Definir función de pérdida**
# DEBE definir la función de pérdida que devuelve un valor de categoría arbitrario. Aquí, tratamos de visualizar una
# categoría como se define No.20 (ouzel) de imagenet.



def loss(output):
    return output[:, 20]


# ### **Visualizar**
#
# Las clases de Imagenet - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# En[15]:


from tf_keras_vis.utils.callbacks import Print

activation = activation_maximization(loss,
                                     callbacks=[Print(interval=50)])
image = activation[0].astype(np.uint8)

subplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
ax.imshow(image)
ax.set_title('Ouzel', fontsize=14)
plt.tight_layout()
plt.show()


# ### **Visualización de categorías de salida específicas**
#
# ¡Ahora, visualicemos varias categorías a la vez!
#
# #### **Definir función de pérdida**
#
# DEBE definir una función de pérdida que devuelva valores de categoría arbitrarios. Aquí, tratamos de visualizar las
# categorías definidas No.1 (Goldfish), No.294 (Bear) y No.413 (Rifle de asalto) de imagenet.
#

# En[16]:


image_titles = ['Goldfish', 'Bear', 'Assault rifle']

def loss(output):
    return (output[0, 1], output[1, 294], output[2, 413])


# #### **Crear valores SeedInput**
# Y luego, DEBE preparar el valor de entrada inicial. De forma predeterminada, al visualizar un filtro de conversión,
# tf-keras-vis genera automáticamente una entrada inicial para generar una imagen. Al visualizar múltiples filtros de
# conv, DEBE generar manualmente una entrada semilla cuya muestra-dim sea tanto como el número de filtros que desea
# generar.

# En[17]:


# Definir entradas de semillas cuya forma es (muestras, altura, ancho, canales).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualizar**
#
# Aquí, visualizaremos 3 imágenes mientras que la opción de pasos es 512 para obtener imágenes claras.

# En[18]:



# Do 500 iterations and Generate an optimizing animation
activations = activation_maximization(loss,
                                      seed_input=seed_input,
                                      steps=512,
                                      callbacks=[ Print(interval=50)])
images = [activation.astype(np.uint8) for activation in activations]

# Render
subplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()

plt.show()

# En[ ]:




