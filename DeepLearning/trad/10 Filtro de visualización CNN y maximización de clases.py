#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Maximización de filtros y clases**
#
# ---
#
#
# En esta lección, usamos **Keras con TensorFlow 2.0** para visualizar lo siguiente (ver más abajo). Esto lo ayuda a obtener una mejor comprensión de lo que sucede debajo del capó y desmitifica algunos de los aspectos del aprendizaje profundo.**
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
# 2. Usarás Stochastic Gradient Descent para ajustar los valores de la imagen de entrada para maximizar este valor de activación.
#
# **NOTA** Esto es más fácil de implementar en TF1.14, por lo que degradaremos nuestro paquete Tensorflow para que funcione.

# # **Visualización de maximizaciones de filtros de conversión**
#
# Primero, necesitamos instalar tf-keras-vis. https://github.com/keisen/tf-keras-vis

# En 1]:


get_ipython().system('pip install --upgrade tf-keras-vis tensorflow')


# **Importar nuestras bibliotecas**

# En 2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))


# ### **Cargar un modelo VGG16 preentrenado.**

# En 3]:


from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Cargar modelo
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **En primer lugar, definimos una función para modificar el modelo**
#
# Defina el modificador para reemplazar la salida del modelo por la salida de la capa de destino que tiene los filtros que desea visualizar.

# En[4]:


layer_name = 'block5_conv3' # La capa de destino que es la última capa de VGG16.

def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


# ### **Crear instancia de maximización de activación**
#
# Si el argumento de clonación es Verdadero (predeterminado), el modelo se clonará, por lo que la instancia del modelo NO se modificará, pero requiere recursos de la máquina.

# En[5]:


from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model, model_modifier, clone=False)


# ### **Definir la función Pérdida**
# DEBE definir la función de pérdida que devuelve valores de filtro arbitrarios. Aquí, devuelve el valor del tercer filtro correspondiente en la capa block5_conv3. ActivationMaximization maximizará el valor del filtro.

# En[6]:


filter_number = 7
def loss(output):
    return output[..., filter_number]


# ### **Visualizar**
# ActivationMaximization maximizará el valor de salida del modelo calculado por la función de pérdida. Aquí, tratamos de visualizar un filtro convolucional.

# En[7]:


get_ipython().run_cell_magic('time', '', "from tf_keras_vis.utils.callbacks import Print\n\n# Generar activación máxima\nactivación = activación_maximización(pérdida, devoluciones de llamada=[Imprimir(intervalo=50)])\nimagen = activación[0].astype(np.uint8)\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),\n 'subplot_kw': {'xticks': [], 'yticks': []} }\n \nf, ax = plt.subplots (**subplot_args)\nax.imshow(imagen)\nax.set_title('filtro[{:03d}]'.format(filter_number), fontsize=14)\nplt.tight_layout()\nplt.show()\n ")


# ## **Ahora visualicemos múltiples filtros convolucionales**
#
# #### **Definir función de pérdida**
# Al visualizar múltiples filtros convolucionales, DEBE definir la función de pérdida que devuelve valores de filtro arbitrarios para cada capa.

# En[8]:


filter_numbers = [63, 132, 320]

# Definir la función de pérdida que devuelve múltiples salidas de filtro.
def loss(output):
    return (output[0, ..., 63], output[1, ..., 132], output[2, ..., 320])


# #### **Crear valores SeedInput**
#
# Y luego, DEBE preparar el valor de entrada inicial. De forma predeterminada, al visualizar un filtro de conversión, tf-keras-vis genera automáticamente una entrada inicial para generar una imagen. Al visualizar múltiples filtros de conv, DEBE generar manualmente una entrada semilla cuya muestra-dim sea tanto como el número de filtros que desea generar.

# En[9]:


# Definir entradas de semillas cuya forma es (muestras, altura, ancho, canales).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualizar**
#
# Aquí, visualizaremos 3 imágenes mientras que la opción de pasos es 512 para obtener imágenes claras.

# En[10]:


get_ipython().run_cell_magic('time', '', "\n# Generar activación máxima\nactivaciones = activación_maximización(pérdida,\n seed_input=seed_input, # Para generar múltiples imágenes\n devoluciones de llamada=[Imprimir(intervalo=50)])\nimágenes = [activación.astype(np.uint8) para activación en activaciones]\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),\n 'subplot_kw': {'xticks': [], 'yticks ': []} }\nf, ax = plt.subplots(**subplot_args)\nfor i, filter_number in enumerate(filter_numbers):\n ax[i].set_title('filter[{:03d}]'.format (número_filtro), tamaño de fuente=14)\n ax[i].imshow(imágenes[i])\n \nplt.tight_layout()\nplt.show()\n")


# # **Maximización de clases**
#
# Encontrar una entrada que maximice una clase específica de VGGNet.
#
# #### **Cargar bibliotecas y cargar su modelo VGG16 preentrenado**
#
# Cargar tf.keras.Model¶
# Este tutorial usa el modelo VGG16 en tf.keras, pero si desea usar otros tf.keras.Models, puede hacerlo modificando la sección a continuación.
#

# En[11]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))

from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Cargar modelo
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Definir una función para modificar el modelo**
#
# Definir modificador para reemplazar una función softmax de la última capa a una función lineal.

# En[12]:


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


# #### **Crear instancia de maximización de activación**
#
# Si el argumento de clonación es Verdadero (predeterminado), el modelo se clonará, por lo que la instancia del modelo NO se modificará, pero requiere recursos de la máquina.

# En[13]:


from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)


# #### **Definir función de pérdida**
#
# DEBE definir la función de pérdida que devuelve un valor de categoría arbitrario. Aquí, tratamos de visualizar una categoría como se define No.20 (ouzel) de imagenet.
#
#

# En[14]:


def loss(output):
    return output[:, 20]


# ### **Visualizar**
#
# Las clases de Imagenet - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# En[15]:


get_ipython().run_cell_magic('time', '', "\nfrom tf_keras_vis.utils.callbacks import Print\n\nactivation = activation_maximization(loss,\n                                     callbacks=[Print(interval=50)])\nimage = activation[0].astype(np.uint8)\n\nsubplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),\n                 'subplot_kw': {'xticks': [], 'yticks': []} }\nf, ax = plt.subplots(**subplot_args)\nax.imshow(image)\nax.set_title('Ouzel', fontsize=14)\nplt.tight_layout()\nplt.show()\n")


# ### **Visualización de categorías de salida específicas**
#
# ¡Ahora, visualicemos varias categorías a la vez!
#
# #### **Definir función de pérdida**
#
# DEBE definir una función de pérdida que devuelva valores de categoría arbitrarios. Aquí, tratamos de visualizar las categorías definidas No.1 (Goldfish), No.294 (Bear) y No.413 (Rifle de asalto) de imagenet.
#

# En[16]:


image_titles = ['Goldfish', 'Bear', 'Assault rifle']

def loss(output):
    return (output[0, 1], output[1, 294], output[2, 413])


# #### **Crear valores SeedInput**
# Y luego, DEBE preparar el valor de entrada inicial. De forma predeterminada, al visualizar un filtro de conversión, tf-keras-vis genera automáticamente una entrada inicial para generar una imagen. Al visualizar múltiples filtros de conv, DEBE generar manualmente una entrada semilla cuya muestra-dim sea tanto como el número de filtros que desea generar.

# En[17]:


# Definir entradas de semillas cuya forma es (muestras, altura, ancho, canales).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualizar**
#
# Aquí, visualizaremos 3 imágenes mientras que la opción de pasos es 512 para obtener imágenes claras.

# En[18]:


get_ipython().run_cell_magic('time', '', "\n# Hacer 500 iteraciones y generar una animación de optimización\nactivaciones = activación_maximización(pérdida,\n entrada_semilla=entrada_semilla,\n pasos=512,\n devoluciones de llamada=[ Imprimir(intervalo=50)])\nimágenes = [activación.astype(np .uint8) para activación en activaciones]\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),\n 'subplot_kw': {'xticks' : [], 'yticks': []} }\nf, ax = plt.subplots(**subplot_args)\nfor i, title in enumerate(image_titles):\n ax[i].set_title(title, fontsize=14 )\n ax[i].imshow(imágenes[i])\nplt.tight_layout()\n\nplt.show()\n")


# En[ ]:




