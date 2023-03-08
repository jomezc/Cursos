#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Googe Deep Dream en Keras Tensorflow 2.0**
#
# ---
#
#
# En esta lección, aprendemos a implementar el **Algoritmo de sueño profundo de Google** usando Keras con Tensorflow 2.0. Este método fue introducido por primera vez por Alexander Mordvintsev de Google en julio de 2015.
#
# Nos permite proporcionar el efecto 'Deep Dream' que produce efectos visuales alucinógenos. teniendo una imagen como entrada , detecta partones y los amplifica, dando como salida una imagen
#
# En este tutorial nosotros:
#
#1. Cargar nuestros módulos e imagen base
# 2. Crea nuestras utilidades de preprocesamiento y desprocesamiento de imágenes
# 3. Calcule la pérdida de Deep Dream
# 4. Configure el bucle de ascenso de gradiente para una octava
#
# Fuente: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/deep_dream.ipynb#scrollTo=jIdSV7PumdQh

### **1. Cargar módulos e imagen base**

# En 1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3
import matplotlib.pyplot as plt

base_image_path = keras.utils.get_file("castara-tobago.jpeg", "https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg")
result_prefix = "sky_dream"

# Estos son los nombres de las capas para las que tratamos de maximizar la activación,
# así como su peso en la pérdida final que tratamos de maximizar.
# Puede modificar esta configuración para obtener nuevos efectos visuales.
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# Jugar con estos hiperparámetros también te permitirá lograr nuevos efectos
step = 0.01  # Tamaño de paso de ascenso de gradiente
num_octave = 3  # Número de escalas en las que ejecutar el ascenso de gradiente
octave_scale = 1.4  # Relación de tamaño entre escalas
iterations = 20  # Número de pasos de ascenso por escala
max_loss = 15.0


# ### **Nuestra Imagen Base**

# En 2]:


from IPython.display import Image, display

display(Image(base_image_path))


# ## **El Proceso del Sueño Profundo**
#
# 1. Cargue la imagen original.
# 2. Defina un número de escalas de procesamiento ("octavas"), de menor a mayor.
# 3. Cambia el tamaño de la imagen original a la escala más pequeña.
# 4. Para cada escala, comenzando con la más pequeña (es decir, la actual):
# - Ejecutar ascenso de gradiente
# - Imagen de lujo a la siguiente escala
# - Reinyectar el detalle que se perdió en el momento de la ampliación
# 5. Deténgase cuando volvamos al tamaño original. Para obtener los detalles perdidos durante la mejora, simplemente
# tomamos la imagen original, la reducimos, la escalamos y comparamos el resultado con la imagen original
# (redimensionada).

### **2. Cree nuestras utilidades de preprocesamiento y desprocesamiento de imágenes**
#

# En 3]:


def preprocess_image(image_path):
    '''Util function to open, resize and format pictures
     into appropriate arrays.'''
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    '''Util function to convert a NumPy array into a valid image.'''
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Deshacer preprocesamiento de inception v3
    x /= 2.0
    x += 0.5
    x *= 255.0
    # Convertir a uint8 y recortar al rango válido [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


### **3. Calcule la pérdida de Deep Dream**
# Primero construimos un modelo de extracción de características para recuperar las activaciones de nuestras capas de
# destino dada una imagen de entrada.


# Cree un modelo InceptionV3 cargado con pesos de ImageNet previamente entrenados
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Obtenga las salidas simbólicas de cada capa "clave" (les dimos nombres únicos).
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# Configure un modelo que devuelva los valores de activación para cada capa de destino
# (como un dictado)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)


# El cálculo de la pérdida real es muy simple:

def compute_loss(input_image):
    features = feature_extractor(input_image)
    ''' Initialize the loss '''
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # Evitamos los artefactos de borde al involucrar solo los píxeles que no están en el borde en la pérdida.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


### **4. Configure el bucle de ascenso de gradiente para una octava **

@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # Calcular gradientes.
    grads = tape.gradient(loss, img)
    # Normalizar gradientes.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img


# ### **Ahora ejecuta el ciclo de entrenamiento, iterando sobre diferentes octavas**

original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  #  Hacer una copia
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.preprocessing.image.save_img(result_prefix + ".png", deprocess_image(img.numpy()))


# ## **Muestra tu salida final**

display(Image(result_prefix + ".png"))





