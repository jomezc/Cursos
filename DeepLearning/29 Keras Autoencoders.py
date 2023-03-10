#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Keras - Codificadores automáticos**
#
# ---
#
# En esta lección, aprendemos cómo crear codificadores automáticos en Keras.
#
# Crédito del tutorial: https://blog.keras.io/building-autoencoders-in-keras.html

# Un codificador automático es un algoritmo de aprendizaje automático no supervisado. En nuestro ejemplo, toma una
# imagen como entrada y luego intenta reconstruir esa imagen usando menos información.
#
# Hacen esto proyectando datos dimensionales más altos a una dimensión más baja (similar al Análisis de Componentes
# Principales) manteniendo las características de mayor importancia. Esto se llama el espacio latente.
#
# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522830223/AutoEncoder_kfqad1.png)
#

# ### **Un Autoencoder se compone de estas dos Redes**
#
# **Codificador**: comprime/reduce la muestra de la imagen de entrada en un número menor de bits. Este menor número
# de bits se denomina espacio latente o cuello de botella.
#
# **Decodificador**: intenta reconstruir la entrada usando solo la codificación de la entrada. Si el decodificador
# puede reconstruir la imagen con precisión a partir de la salida del codificador, tiene un codificador que funciona
# correctamente (capaz de producir buenas codificaciones) y un sistema decodificador.

### **1. Creación de un codificador automático simple**
#
# Construiremos una sola capa neuronal completamente conectada como codificador y decodificador.
#
# Importar y crear algunas funciones auxiliares
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def preprocess(array):
    """Normaliza la matriz proporcionada y la remodela en el formato adecuado.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def noise(array):
    """Agrega ruido aleatorio a cada imagen en la matriz proporcionada.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """Muestra diez imágenes aleatorias de cada una de las matrices suministradas.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


### **1. Cargar y preprocesar nuestro conjunto de datos MNIST**
#
# #### **Luego creamos una versión ruidosa de nuestros datos de prueba y entrenamiento**



# Dado que solo necesitamos imágenes del conjunto de datos para codificar y decodificar,
# no usará las etiquetas.
(train_data, _), (test_data, _) = mnist.load_data()

# Normalizar y remodelar los datos
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Crear una copia de los datos con ruido añadido
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Muestra los datos del tren y una versión con ruido añadido
display(train_data, noisy_train_data)


### **2. Ahora vamos a crear también nuestro modelo de codificador automático**

# Vamos a usar la API funcional para construir nuestro codificador automático convolucional (esto debería ser familiar
# para los usuarios de PyTorch).


# Nuestra forma de entrada es 28 x 28 x 1
input = layers.Input(shape=(28, 28, 1))

# El modelo del codificador
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# El modelo del decodificador
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Codificador automático: tenga en cuenta que es la concatenación completa del codificador y el decodificador
autoencoder = Model(input, x)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


### **3. Entrena a nuestro codificador automático**
#
# Entrenaremos nuestro codificador automático usando `train_data` como nuestros datos de entrada y objetivo.
#
# NOTA: Estamos configurando los datos de validación usando el mismo formato.


autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data),
)


#### **4. Obtén nuestras predicciones del codificador automático**
#
# Ahora haremos predicciones en nuestro conjunto de datos de prueba y mostraremos la imagen original junto con la
# predicción de nuestro codificador automático.
#
# Observe qué tan cerca están las predicciones de la imagen original. Sin embargo, si miras de cerca, puedes ver
# ligeras diferencias.



predictions = autoencoder.predict(test_data)
display(test_data, predictions)


### **5. Ahora usemos nuestro codificador automático como eliminador de ruido**
#
# Para hacer esto, lo volveremos a entrenar usando los datos ruidosos como nuestra entrada y los datos limpios como
# nuestro objetivo. Esto le enseña a nuestro codificador automático a aprender a eliminar el ruido de las imágenes.

autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)


### **6. Ahora vamos a evaluar su rendimiento en nuestros ruidosos datos de prueba**


predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)



