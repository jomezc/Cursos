#!/usr/bin/env python
# codificación: utf-8

#
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Redes adversarias generativas (GAN) en Keras: GAN convolucional profunda o DCGAN con MNIST**
#
# ---
#
#
# En esta lección, primero aprendemos a implementar **Redes adversas generativas (GAN) o DCGAN** con Keras utilizando el conjunto de datos MNIST.
#
# En este tutorial demostramos el algoritmo de transferencia de estilo original. Optimiza el contenido de la imagen a un estilo particular. Los enfoques modernos entrenan un modelo para generar la imagen estilizada directamente (similar a [cyclegan](cyclegan.ipynb)). Este enfoque es mucho más rápido (hasta 1000x).
#
# 1. Configurar, cargar y preparar el conjunto de datos
# 2. Cargue y prepare el conjunto de datos
# 3. Definir nuestro Modelo de Generador
# 4. Definir nuestro Modelo Discriminador
# 5. Definir la pérdida y los optimizadores
# 6. Define el ciclo de entrenamiento
# 7. Entrenando al modelo
#
#

# ## **Resumen de las GAN**
# [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GAN) son un dominio realmente genial de Deep Learning donde producimos nuevos datos que probablemente provienen del conjunto de datos en el que entrenamos.
#
# Para hacer esto, usamos una red *generadora* que aprende a crear imágenes que parecen reales, mientras que una red *discriminadora* aprende a diferenciar las imágenes reales de las falsas.
#
# A medida que avanza el entrenamiento, el *generador* se vuelve mejor para crear imágenes que parecen reales, mientras que, al mismo tiempo, el *discriminador* se vuelve mejor para diferenciarlas.
#
# El proceso alcanza el equilibrio cuando el *discriminador* ya no puede distinguir las imágenes reales de las falsificaciones. En este tutorial demostramos este proceso en el conjunto de datos MNIST.
#
#
# ![Un segundo diagrama de un generador y discriminador](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202021-06-03%20at%209.54.37%402x.png)
#
#

### **1. Configurar, cargar y preparar el conjunto de datos**

# En 1]:


# Para generar GIF
get_ipython().system('pip install imageio')
get_ipython().system('pip install git+https://github.com/tensorflow/docs')


# En 2]:


import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


### **2. Cargue y prepare el conjunto de datos**
#
# Entrenaremos nuestra GAN usando el conjunto de datos MNIST, al final queremos que nuestro generador pueda generar dígitos escritos a mano que se parezcan a los del conjunto de datos MNIST.

# En 3]:


# Descargar MNIST
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Reformar y Normalizar
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normaliza las imágenes a [-1, 1]

# Establecer lote y tamaño de búfer
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Lote y baraje los datos
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


### **3. Definir nuestro Modelo de Generador**
#
# Tanto el generador como el discriminador se definen mediante la API secuencial de Keras.
#
# El generador usa capas `tf.keras.layers.Conv2DTranspose` (sobremuestreo) para producir una imagen a partir de una semilla (ruido aleatorio). Comience con una capa "Densa" que tome esta semilla como entrada, luego aumente la muestra varias veces hasta que alcance el tamaño de imagen deseado de 28x28x1. Observe la activación `tf.keras.layers.LeakyReLU` para cada capa, excepto la capa de salida que usa tanh.
#

# En[5]:


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Nota: Ninguno es el tamaño del lote

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Ahora veamos qué tan bueno es nuestro generador no entrenado para hacer una imagen en escala de grises de 28x28 (como MNIST).

# En[6]:


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# No muy bueno....

### **4. Definir nuestro Modelo Discriminador**
#
# El discriminador es simplemente un clasificador de imágenes basado en CNN.

# En[7]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Tenga en cuenta que la salida es un solo nodo binario

    return model


# Ahora usemos el discriminador no entrenado para clasificar las imágenes generadas como reales o falsas.
#
# El modelo será entrenado para generar valores positivos para imágenes reales y valores negativos para imágenes falsas.

# En[9]:


discriminator = make_discriminator_model()
# Use la imagen generada producida por nuestro Generador no entrenado
decision = discriminator(generated_image)
print(decision)


### **5. Definir la pérdida y los optimizadores**
#
# Definir funciones de pérdida y optimizadores para ambos modelos.
#

# En[10]:


# Este método devuelve una función auxiliar para calcular la pérdida de entropía cruzada
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ### **Pérdida de discriminador**
#
# Este método cuantifica qué tan bien el discriminador es capaz de distinguir las imágenes reales de las falsas. Compara las predicciones del discriminador sobre imágenes reales con una matriz de 1 y las predicciones del discriminador sobre imágenes falsas (generadas) con una matriz de 0.

# En[11]:


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# ### **Pérdida del generador**
# La pérdida del generador cuantifica qué tan bien pudo engañar al discriminador. Intuitivamente, si el generador funciona bien, el discriminador clasificará las imágenes falsas como reales (o 1). Aquí, compare las decisiones de los discriminadores en las imágenes generadas con una matriz de 1.

# En[12]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Los optimizadores discriminador y generador son diferentes ya que entrenará dos redes por separado.

# En[13]:


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# ### **Guardar puntos de control**
# Este cuaderno también demuestra cómo guardar y restaurar modelos, lo que puede ser útil en caso de que se interrumpa una tarea de entrenamiento de larga duración.

# En[14]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


### **6. Definir el ciclo de entrenamiento**
#

# En[15]:


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Reutilizarás esta semilla con el tiempo (para que sea más fácil)
# para visualizar el progreso en el GIF animado)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# El ciclo de entrenamiento comienza cuando el generador recibe una **semilla aleatoria** como entrada.
#
# Esa semilla se usa para producir una imagen.
#
# El discriminador se usa luego para clasificar imágenes reales (extraídas del conjunto de entrenamiento) e imágenes falsas (producidas por el generador).
#
# La pérdida se calcula para cada uno de estos modelos y los gradientes se utilizan para actualizar el generador y el discriminador.

# En[16]:


# Note el uso de `tf.function`
# Esta anotación hace que la función sea "compilada".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# ### **Generar y guardar imágenes**
#

# En[18]:


def generate_and_save_images(model, epoch, test_input):
  # Observe que `entrenamiento` está establecido en False.
  # Esto es para que todas las capas se ejecuten en modo de inferencia (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# En 19]:


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce imágenes para el GIF sobre la marcha
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Guarda el modelo cada 15 épocas
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generar después de la época final
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


### **7. Entrenando al modelo**
# Llame al método `train()` definido anteriormente para entrenar el generador y el discriminador simultáneamente. Tenga en cuenta que entrenar GAN puede ser complicado. Es importante que el generador y el discriminador no se dominen entre sí (por ejemplo, que entrenen a un ritmo similar).
#
# Al comienzo del entrenamiento, las imágenes generadas parecen ruido aleatorio. A medida que avanza el entrenamiento, los dígitos generados se verán cada vez más reales. Después de unas 50 épocas, se asemejan a los dígitos MNIST. Esto puede demorar aproximadamente un minuto por época con la configuración predeterminada en Colab.

# En 20]:


train(train_dataset, EPOCHS)


# Restaurar el último punto de control.

# En[21]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


### **7. Salida - Vamos a crear un GIF**
#

# En[22]:


# Mostrar una sola imagen usando el número de época
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# En[23]:


display_image(EPOCHS)


# Use `imageio` para crear un gif animado usando las imágenes guardadas durante el entrenamiento.

# En[24]:


anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


# En[25]:


import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)


# ## Próximos pasos
#

# Este tutorial ha mostrado el código completo necesario para escribir y entrenar una GAN. Como siguiente paso, es posible que desee experimentar con un conjunto de datos diferente, por ejemplo, el conjunto de datos de atributos de rostros de celebridades a gran escala (CelebA) [disponible en Kaggle] (https://www.kaggle.com/jessicali9530/celeba-dataset) . Para obtener más información sobre las GAN, consulte el [Tutorial de NIPS 2016: Redes antagónicas generativas] (https://arxiv.org/abs/1701.00160).
#
