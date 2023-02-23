#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Transferencia de estilo neuronal en Keras Tensorflow 2.0**
#
# ---
#
#
# En esta lección, primero aprendemos a implementar el **Algoritmo de transferencia de estilo neuronal** usando Keras con Tensorflow 2.0. También comenzamos a aprender a cargar y usar modelos desde el TF-Hub.
#
# Aplicamos la técnica conocida como *transferencia de estilo neuronal* que se muestra en la investigación publicada aquí <a href="https://arxiv.org/abs/1508.06576" class="external">Un algoritmo neuronal de estilo artístico</a > (Gatys et al.).
#
# En este tutorial demostramos el algoritmo de transferencia de estilo original. Optimiza el contenido de la imagen a un estilo particular. Los enfoques modernos entrenan un modelo para generar la imagen estilizada directamente (similar a [cyclegan](cyclegan.ipynb)). Este enfoque es mucho más rápido (hasta 1000x).
#
# 1. Configuración, módulos de carga y función auxiliar
# 2. Transferencia de estilo rápido usando TF-Hub
# 3. Implementando nuestro modelo desde cero
# 4. Construye el modelo
# 5. Extrae estilo y contenido
# 6. Descenso de pendiente en carrera
# 7. Pérdida de variación total - Reducción de artefactos de alta frecuencia
# 8. Vuelva a ejecutar la optimización
#
# Fuente: https://www.tensorflow.org/tutorials/generative/style_transfer
#
#

### **1. Configurar, cargar módulos y funciones auxiliares**
#
#
#

# En[36]:


import os
import tensorflow as tf

# Cargar modelos comprimidos desde tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# En[37]:


# Establecer nuestros parámetros de trazado de imagen e importar algunos módulos
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


# En[38]:


# función que transforma un tensor en imagen
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


# En[39]:


# cargar nuestro contenido y diseñar imágenes
content_path = tf.keras.utils.get_file('labrador.jpeg', 'https://github.com/rajeevratan84/ModernComputerVision/raw/main/labrador.jpeg')
style_path = tf.keras.utils.get_file('the_wave.jpg','https://github.com/rajeevratan84/ModernComputerVision/raw/main/the_wave.jpg')


# mosaico - https://github.com/rajeevratan84/ModernComputerVision/raw/main/mosaic.jpg
# plumas - https://github.com/rajeevratan84/ModernComputerVision/raw/main/feathers.jpg


# Defina una función para cargar una imagen y limite su dimensión máxima a 512 píxeles.

# En[40]:


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


# Crea una función simple para mostrar una imagen:

# En[41]:


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# En[42]:


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


### **2. Transferencia rápida de estilos usando TF-Hub**
#
# Antes de implementar el algoritmo por nuestra cuenta, intentemos usar un fundador de modelo preentrenado simple en TensorFlow Hub.
#
# [Modelo de TensorFlow Hub](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

# En[43]:


# importar el centro de tensorflow que nos permite descargar directamente modelos preentrenados
import tensorflow_hub as hub

# Obtener nuestro modelo
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# imágenes de entrada o estilo y contenido
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# convertir el tensor devuelto en una imagen
tensor_to_image(stylized_image)


### **3. Implementando nuestro modelo desde cero**
#
# ### **Definir representaciones de contenido y estilo**
#
# Usamos las capas intermedias del modelo para obtener las representaciones de *contenido* y *estilo* de la imagen.
#
# A partir de la capa de entrada de la red, las primeras activaciones de capa representan características de bajo nivel como bordes y texturas. A medida que avanza por la red, las últimas capas representan características de nivel superior: partes de objetos como *ruedas* u *ojos*.
#
# Aquí usaremos la arquitectura de red VGG19, una red de clasificación de imágenes previamente entrenada. Estas capas intermedias son necesarias para definir la representación de contenido y estilo de las imágenes. Para una imagen de entrada, intente hacer coincidir las representaciones de destino de contenido y estilo correspondientes en estas capas intermedias.
#

# En[44]:


# Cargar VGG19 sin la cabeza
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# imprimir una lista de nombres de capas
for layer in vgg.layers:
  print(layer.name)


# Elija capas intermedias de la red para representar el estilo y el contenido de la imagen:

# En[45]:


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# #### **¿Por qué elegir capas intermedias para representaciones de estilo y contenido?**
#
# Entonces, ¿por qué estas salidas intermedias dentro de nuestra red de clasificación de imágenes preentrenada nos permiten definir representaciones de estilo y contenido?
#
# En un nivel alto, para que una red realice la clasificación de imágenes (para lo cual esta red ha sido entrenada), debe comprender la imagen. Esto requiere tomar la imagen sin procesar como píxeles de entrada y crear una representación interna que convierta los píxeles de la imagen sin procesar en una comprensión compleja de las características presentes en la imagen.
#
# Esta es también una de las razones por las que las redes neuronales convolucionales pueden generalizar bien: **son capaces de capturar las invariancias y definir características dentro de las clases** (p. ej., gatos frente a perros) que son independientes del ruido de fondo y otras molestias. Por lo tanto, en algún lugar entre donde la imagen sin procesar se introduce en el modelo y la etiqueta de clasificación de salida, el modelo sirve como extractor de características complejas. Al acceder a las capas intermedias del modelo, puede describir el contenido y el estilo de las imágenes de entrada.

### **4. Construye el modelo**
#
# Las redes en `tf.keras.applications` están diseñadas para que pueda extraer fácilmente los valores de la capa intermedia utilizando la API funcional de Keras.
#
# Para definir un modelo utilizando la API funcional, especifique las entradas y salidas:
#
# `modelo = Modelo(entradas, salidas)`
#
# La siguiente función crea un modelo VGG19 que devuelve una lista de salidas de capas intermedias:

# En[46]:


def vgg_layers(layer_names):
  """ Crea un modelo vgg que devuelve una lista de valores de salida intermedios."""
  # Carga nuestro modelo. Cargue VGG preentrenado, entrenado en datos de imagenet
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  #
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


# Ahora usamos la función anterior para obtener nuestro extractor de estilo y salidas de estilo

# En[47]:


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Mira las estadísticas de salida de cada capa
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()


# ### **Calcular estilo**
#
# El contenido de una imagen está representado por los valores de los mapas de características intermedias.
#
# Resulta que **el estilo de una imagen se puede describir mediante las medias y las correlaciones entre los diferentes mapas de funciones.**
#
# Podemos usar esto para calcular una **matriz de Gram** que incluye esta información tomando el producto externo del vector de características consigo mismo en cada ubicación y promediando ese producto externo en todas las ubicaciones. Esta matriz de Gram se puede calcular para una capa en particular como:
#
# $$G^l_{cd} = \frac{\sum_{ij} F^l_{ijc}(x)F^l_{ijd}(x)}{IJ}$$
#
# Esto se puede implementar de manera concisa usando la función `tf.linalg.einsum`:

# En[55]:


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


### **5. Extraer estilo y contenido**
# Cree un modelo que devuelva los tensores de estilo y contenido.

# En[49]:


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


# Cuando se llama a una imagen, este modelo devuelve la matriz de gramo (estilo) de `style_layers` y el contenido de `content_layers`:

# En[56]:


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())


### **6. Descenso de pendiente en carrera**
#
# ¡Con este extractor de estilo y contenido, ahora puede implementar el algoritmo de transferencia de estilo!
#
# Haga esto calculando el error cuadrático medio para la salida de su imagen en relación con cada objetivo, luego tome la suma ponderada de estas pérdidas.

# En[16]:


# Establezca sus valores objetivo de estilo y contenido
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# Defina una `tf.Variable` para que contenga la imagen a optimizar. Para hacerlo rápido, inicialícelo con la imagen del contenido (la `tf.Variable` debe tener la misma forma que la imagen del contenido):

# En[17]:


image = tf.Variable(content_image)


# Dado que esta es una imagen flotante, defina una función para mantener los valores de píxel entre 0 y 1:

# En[18]:


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Crear un optimizador. El documento recomienda LBFGS, pero `Adam` también funciona bien:

# En 19]:


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# Para optimizar esto, use una combinación ponderada de las dos pérdidas para obtener la pérdida total:

# En 20]:


style_weight=1e-2
content_weight=1e4


# En[57]:


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# Use `tf.GradientTape` para actualizar la imagen.
#

# En[58]:


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# Ahora ejecute algunos pasos para probar:

# En[59]:


train_step(image)
train_step(image)
train_step(image)
tensor_to_image(image)


# **¡Funciona!**
#
# Ya que está funcionando, realice una optimización más larga:

# En[60]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))


### **7. Pérdida de variación total - Reducción de artefactos de alta frecuencia**
#
# Una desventaja de esta implementación básica es que produce muchos artefactos de alta frecuencia. Disminuya estos utilizando un término de regularización explícito en los componentes de alta frecuencia de la imagen. En transferencia de estilo, esto a menudo se denomina *pérdida de variación total*:

# En[25]:


def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var


# ## Ver visualmente los componentes de alta frecuencia

# En[26]:


x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")


# Esto muestra como se han incrementado los componentes de alta frecuencia.
#
# Además, este componente de alta frecuencia es básicamente un detector de bordes. Puede obtener un resultado similar del detector de bordes Sobel, por ejemplo:

# En[27]:


plt.figure(figsize=(14, 10))

sobel = tf.image.sobel_edges(content_image)
plt.subplot(1, 2, 1)
imshow(clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
plt.subplot(1, 2, 2)
imshow(clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")


# La pérdida de regularización asociada a esto es la suma de los cuadrados de los valores:

# En[61]:


def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


# Eso demostró lo que hace. Pero no es necesario que lo implementes tú mismo, TensorFlow incluye una implementación estándar:

# En[63]:


tf.image.total_variation(image).numpy()


### **8. Vuelva a ejecutar la optimización**
#
# Elija un peso para `total_variation_loss`, elegiremos 30

# En[65]:


total_variation_weight=30


# Ahora inclúyelo en la función `train_step`:

# En[66]:


@tf.function()
def train_step(image, total_variation_weight):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# En[67]:


image = tf.Variable(content_image)


# Y ejecuta la optimización:

# En[68]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100
total_variation_weight=30

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image, total_variation_weight)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))


# ### **¡¡Mucho mejor!!**
# #### **Guarda el resultado y presume a tus amigos**

# En[70]:


file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download(file_name)


# En[ ]:




