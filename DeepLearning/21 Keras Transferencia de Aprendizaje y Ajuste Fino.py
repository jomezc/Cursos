#!/usr/bin/env python
# codificación: utf-8

# # **Keras - Transferencia de aprendizaje con perros y gatos**
#
# ---
#
# En esta lección, aprendemos cómo configurar generadores de datos para cargar nuestro propio conjunto de datos y
# entrenar un clasificador usando Keras.
# 1. Comprender las capas entrenables de una red neuronal
# 2. Configuración de nuestros datos
# 3. Construyendo nuestro modelo para Transfer Learning
# 4. Realice un ajuste fino


# Importación de bibliotecas
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ## **Capas entrenables**
#
# Las capas y los modelos tienen **tres** atributos de peso:
#
# - `pesos` es la lista de todas las variables de peso de la capa.
# - `trainable_weights` es la lista de aquellos que deben actualizarse (a través de gradiente
# descenso) para minimizar la pérdida durante el entrenamiento.
# - `non_trainable_weights` es la lista de aquellos que no están destinados a ser entrenados.
# Por lo general, el modelo los actualiza durante el pase hacia adelante.
#
# **Ejemplo: la capa `Densa` tiene 2 pesos entrenables (núcleo y sesgo)**

# En 2]:


layer = keras.layers.Dense(4)

# Crea los pesos usando layer.build
layer.build((None, 2))  

print(f'Number of weights: {len(layer.weights)}')
print(f'Number of trainable_weights: {len(layer.trainable_weights)}')
print(f'Number of non_trainable_weights: {len(layer.non_trainable_weights)}')


# Todas las capas se pueden entrenar con la excepción de **BatchNormalization**. Utiliza pesos no entrenables para
# realizar un seguimiento de la media y la varianza de sus entradas durante el entrenamiento.

# **Capas y modelos** también cuentan con un atributo booleano `entrenable`.
#
# Su valor se puede cambiar configurando `layer.trainable` a `False` mueve todos los pesos de la capa de entrenable a
# no entrenable.
#
# Esto se llama **"congelar"** la capa: el estado de una capa congelada no
# ser actualizado durante el entrenamiento (ya sea cuando se entrena con `fit()` o cuando se entrena con
# cualquier bucle personalizado que dependa de `trainable_weights` para aplicar actualizaciones de gradiente).
#
# ### **Ejemplo: establecer `trainable` en `False`**

# En 3]:


# Haz un modelo con 2 capas
layer1 = keras.layers.Dense(3, activation="relu")
layer2 = keras.layers.Dense(3, activation="sigmoid")
model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])

# Congelar la primera capa
layer1.trainable = False

# Guarde una copia de los pesos de la capa 1 para referencia posterior
initial_layer1_weights_values = layer1.get_weights()

# Entrenar al modelo
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))


# Verifique que los pesos de la capa 1 no hayan cambiado durante el entrenamiento
final_layer1_weights_values = layer1.get_weights()

if initial_layer1_weights_values[0].all() == final_layer1_weights_values[0].all():
  print('Weights unchanged')

if initial_layer1_weights_values[1].all() == final_layer1_weights_values[1].all():
  print('Weights unchanged')


# **Nota**: **`.trianable` es recursivo**, lo que significa que en un modelo o en cualquier capa que tenga subcapas,
# todas las capas secundarias tampoco se pueden entrenar.

# ## **Implementación del aprendizaje por transferencia**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/blob/main/Screenshot%202021-05-11%20at%2011.49.01%20pm.png?raw=true)
#
# ## **Flujo de trabajo de transferencia de aprendizaje**
#
# 1. Instanciamos un **modelo base y cargamos pesos previamente entrenados** en él.
# 2. **Congele** todas las capas en el modelo base configurando `trainable = False`.
# 3. Cree un **nuevo modelo encima** de la salida de una (o varias) capas desde la base
#  modelo.
# 4. Entrene su nuevo modelo en su nuevo conjunto de datos.

# ### **Paso 1. Cargue un modelo base con pesos previamente entrenados (ImageNet)**

# En[5]:


import tensorflow_datasets as tfds

tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserva 10% para validación y 10% para test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Incluir etiquetas
)

print(f'Number of training samples: {tf.data.experimental.cardinality(train_ds)}')
print(f'Number of validation samples: {tf.data.experimental.cardinality(validation_ds)}')
print(f'Number of test samples: {tf.data.experimental.cardinality(test_ds)}')


# Estas son las primeras 9 imágenes en el conjunto de datos de entrenamiento; como puede ver, todas son
#  diferentes tamaños.

# En[6]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title('Cat' if int(label) == 0 else 'Dog')
    plt.axis("off")
plt.show()

# ## **Estandarizar nuestros datos**
#
# - Estandarizar a un tamaño de imagen fijo. Elegimos 150x150.
# - Normalice los valores de píxel entre -1 y 1. Haremos esto usando una capa de `Normalización` como
# parte del propio modelo.

size = (150, 150)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))


# Procesaremos los datos por lotes y utilizaremos el almacenamiento en caché y la captación previa para optimizar la
# velocidad de carga.

batch_size = 32

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


# ### **Introducir algunos aumentos de datos aleatorios**


from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)


# #### **Visualizar nuestros aumentos de datos**


import numpy as np

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

### **3. Construyendo nuestro modelo**
#
# Ahora construyamos un modelo que siga el modelo que hemos explicado anteriormente.
#
# Tenga en cuenta que:
#
# - Agregamos una capa de `Normalización` para escalar los valores de entrada (inicialmente en `[0, 255]`
# rango) al rango `[-1, 1]`.
# - Agregamos una capa `Dropout` antes de la capa de clasificación, para la regularización.
# - Nos aseguramos de pasar `training=False` al llamar al modelo base, para que
# se ejecuta en modo de inferencia, por lo que las estadísticas de normas por lotes no se actualizan
# incluso después de descongelar el modelo base para realizar ajustes.
#
# - Usaremos el **Modelo Xception** como base.


base_model = keras.applications.Xception(
    weights="imagenet",  # Cargue pesos previamente entrenados en ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # No incluya el clasificador ImageNet en la parte superior.

# Congelar el modelo_base
base_model.trainable = False

# Crear nuevo modelo en la parte superior
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Aplicar aumento de datos aleatorios

# Los pesos de Xception preentrenados requieren que la entrada se escale
# de (0, 255) a un rango de (-1., +1.), la capa de reescalado
# salidas: `(entradas * escala) + compensación`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# El modelo base contiene capas de normas por lotes. Queremos mantenerlos en modo de inferencia.
# cuando descongelamos el modelo base para ajustarlo, así nos aseguramos de que el
# base_model se está ejecutando en modo de inferencia aquí.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularizar con deserción
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()


# ## **Ahora entrenemos nuestra capa superior**
#
# Tenga en cuenta del resumen anterior que solo tenemos 2,049 parámetros entrenables.


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 5
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


### **4. Sintonia FINA**
#
# **Descongelamos** el modelo base y entrenamos todo el modelo de principio a fin con una tasa de aprendizaje **baja**.
#
# **Notas** aunque el modelo base se vuelve entrenable, aún se ejecuta en modo de inferencia ya que pasamos
# `training=False` cuando lo llamamos cuando construimos el modelo.
#
# Esto significa que las capas de normalización de lotes internas no actualizarán sus estadísticas de lotes. Si lo
# hicieran, causarían estragos en las representaciones aprendidas por el modelo hasta el momento.


# Descongele el modelo_base. Tenga en cuenta que sigue ejecutándose en modo de inferencia
# desde que pasamos `training=False` al llamarlo. Esto significa que
# las capas batchnorm no actualizarán sus estadísticas de lotes.
# Esto evita que las capas de normas por lotes deshagan todo el entrenamiento
# que hemos hecho hasta ahora.
base_model.trainable = True
model.summary()

# CON TASA BAJA TARDA MUCHO
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Baja tasa de aprendizaje
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 1
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
