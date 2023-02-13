#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Regularización en Keras - Parte 2 - Con Regularización**
# ### **Primero entrenamos una CNN en el conjunto de datos Fashion-MNIST sin usar métodos de regularización**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)
# ---
#
#
#
# ---
#
#
# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** usando Keras con TensorFlow 2.0 y lo entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, sin el uso de ningún método de regularización.
#1. Cargando, Inspeccionando y Visualizando nuestros datos
# 2. Preprocesar nuestros datos y definir nuestro **Aumento de datos**
# 3. Construya una CNN simple con regularización
# - Regularización L2
# - Aumento de datos
#   - Abandonar
# - Norma de lote
#4. Capacitar a nuestra CNN con Regularización
#
#

# # **Cargando, Inspeccionando y Visualizando nuestros datos**

# En 1]:


# Cargamos nuestros datos directamente desde los conjuntos de datos incluidos en tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist

# carga el conjunto de datos de entrenamiento y prueba de Fashion-MNIST
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

# Nuestros nombres de clase, al cargar datos de .datasets() nuestras clases son números enteros
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# #### **Comprueba si estamos usando la GPU**

# En 2]:


# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


# ### **Inspeccionar nuestros datos**

# En 3]:


# Mostrar el número de muestras en x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))

# Imprimir el número de muestras en nuestros datos
print ("Number of samples in our training data: " + str(len(x_train)))
print ("Number of labels in our training data: " + str(len(y_train)))
print ("Number of samples in our test data: " + str(len(x_test)))
print ("Number of labels in our test data: " + str(len(y_test)))

# Imprimir las dimensiones de la imagen y no. de etiquetas en nuestros datos de entrenamiento y prueba
print("\n")
print ("Dimensions of x_train:" + str(x_train[0].shape))
print ("Labels in x_train:" + str(y_train.shape))
print("\n")
print ("Dimensions of x_test:" + str(x_test[0].shape))
print ("Labels in y_test:" + str(y_test.shape))


# ### **Visualización de algunos de nuestros datos de muestra**
#
# Tracemos 50 imágenes de muestra.

# En[4]:


# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

# Crear figura y cambiar tamaño
figure = plt.figure()
plt.figure(figsize=(16,10))

# Establecer cuantas imágenes deseamos ver
num_of_images = 50 

# iterar índice de 1 a 51
for index in range(1, num_of_images + 1):
    class_names = classes[y_train[index]]
    plt.subplot(5, 10, index).set_title(f'{class_names}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')


# # **2. Preprocesamiento de datos con ImageDataGenerator**
#
# Primero remodelamos y cambiamos nuestros tipos de datos como lo habíamos hecho anteriormente.

# En[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras import backend as K

# Reformar nuestros datos para que tengan el formato [número de muestras, ancho, alto, color_profundidad]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cambiar el tipo de datos a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# **Recopilamos el tamaño y la forma de nuestra imagen y normalizamos nuestros datos de prueba**
#
# Usaremos ImageDataGenerator para normalizar y proporcionar aumentos de datos para nuestros **datos de entrenamiento**.

# En[6]:


# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# Normalizar nuestros datos entre 0 y 1
x_test /= 255.0


# ### **Una codificación en caliente de nuestras etiquetas**

# En[7]:


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


## **3. Construyendo nuestro modelo**
#
# Esta es la misma CNN que usamos anteriormente para el proyecto de clasificación MNIST.

# En[8]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras import regularizers

L2 = 0.001

# crear modelo
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_regularizer = regularizers.l2(L2),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_regularizer = regularizers.l2(L2)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.SGD(0.001, momentum=0.9),
              metrics = ['accuracy'])

print(model.summary())


# # **Entrenando Nuestro Modelo**

# En[9]:


# Definir generador de datos para aumento
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# Aquí ajustamos el generador de datos a algunos datos de muestra.
#train_datagen.fit(x_train)

batch_size = 32
epochs = 15

# Ajustar el modelo
# Tenga en cuenta que usamos train_datagen.flow, esto toma datos y etiqueta matrices, genera lotes de datos aumentados.
history = model.fit(train_datagen.flow(x_train, y_train, batch_size = batch_size),
                              epochs = epochs,
                              validation_data = (x_test, y_test),
                              verbose = 1,
                              steps_per_epoch = x_train.shape[0] // batch_size)

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# En[ ]:




