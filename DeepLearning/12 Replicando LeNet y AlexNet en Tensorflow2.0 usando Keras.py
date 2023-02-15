#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Replicando LeNet y AlexNet en Tensorflow2.0 usando Keras**
#
# ---
#
# En esta lección, usamos **Keras con TensorFlow 2.0** Backend para replicar **LeNet y AlexNet** en Keras y entrenarlo para **reconocer dígitos escritos a mano en el conjunto de datos MNIST y las 10 clases de imágenes de CIFAR10 **
# 1. Replicar la arquitectura LeNet CNN
# 2. Replicar la arquitectura CNN de AlexNet

# ## **¡Construyamos LeNet en Keras!**
#
# ![](https://www.researchgate.net/profile/Sheraz_Khan8/publication/321586653/figure/fig4/AS:568546847014912@1512563539828/The-LeNet-5-Architecture-a-convolutional-neural-network.png)
# ## **Arquitectura LeNet**
# S.No | Capas | Forma de salida (alto, ancho, canales)
# --- | --- | ---
# 1 | Capa de entrada | 32x32x1
# 2 | Conv2d [6 filtros de tamaño = 5x5, zancada = 1, relleno = 0] | 28x28x6
# 3 | Agrupación promedio [zancada = 2, relleno = 0] | 14x14x6
# 4 | Conv2d [16 filtros de tamaño = 5x5, zancada = 1, relleno = 0] | 10x10x16
# 5 | Agrupación promedio [zancada = 2, relleno = 0] | 5x5x16
# 6 | Conv2d [120 filtros de tamaño = 5x5, zancada = 1, relleno = 0] | 1x1x120
# 7 | Capa Lineal1 | 120
# 8 | Capa Lineal2 | 84
# 9 | Capa lineal final | 10
#
#
# ### **Cargando y preprocesando nuestros Datos**

# En[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta

# carga el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Obtener nuestra fecha en la 'forma' correcta necesaria para Keras
# Necesitamos agregar una cuarta dimensión a nuestra fecha, cambiando así nuestra
# Nuestra forma de imagen original de (60000,28,28) a (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# cambiar nuestro tipo de imagen a tipo de datos float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizar nuestros datos cambiando el rango de (0 a 255) a (0 a 1)
x_train /= 255
x_test /= 255

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# ### **Ahora vamos a crear nuestras capas para replicar LeNet**

# En[5]:


# crear modelo
model = Sequential()

# 2 juegos de CRP (Convolución, RELU, Pooling)
model.add(Conv2D(6, (5, 5),
                 padding = "same", # para establecer el relleno en 0, use "válido"
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Conv2D(16, (5, 5),
                 padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Conv2D(120, (5, 5),
                 padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Capas completamente conectadas (con RELU)
model.add(Flatten())
model.add(Dense(120))
model.add(Activation("relu"))

model.add(Dense(84))
model.add(Activation("relu"))
# Softmax (para clasificación)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adadelta(),
              metrics = ['accuracy'])
    
print(model.summary())


# ### **Ahora entrenemos a LeNet en nuestro conjunto de datos MNIST**

# En[7]:


# Parámetros de entrenamiento
batch_size = 128
epochs = 50

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("mnist_LeNet.h5")

# Evaluar el desempeño de nuestro modelo entrenado
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# ## **Ahora repliquemos AlexNET y entrenémonos en el conjunto de datos CIFAR10**
#
# AlexNet fue el ganador de ImageNet en 2012 y logró un error entre los 5 primeros del 15,3 %, ¡más de 10,8 puntos porcentuales menos que el del subcampeón!
#
# ![](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-22_at_6.35.45_PM.png)
#
# ![](https://production-media.paperswithcode.com/datasets/CIFAR-10-0000000431-b71f61c0_U5n3Glr.jpg)

# En[8]:


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical

# Carga el conjunto de datos CIFAR
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Mostrar nuestra forma/dimensiones de datos
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Ahora codificamos las salidas en caliente
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# En[9]:


l2_reg = 0.001

# Inicializar modelo
model = Sequential()

# 1ra capa de conversión
model.add(Conv2D(96, (11, 11), input_shape=x_train.shape[1:],
    padding='same', kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2da capa de conversión
model.add(Conv2D(256, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3ra capa de conversión
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4ta capa de conversión
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5ta capa de conversión
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 1ra Capa FC
model.add(Flatten())
model.add(Dense(3072))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 2da Capa FC
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Mostrar decodificador
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adadelta(),
              metrics = ['accuracy'])


# En[10]:


# Parámetros de entrenamiento
batch_size = 64
epochs = 25

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("CIFAR10_AlexNet_10_Epoch.h5")

# Evaluar el desempeño de nuestro modelo entrenado
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# ## **Mejores resultados actuales en CIFAR10**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-12-04%20at%207.56.25%20pm.png)

# En[ ]:




