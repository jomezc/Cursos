#!/usr/bin/env python
# codificación: utf-8

# # **Replicando LeNet y AlexNet en Tensorflow2.0 usando Keras**
#
# ---
#
# En esta lección, usamos **Keras con TensorFlow 2.0** Backend para replicar **LeNet y AlexNet** en Keras y entrenarlo
# para **reconocer dígitos escritos a mano en el conjunto de datos MNIST y las 10 clases de imágenes de CIFAR10 **
# 1. Replicar la arquitectura LeNet CNN
# 2. Replicar la arquitectura CNN de AlexNet

# ## **¡Construyamos LeNet en Keras!**
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
# ### **Cargando y preprocesando nuestros Datos**

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta

# carga el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

# crear modelo
model = Sequential()

# 2 juegos de CRP (Convolución, RELU, Pooling)
model.add(Conv2D(6, (5, 5),  # 6 filtros con kernel 5X5
                 padding = "same",  # same salida = entrada, para establecer el relleno en 0, use "valid"
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
           
model.compile(loss = 'categorical_crossentropy',  # apropiado para una clasificación multiclase como esta
              optimizer = Adadelta(),  # suele usar
              metrics = ['accuracy'])
    
print(model.summary())
'''Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 6)         156       
                                                                 
 activation (Activation)     (None, 28, 28, 6)         0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 6)        0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 16)        2416      
                                                                 
 activation_1 (Activation)   (None, 14, 14, 16)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 16)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 7, 7, 120)         48120     
                                                                 
 activation_2 (Activation)   (None, 7, 7, 120)         0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 3, 3, 120)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1080)              0         
                                                                 
 dense (Dense)               (None, 120)               129720    
                                                                 
 activation_3 (Activation)   (None, 120)               0         
                                                                 
 dense_1 (Dense)             (None, 84)                10164     
                                                                 
 activation_4 (Activation)   (None, 84)                0         
                                                                 
 dense_2 (Dense)             (None, 10)                850       
                                                                 
 activation_5 (Activation)   (None, 10)                0         
                                                                 
=================================================================
Total params: 191,426
Trainable params: 191,426
Non-trainable params: 0
_________________________________________________________________'''

# ### **Ahora entrenemos a LeNet en nuestro conjunto de datos MNIST**


# Parámetros de entrenamiento
batch_size = 128  # tamaño del batch
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
'''
...
469/469 [==============================] - 1s 2ms/step - loss: 0.3535 - accuracy: 0.9005 - val_loss: 0.3297 - val_accuracy: 0.9053
Epoch 48/50
469/469 [==============================] - 1s 3ms/step - loss: 0.3478 - accuracy: 0.9017 - val_loss: 0.3245 - val_accuracy: 0.9066
Epoch 49/50
469/469 [==============================] - 1s 3ms/step - loss: 0.3424 - accuracy: 0.9025 - val_loss: 0.3193 - val_accuracy: 0.9074
Epoch 50/50
469/469 [==============================] - 1s 2ms/step - loss: 0.3371 - accuracy: 0.9042 - val_loss: 0.3142 - val_accuracy: 0.9094
313/313 [==============================] - 0s 1ms/step - loss: 0.3142 - accuracy: 0.9094
Test loss: 0.31422871351242065
Test accuracy: 0.9093999862670898
'''

# ## **Ahora repliquemos AlexNET y entrenémonos en el conjunto de datos CIFAR10**
# AlexNet fue el ganador de ImageNet en 2012 y logró un error entre los 5 primeros del 15,3 %, ¡más de 10,8 puntos
# porcentuales menos que el del subcampeón!
#
# ![](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-22_at_6.35.45_PM.png)
#
# ![](https://production-media.paperswithcode.com/datasets/CIFAR-10-0000000431-b71f61c0_U5n3Glr.jpg)


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
'''x_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
Model: "sequential_1"
_________________________'''
# Ahora codificamos las salidas en caliente
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

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
'''________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_3 (Conv2D)           (None, 32, 32, 96)        34944     
                                                                 
 batch_normalization (BatchN  (None, 32, 32, 96)       384       
 ormalization)                                                   
                                                                 
 activation_6 (Activation)   (None, 32, 32, 96)        0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 16, 16, 96)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 16, 16, 256)       614656    
                                                                 
 batch_normalization_1 (Batc  (None, 16, 16, 256)      1024      
 hNormalization)                                                 
                                                                 
 activation_7 (Activation)   (None, 16, 16, 256)       0         
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 8, 8, 256)        0         
 2D)                                                             
                                                                 
 zero_padding2d (ZeroPadding  (None, 10, 10, 256)      0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 10, 10, 512)       1180160   
                                                                 
 batch_normalization_2 (Batc  (None, 10, 10, 512)      2048      
 hNormalization)                                                 
                                                                 
 activation_8 (Activation)   (None, 10, 10, 512)       0         
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 5, 5, 512)        0         
 2D)                                                             
                                                                 
 zero_padding2d_1 (ZeroPaddi  (None, 7, 7, 512)        0         
 ng2D)                                                           
                                                                 
 conv2d_6 (Conv2D)           (None, 7, 7, 1024)        4719616   
                                                                 
 batch_normalization_3 (Batc  (None, 7, 7, 1024)       4096      
 hNormalization)                                                 
                                                                 
 activation_9 (Activation)   (None, 7, 7, 1024)        0         
                                                                 
 zero_padding2d_2 (ZeroPaddi  (None, 9, 9, 1024)       0         
 ng2D)                                                           
                                                                 
 conv2d_7 (Conv2D)           (None, 9, 9, 1024)        9438208   
                                                                 
 batch_normalization_4 (Batc  (None, 9, 9, 1024)       4096      
 hNormalization)                                                 
                                                                 
 activation_10 (Activation)  (None, 9, 9, 1024)        0         
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 4, 4, 1024)       0         
 2D)                                                             
                                                                 
 flatten_1 (Flatten)         (None, 16384)             0         
                                                                 
 dense_3 (Dense)             (None, 3072)              50334720  
                                                                 
 batch_normalization_5 (Batc  (None, 3072)             12288     
 hNormalization)                                                 
                                                                 
 activation_11 (Activation)  (None, 3072)              0         
                                                                 
 dropout (Dropout)           (None, 3072)              0         
                                                                 
 dense_4 (Dense)             (None, 4096)              12587008  
                                                                 
 batch_normalization_6 (Batc  (None, 4096)             16384     
 hNormalization)                                                 
                                                                 
 activation_12 (Activation)  (None, 4096)              0         
                                                                 
 dropout_1 (Dropout)         (None, 4096)              0         
                                                                 
 dense_5 (Dense)             (None, 10)                40970     
                                                                 
 batch_normalization_7 (Batc  (None, 10)               40        
 hNormalization)                                                 
                                                                 
 activation_13 (Activation)  (None, 10)                0         
                                                                 
=================================================================
Total params: 78,990,642
Trainable params: 78,970,462
Non-trainable params: 20,180
_________________________________________________________________'''
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adadelta(),  #él  usa adam
              metrics = ['accuracy'])


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
'''782/782 [==============================] - 46s 59ms/step - loss: 1.2742 - accuracy: 0.6143 - val_loss: 1.2881 - val_accuracy: 0.6006
Epoch 24/25
782/782 [==============================] - 46s 59ms/step - loss: 1.2660 - accuracy: 0.6146 - val_loss: 1.2812 - val_accuracy: 0.6028
Epoch 25/25
782/782 [==============================] - 46s 59ms/step - loss: 1.2526 - accuracy: 0.6241 - val_loss: 1.2725 - val_accuracy: 0.6066
313/313 [==============================] - 4s 11ms/step - loss: 1.2725 - accuracy: 0.6065
Test loss: 1.2724741697311401
Test accuracy: 0.6065000295639038'''

# ## **Mejores resultados actuales en CIFAR10**
# https://paperswithcode.com/sota/image-classification-on-cifar-10




