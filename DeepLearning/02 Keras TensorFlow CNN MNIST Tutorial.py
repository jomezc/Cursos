#!/usr/bin/env python
# codificación: utf-8

# # **Introducción a Keras usando un backend de TensorFlow 2.0**
# ### **Entrenamiento de una CNN simple en el conjunto de datos MNIST - Dígitos escritos a mano**

# En esta lección, usamos **Keras con TensorFlow 2.0** Backend para crear un **modelo de red neuronal convolucional
# simple** en PyTorch y entrenarlo para **reconocer dígitos escritos a mano en el conjunto de datos MNIST.**
# 1. Cargando nuestro conjunto de datos MNIST
# 2. Inspeccionar nuestro conjunto de datos
# 3. Visualización de nuestro conjunto de datos de imágenes
# 5. Preprocesamiento de nuestro conjunto de datos
# 6. Construyendo nuestro Modelo
# 7. Entrenando a nuestro Modelo
# 8. Graficando nuestros registros de entrenamiento
# 9. Guardando y Cargando nuestro Modelo
# 10. Probando nuestro modelo con datos de prueba

### **1. Cargando nuestros datos**
#
# Hay conjuntos de datos incorporados de ```tensorflow.keras.datasets``` para cargar nuestros datos. Usamos la función
# ```mnist.load_data()```.
#
# Devuelve: **2 tuplas**
# - x_train, x_test: matriz uint8 de datos de imagen RGB con forma (num_samples, 3, 32, 32) o (num_samples, 32, 32, 3)
# según la configuración del backend image_data_format de channel_first o channel_last respectivamente.
# - y_train, y_test: matriz uint8 de etiquetas de categoría (enteros en el rango 0-9) con forma (num_samples, 1).

# - Más información sobre los datos disponibles en https://keras.io/datasets/

# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# #### **Una revisión rápida para ver si estamos usando la GPU**

# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
'''
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 12361390168452218670
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 14190510080
locality {
  bus_id: 1
  links {
  }
}
incarnation: 12939192816315270802
physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3080 Ti Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6"
xla_global_id: 416903419
]'''

# ## **2. Inspeccionando nuestro conjunto de datos**

# Mostrar el número de muestras en x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))
# Initial shape or dimensions of x_train (60000, 28, 28)


# Imprimir el número de muestras en nuestros datos
print("Number of samples in our training data: " + str(len(x_train)))  # Number of samples in our training data: 60000
print("Number of labels in our training data: " + str(len(y_train)))  # Number of labels in our training data: 60000
print("Number of samples in our test data: " + str(len(x_test)))  # Number of samples in our test data: 10000
print("Number of labels in our test data: " + str(len(y_test)))  # Number of labels in our test data: 10000


# Imprimir las dimensiones de la imagen y nº de etiquetas en nuestros datos de entrenamiento y prueba
print("\n")
print("Dimensions of x_train:" + str(x_train[0].shape))
print("Labels in x_train:" + str(y_train.shape))
print("\n")
print("Dimensions of x_test:" + str(x_test[0].shape))
print("Labels in y_test:" + str(y_test.shape))
'''Dimensions of x_train:(28, 28)
Labels in x_train:(60000,)

Dimensions of x_test:(28, 28)
Labels in y_test:(10000,)'''

# ## **3. Visualizando nuestro conjunto de datos de imágenes**
# Echemos un vistazo a algunas de las imágenes en este conjunto de datos
# - Usando OpenCV
# - Usando Matplotlib


# Usando OpenCV
# importar opencv y numpy
import cv2 
import numpy as np
from matplotlib import pyplot as plt


def imshow(title, image = None, size = 6):
    if image.any():
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()
    else:
      print("Image not found")


# Use OpenCV para mostrar 6 imágenes aleatorias de nuestro conjunto de datos
for i in range(0,6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    imshow("Sample", img, size = 2)


# ### **Hagamos lo mismo, pero usando matplotlib para trazar 6 imágenes**
# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

# Crear figura y cambiar tamaño
figure = plt.figure()
plt.figure(figsize=(16,10))

# Establecer cuantas imágenes deseamos ver
num_of_images = 50 

# iterar índice de 1 a 51
for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index).set_title(f'{y_train[index]}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')


# ## **4. Preprocesando nuestro conjunto de datos**

# Antes de pasar nuestros datos a nuestra CNN para entrenamiento, primero debemos prepararlos. Este entials:
# 1. Remodelar nuestros datos agregando una 4ta Dimensión
# 2. Cambiar nuestro tipo de datos de uint8 a float32
# 3. Normalizando nuestros datos a valores entre 0 y 1
# 4. Una codificación en caliente

# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# Obtener nuestros datos en la 'forma' correcta necesaria para Keras
# Necesitamos agregar una cuarta dimensión a nuestros datos, cambiando así nuestra
# Nuestra forma de imagen original de (60000,28,28) a (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# cambiar nuestro tipo de imagen a tipo de datos float32
x_train = x_train.astype('float32') # uint8 originalmente
x_test = x_test.astype('float32')

# Normalizar nuestros datos cambiando el rango de (0 a 255) a (0 a 1)
x_train /= 255.0
x_test /= 255.0

print('x_train shape:', x_train.shape)  # x_train shape: (60000, 28, 28, 1)


print(x_train.shape[0], 'train samples')  # 60000 train samples
print(x_test.shape[0], 'test samples' ) # 10000 test samples



print(img_rows, img_cols)  # 28 28


# #### **Una codificación en caliente de nuestras etiquetas (Y)**
#
# Podemos implementar fácilmente este transformm usando ```to_categorical``` de ``` tensorflow.keras.utils```

# En[ ]:


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print ("Number of Classes: " + str(y_test.shape[1]))  # Number of Classes: 10

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# #### **Ejemplo de una codificación activa**
# ![Imagen de una codificación activa]
# (https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/hotoneencode.JPG)

# Mira nuestros datos sin procesar
y_train[0]


# ## **5. Construyendo nuestro modelo**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%204.21.04%402x.png)
# - Estamos construyendo una CNN simple pero efectiva que utiliza 32 filtros de tamaño 3x3
# - Hemos agregado una segunda capa CONV de 64 filtros del mismo tamaño 3x3
# - Luego reducimos la muestra de nuestros datos a 2x2
# - Luego aplanamos nuestra salida Max Pool que está conectada a una capa Dense/FC que tiene un tamaño de salida de 128
# - Luego conectamos nuestras 128 salidas a otra capa FC/Densa que da salida a las 10 unidades categóricas


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD 

# crear modelo
model = Sequential()

# Nuestra primera capa de convolución, tamaño de filtro 32 que reduce el tamaño de nuestra capa a 26 x 26 x 32
# Usamos la activación de ReLU y especificamos nuestro input_shape que es 28 x 28 x 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Nuestra segunda capa de convolución, tamaño de filtro 64 que reduce el tamaño de nuestra capa a 24 x 24 x 64
model.add(Conv2D(64, (3, 3), activation='relu'))

# Usamos MaxPooling con un tamaño de kernel de 2 x 2, esto reduce nuestro tamaño a 12 x 12 x 64
model.add(MaxPooling2D(pool_size=(2, 2)))

# Luego aplanamos nuestro objeto tensor antes de ingresarlo en nuestra capa densa
# Una operación de aplanamiento en un tensor remodela el tensor para que tenga la forma que es
# igual al número de elementos contenidos en el tensor
# En nuestra CNN va de 12*12*64 a 9216*1
model.add(Flatten())

# Conectamos esta capa a una capa Totalmente Conectada/Densa de tamaño 1 * 128
model.add(Dense(128, activation='relu'))

# Creamos nuestra capa final totalmente conectada/densa con una salida para cada clase (10)
model.add(Dense(num_classes, activation='softmax'))

# Compilamos nuestro modelo, esto crea un objeto que almacena el modelo que acabamos de crear
# Configuramos nuestro Optimizer para usar Stochastic Gradient Descent (tasa de aprendizaje de 0.001)
# Configuramos nuestra función de pérdida para que sea categorical_crossentropy ya que es adecuada para problemas
# multiclase
# Finalmente, las métricas (en qué juzgamos nuestro desempeño) para ser precisión
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.001),
              metrics = ['accuracy'])

# Podemos usar la función de resumen para mostrar las capas y los parámetros de nuestro modelo
print(model.summary())
'''Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 9216)              0         
                                                                 
 dense (Dense)               (None, 128)               1179776   
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
'''


### **6. Entrenando a nuestro Modelo**
# - Nuestros datos preprocesados se utilizan como entrada
# - Establecemos el tamaño del lote en 128 (o cualquier número entre 8 y 256 es bueno)
# - Establecemos el número de épocas en 2, esto es solo para el propósito de este tutorial, pero se debe usar un valor
# de al menos 10
# - Almacenamos los resultados de entrenamiento de nuestro modelo para trazar en el futuro
# - Luego usamos la función de evaluación de Molel de Kera para generar el rendimiento final del modelo. Aquí estamos
# examinando la pérdida de prueba y la precisión de la prueba

# En[ ]:


batch_size = 128
epochs = 25

# Almacene nuestros resultados aquí para que podamos graficar más tarde
# En nuestra función de ajuste especificamos nuestros conjuntos de datos (x_train y y_train),
# el tamaño del lote (típicamente de 16 a 128 dependiendo de su RAM), el número de
# épocas (generalmente de 10 a 100) y nuestros conjuntos de datos de validación (x_test & y_test)
# verbose = 1, configura nuestro entrenamiento para generar métricas de rendimiento cada época
history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,  # la información que te da actualmente 0,1 y 2 ma numeración + info
                    validation_data = (x_test, y_test))

'''469/469 [==============================] - 41s 54ms/step - loss: 2.2307 - accuracy: 0.3249 - val_loss: 2.1226 - val_accuracy: 0.5544
Epoch 2/25
469/469 [==============================] - 24s 52ms/step - loss: 1.8059 - accuracy: 0.6838 - val_loss: 1.2555 - val_accuracy: 0.7972
Epoch 3/25
469/469 [==============================] - 23s 49ms/step - loss: 0.8263 - accuracy: 0.8237 - val_loss: 0.5420 - val_accuracy: 0.8657
Epoch 4/25
469/469 [==============================] - 25s 53ms/step - loss: 0.4859 - accuracy: 0.8678 - val_loss: 0.4071 - val_accuracy: 0.8884
Epoch 5/25
469/469 [==============================] - 24s 50ms/step - loss: 0.4012 - accuracy: 0.8860 - val_loss: 0.3566 - val_accuracy: 0.8988
Epoch 6/25
469/469 [==============================] - 34s 72ms/step - loss: 0.3617 - accuracy: 0.8951 - val_loss: 0.3283 - val_accuracy: 0.9047
Epoch 7/25
469/469 [==============================] - 97s 206ms/step - loss: 0.3367 - accuracy: 0.9017 - val_loss: 0.3072 - val_accuracy: 0.9120
Epoch 8/25
469/469 [==============================] - 21s 44ms/step - loss: 0.3180 - accuracy: 0.9079 - val_loss: 0.2925 - val_accuracy: 0.9154
Epoch 9/25
469/469 [==============================] - 4s 8ms/step - loss: 0.3027 - accuracy: 0.9117 - val_loss: 0.2806 - val_accuracy: 0.9198
Epoch 10/25
469/469 [==============================] - 3s 7ms/step - loss: 0.2894 - accuracy: 0.9155 - val_loss: 0.2674 - val_accuracy: 0.9249
Epoch 11/25
469/469 [==============================] - 4s 7ms/step - loss: 0.2776 - accuracy: 0.9183 - val_loss: 0.2580 - val_accuracy: 0.9261
Epoch 12/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2670 - accuracy: 0.9214 - val_loss: 0.2496 - val_accuracy: 0.9290
Epoch 13/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2572 - accuracy: 0.9244 - val_loss: 0.2384 - val_accuracy: 0.9325
Epoch 14/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2478 - accuracy: 0.9272 - val_loss: 0.2316 - val_accuracy: 0.9346
Epoch 15/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2393 - accuracy: 0.9291 - val_loss: 0.2232 - val_accuracy: 0.9362
Epoch 16/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2311 - accuracy: 0.9320 - val_loss: 0.2144 - val_accuracy: 0.9379
Epoch 17/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2234 - accuracy: 0.9342 - val_loss: 0.2088 - val_accuracy: 0.9403
Epoch 18/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2159 - accuracy: 0.9366 - val_loss: 0.2022 - val_accuracy: 0.9419
Epoch 19/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2092 - accuracy: 0.9387 - val_loss: 0.1964 - val_accuracy: 0.9432
Epoch 20/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2028 - accuracy: 0.9407 - val_loss: 0.1909 - val_accuracy: 0.9456
Epoch 21/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1970 - accuracy: 0.9425 - val_loss: 0.1856 - val_accuracy: 0.9465
Epoch 22/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1914 - accuracy: 0.9437 - val_loss: 0.1812 - val_accuracy: 0.9478
Epoch 23/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1862 - accuracy: 0.9453 - val_loss: 0.1795 - val_accuracy: 0.9466
Epoch 24/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1812 - accuracy: 0.9472 - val_loss: 0.1724 - val_accuracy: 0.9489
Epoch 25/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1770 - accuracy: 0.9479 - val_loss: 0.1683 - val_accuracy: 0.9522'''

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])  # Test loss: 0.16829311847686768
print('Test accuracy:', score[1])  # Test accuracy: 0.9521999955177307


### **7. Trazado de nuestras tablas de pérdida y precisión**


history_dict = history.history
history_dict


# Trazando nuestras tablas de pérdidas
import matplotlib.pyplot as plt

# Use el objeto Historial que creamos para obtener nuestros resultados de rendimiento guardados
history_dict = history.history

# Extraer la pérdida y las pérdidas de validación
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# Obtenga el número de épocas y cree una matriz hasta ese número usando range()
epochs = range(1, len(loss_values) + 1)

# Trazar gráficos de líneas para validación y pérdida de entrenamiento
line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# #### **Nuestras tablas de precisión**


# Trazando nuestros gráficos de precisión
import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


# ## **8. Guardando y Cargando nuestro Modelo**
#
# **Guardar nuestro modelo es simple, solo use:**
#
# ```modelo.guardar("modelo_nombre_archivo.h5")```

# En[ ]:


model.save("mnist_simple_cnn_10_Epochs.h5")
print("Model Saved")  # Model Saved


# **Cargar nuestro modelo guardado también es simple, solo use:**

# ```load_model(modelo_nombre_archivo.h5)```

# Necesitamos importar nuestra función load_model
from tensorflow.keras.models import load_model

classifier = load_model('mnist_simple_cnn_10_Epochs.h5')

# ## **9. Obtener predicciones de nuestros datos de prueba de muestra**
#
# **Predicción de todos los datos de prueba**


# x_prueba = x_prueba.reforma(10000,28,28,1)
print(x_test.shape)  # (10000, 28, 28, 1)

print("Predicting classes for all 10,000 test images...")  # Predicting classes for all 10,000 test images...

pred = np.argmax(classifier.predict(x_test), axis=-1)
print("Completed.\n")  # Completed.

print(pred)  # [7 2 1 ... 4 5 6]
print(type(pred))  # <class 'numpy.ndarray'>
print(len(pred))  # 10000


# **Predecir una imagen de prueba individual**

# Obtenga la primera imagen por índice 0 de x_test y muestre su forma
input_im = x_test[0]
print(input_im.shape)  # (28, 28, 1)

# Necesitamos agregar una cuarta dimensión al primer eje
input_im = input_im.reshape(1,28,28,1)
print(input_im.shape)  # (1, 28, 28, 1)

# Ahora obtenemos las predicciones para esa sola imagen
pred = np.argmax(classifier.predict(input_im), axis=-1)   # 1/1 [==============================] - 0s 34ms/step
print(pred)  # [7]
print(type(pred))  # <class 'numpy.ndarray'>
print(len(pred))  # 1


# ### **Ahora hagamos algo elegante, pongamos la etiqueta predicha en una imagen con la imagen de datos de prueba**

import cv2
import numpy as np

# Recargar nuestros datos ya que lo reescalamos
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

def draw_test(name, pred, input_im):  
    '''Function that places the predicted class next to the original image'''
    # Crea nuestro fondo negro
    BLACK = [0,0,0]
    # Ampliamos nuestra imagen original a la derecha para crear espacio para colocar nuestro texto de clase predicho
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    # convertir nuestra imagen en escala de grises a color
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    # Ponga nuestro texto de clase predicho en nuestra imagen expandida
    cv2.putText(expanded_image, str(pred), (150, 80) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0,255,0), 2)
    imshow(name, expanded_image)

for i in range(0,10):
    # Obtenga una imagen de datos aleatorios de nuestro conjunto de datos de prueba
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]

    # Cree una imagen redimensionada más grande para contener nuestro texto y permitir una visualización más grande
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    # Reformar nuestros datos para que podamos ingresarlos (hacia adelante propagarlos) a nuestra red
    input_im = input_im.reshape(1,28,28,1) 
    
    # Obtener predicción, use [0] para acceder al valor en la matriz numpy ya que está almacenada como una matriz
    res = str(np.argmax(classifier.predict(input_im), axis=-1)[0])

    # Coloque la etiqueta en la imagen de nuestra muestra de datos de prueba
    draw_test("Prediction", res,  np.uint8(imageL)) 

