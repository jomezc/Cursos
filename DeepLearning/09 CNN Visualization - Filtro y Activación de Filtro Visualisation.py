#!/usr/bin/env python
# codificación: utf-8
#######################################################################
# 09 CNN Visualization - Filtro y Activación de Filtro Visualisation####
#######################################################################
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Visualización de lo que aprenden las CNN**

# En esta lección, usamos **Keras con TensorFlow 2.0** para visualizar lo siguiente (ver más abajo). Esto lo ayuda a obtener una mejor comprensión de lo que sucede debajo del capó y desmitifica algunos de los aspectos del aprendizaje profundo.**
# 1. Entrenamiento de una CNN básica en el conjunto de datos MNIST
#2. Visualiza sus filtros
# 3. Visualiza las activaciones del filtro mientras propagamos una imagen de entrada

# # **Entrenamiento de una CNN básica en el conjunto de datos MNIST**

# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

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

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.001),
              metrics = ['accuracy'])

print(model.summary())


batch_size = 128
epochs = 10

# Almacene nuestros resultados aquí para que podamos graficar más tarde
# En nuestra función de ajuste especificamos nuestros conjuntos de datos (x_train y y_train),
# el tamaño del lote (típicamente de 16 a 128 dependiendo de su RAM), el número de
# épocas (generalmente de 10 a 100) y nuestros conjuntos de datos de validación (x_test & y_test)
# verbose = 1, configura nuestro entrenamiento para generar métricas de rendimiento cada época
history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_test, y_test))

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ## **Obtenga las salidas simbólicas de cada capa "clave" (les dimos nombres únicos).**


layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_dict


# ## **Obtenga las formas solo de nuestros filtros de conversión**


# resumir formas de filtro
for layer in model.layers:
  # verificar la capa convolucional
  
  if 'conv' not in layer.name:
    continue

  # obtener pesos de filtro
  filters, biases = layer.get_weights()
  print(layer.name, filters.shape)


# ## **Echemos un vistazo a los pesos de nuestra primera capa de conversión**

# recuperar pesos de la primera capa Conv (oculta)
filters, biases = model.layers[0].get_weights()


# Echemos un vistazo a nuestros filtros
print(filters.shape)
filters



# Y ahora veamos nuestros sesgos
print(biases.shape)
biases


# ## **Normalicemos los valores de filtro a 0-1 para poder visualizarlos**
#
# Nuestra gama de pesos de filtro

# normalizar los valores de filtro a 0-1 para que podamos visualizarlos
f_min, f_max = filters.min(), filters.max()
print(f'Before Normalisation, Min = {f_min} and Max =  {f_max}')
filters = (filters - f_min) / (f_max - f_min)
print(f'After Normalisation, Min = {filters.min()} and Max =  {filters.max()}')


# ## **Visualizar nuestros rellenos entrenados**


import matplotlib.pyplot as plt
import numpy as np

# trazar los primeros filtros y establecer el tamaño de la trama
n_filters, ix = 32, 1
plt.figure(figsize=(12,20))

for i in range(n_filters):
    # obtener el filtro
    f = filters[:, :, :, i]
    # imprimir(f.forma)

    # Arreglar en subparcela de 4 x 8
    ax = plt.subplot(n_filters, 4, ix)
    ax.set_xticks([])
    ax.set_yticks([])

    # trazar el canal de filtro en escala de grises
    plt.imshow(np.squeeze(f, axis=2), cmap='gray')
    ix += 1
    
# mostrar la figura
plt.show()


# # **Activaciones de filtros**


from tensorflow.keras.models import Model

# Extrae las salidas de las 2 capas superiores
layer_outputs = [layer.output for layer in model.layers[:2]]

# Crea un modelo que devolverá estos resultados, dada la entrada del modelo
activation_model = Model(inputs=model.input, outputs=layer_outputs)



layer_outputs


import matplotlib.pyplot as plt

img_tensor = x_test[22].reshape(1,28,28,1)
fig = plt.figure(figsize=(5,5))
plt.imshow(img_tensor[0,:,:,0],cmap="gray")
plt.axis('off')


# ## **Obtenga la salida después del segundo filtro de conversión (después de ReLU)**

# Devuelve una lista de dos matrices Numpy: una matriz por activación de capa
activations = activation_model.predict(img_tensor)

print("Number of layer activations: " + str(len(activations)))


# #### **La activación de la primera capa de convolución para la entrada de imagen**


first_layer_activation = activations[0]
print(first_layer_activation.shape)



second_layer_activation = activations[1]
print(second_layer_activation.shape)


print(model.summary())


# ## **La salida del mapa de características del 4.° filtro de conversión en la primera capa de conversión**


plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.legend()


# ## **Crear una función que muestre la activación de capas específicas**

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='viridis')
            activation_index += 1


display_activation(activations, 4, 8, 0)




