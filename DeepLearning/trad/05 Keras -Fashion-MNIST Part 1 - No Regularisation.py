#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Regularización en Keras - Parte 1 - Sin regularización**
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
# 2. Preprocesamiento de nuestros datos
# 3. Construya una CNN simple sin regularización
# 4. Capacitar a nuestra CNN
# 5. Eche un vistazo al aumento de datos
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


# # **2. Preprocesamiento de datos**
#
# Primero, hagamos un seguimiento de algunas dimensiones de datos:
# - ```img_rows``` que debería ser 28
# - ```img_cols``` que debería ser 28
# - ```input_shape```, que es 28 x 28 x 1

# En[5]:


# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)


# ### **Una codificación en caliente de nuestras etiquetas**
# **Ejemplo de una codificación activa**
# ![Imagen de una codificación activa](https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/hotoneencode.JPG)
# Además, mantenga las clases de números almacenadas como una variable, ```num_classess```

# En[6]:


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]


## **3. Construyendo nuestro modelo**
#
# Esta es la misma CNN que usamos anteriormente para el proyecto de clasificación MNIST.
#
# **Agregar capas de conversión**
#
# ```Conv2D(32, kernel_size=(3, 3),
# activación='releer',
# forma_entrada=forma_entrada)```
#
# Nuestro **Conv2D()** crea el filtro con los siguientes argumentos:
# - Número de filtros, usamos 32
# - kernel_size, usamos un filtro 3x3 por lo que se define como una tupla ```(3,3)```
# - activación, donde especificamos ```'relu'```
# - input_shape, que obtuvimos y almacenamos en una variable anterior, en nuestro ejemplo es una imagen en escala de grises de 28 x 28. Por lo tanto, nuestra forma es ```(28,28,1)```

# **Agregar capas de MaxPool**
#
# De nuevo, usamos ```model.add()``` y especificamos ```MaxPooling2D(pool_size=(2,2))```.
#
# Usamos el argumento de entrada pool_size para definir el tamaño de nuestra ventana. Podemos especificar la zancada y el relleno de esta manera:
#
# ```pool_size=(2, 2), strides=Ninguno, relleno='válido'```
#
# Sin embargo, tenga en cuenta que la zancada predeterminada se usa como el tamaño de la ventana de agrupación (2 en nuestro caso).
#
# Usar ```padding ='valid'``` significa que no usamos relleno.
#
# **Añadiendo Flatten**
#
# Usando model.add(Flatten()) simplemente estamos aplanando la salida de nuestro último nodo. Lo que equivale a 12 x 12 * 64 * 1 = 9216.
#
# **Agregar capas densas o completamente conectadas**
#
# ```modelo.add(Dense(128, activación='relu'))```
#
# Usamos ```model.add()``` una vez más y especificamos el número de nodos que nuestra capa anterior también conectará. También especificamos la función de activación de ReLU aquí.

# En[7]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD 

# crear modelo
model = Sequential()

# Agrega nuestras capas usando model.add()

# Creamos una capa Conv2D con nuestras especificaciones
# Aquí estamos usando 32 filtros, de tamaño 3x3 con activación ReLU
# Nuestra forma de entrada es 28 x 28 x 1
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# Agregamos una segunda capa Conv con 64 filtros, 3x3 y activación ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Usamos un MaxPool estándar de 2x2 y un paso de 2 (por defecto, Kera usa el mismo paso que el tamaño de la ventana)
model.add(MaxPooling2D(pool_size=(2, 2)))
# Ahora aplanamos la salida de nuestras capas anteriores, que es 12 x 12 * 64 * 1 = 9216
model.add(Flatten())
# Ahora conectamos esto aplanado más tarde a 128 Nodos de nuestra Capa Completamente Conectada o Densa, nuevamente usando ReLU
model.add(Dense(128, activation='relu'))
# Ahora creamos nuestra última capa totalmente conectada/densa que consta de 10 nodos que corresponden a nuestras clases de salida
# Esto luego se usa con una activación 'softmax' para darnos nuestras probabilidades finales de clase
model.add(Dense(num_classes, activation='softmax'))


# ### **Compilando nuestro modelo**
#
# Aquí usamos ```model.compile()``` para compilar nuestro modelo. Especificamos lo siguiente:
# - Función de pérdida - categorical_crossentropy
# - Optimizer - SGD o Stochastic Gradient Descent (tasa de aprendizaje de 0.001 y momento 0.9)
# - métricas - con qué criterios evaluaremos el rendimiento. Usamos precisión aquí.

# En[8]:


# Compilamos nuestro modelo, esto crea un objeto que almacena el modelo que acabamos de crear
# Configuramos nuestro Optimizer para usar Stochastic Gradient Descent (tasa de aprendizaje de 0.001)
# Configuramos nuestra función de pérdida para que sea categorical_crossentropy ya que es adecuada para problemas multiclase
# Finalmente, las métricas (en qué juzgamos nuestro desempeño) para ser precisión
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.001, momentum=0.9),
              metrics = ['accuracy'])

# Podemos usar la función de resumen para mostrar las capas y los parámetros de nuestro modelo
print(model.summary())


# # **4. Entrenando Nuestro Modelo**

# En[9]:


# Establecer nuestro tamaño de lote y épocas
batch_size = 32
epochs = 15

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


# # **5. Ejemplo de aumento de datos**
#
# Usamos generadores porque no podemos cargar todo el conjunto de datos en la memoria de nuestros sistemas. Por lo tanto, utilícelo para crear un iterador para que podamos acceder a lotes de nuestros datos para el aumento o preprocesamiento de datos y propagarlos a través de nuestra red.

# En[11]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Recargar nuestros datos
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

# Reformar nuestros datos para que tengan el formato [número de muestras, ancho, alto, color_profundidad]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cambiar el tipo de datos a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Definir generador de datos para aumento
data_aug_datagen = ImageDataGenerator(rotation_range=30,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.2,
                                      zoom_range=0.1,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

# Crea nuestro iterador
aug_iter = data_aug_datagen.flow(x_train[0].reshape(1,28,28,1), batch_size=1)


# #### **Mostrar los resultados de nuestro aumento de datos**

# En[12]:


import cv2

def showAugmentations(augmentations = 6):
    fig = figure()
    for i in range(augmentations):
        a = fig.add_subplot(1,augmentations,i+1)
        img = next(aug_iter)[0].astype('uint8')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axis('off')

showAugmentations(6)


# En[ ]:




