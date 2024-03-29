#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Keras Cats vs Dogs - Entrenamiento con tus propios datos**
#
# ---
#
# En esta lección, aprendemos cómo configurar generadores de datos para cargar nuestro propio conjunto de datos y
# entrenar un clasificador usando Keras.

# 1. Descarga y explora nuestros datos
# 2. Crea una CNN simple
# 3. Crea nuestros Generadores de Datos
# 4. Entrena a nuestro modelo
# 5. Prueba algunas inferencias
# 6. Puntos de control

# En[5]:


# importar nuestros paquetes

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


### **1. Descarga y explora nuestros datos** ( en 16 ***)


# Definir el tamaño de nuestras imágenes
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# ### **Cargando nuestros datos y sus etiquetas en un marco de datos**
#
# Hay muchas formas en que podemos hacer esto, esta forma es relativamente simple de seguir.
#
# `perro.1034234.jpg`


filenames = os.listdir("images/gatos_perros/train")

categories = []

for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'class': categories
})
df.head()


# ### **Consulta los conteos en cada clase**

# En[ ]:


df['class'].value_counts().plot.bar()


# #### **Ver una imagen de muestra**

# En[9]:


sample = random.choice(filenames)
image = load_img("images/gatos_perros/train/" + sample)
plt.imshow(image)


### **2. Crea nuestro modelo**

# En[11]:


from keras.models import Sequential
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  #2 porque tenemos clases de perros y gatos

opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# ## **3. Crea nuestros generadores de datos**

df["class"] = df["class"].replace({0: 'cat', 1: 'dog'}) 
df.head()


# #### **Dividir nuestro conjunto de datos usando train_test_split**

# En[13]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=7)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


train_df.head()


validate_df.head()


# ### **Cree nuestro generador de datos de entrenamiento**

batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "images/gatos_perros/train/",
    x_col = 'filename',
    y_col = 'class',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size,
    # Error , sol:https://stackoverflow.com/questions/57123656/way-to-print-invalid-filenames-for-generator-in-keras
    validate_filenames=False
)


# ### **Crear nuestro Generador de Datos de Validación**

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "images/gatos_perros/train",
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
    validate_filenames=False
)


# #### **Crear un generador de datos de ejemplo para cargar solo una imagen**


example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "images/gatos_perros/train",
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    validate_filenames=False
)


# #### **Vista previa de esa imagen**


plt.figure(figsize=(6, 6))

for X_batch, Y_batch in example_generator:
    image = X_batch[0]
    plt.imshow(image)
    break

plt.tight_layout()
plt.show()


### **4. Comience a entrenar nuestro modelo**

epochs = 10

history = model.fit(
    train_generator, 
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = 5000//batch_size,
    steps_per_epoch = 20000//batch_size,
)

# Guardar nuestro modelo
model.save_weights("cats_vs_dogs_10_epochs.h5")


# Viewour para gráficos de rendimiento

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# #### **Obtenga las predicciones de nuestras imágenes de validación**

# Ejecutamos nuestra predicción en todas las imágenes del conjunto de validación
predict = model.predict_generator(validation_generator, steps = np.ceil(5000/batch_size))


# #### **Añádelo a nuestro marco de datos para verlo fácilmente**


validate_df['predicted'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
validate_df['predicted'] = validate_df['predicted'].replace(label_map)
print(validate_df)


# ## **Inferencia en un lote de imágenes de nuestro conjunto de datos de validación**

sample_test = validate_df.head(18)
sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['predicted']
    img = load_img("images/gatos_perros/train/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()
plt.show()



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



checkpoint = ModelCheckpoint("MNIST_Checkpoint.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', # valor que se está monitoreando para mejorar
                          min_delta = 0, # Valor de Abs y es el cambio mínimo requerido antes de parar
                          patience = 5, # Número de épocas que esperamos antes de parar
                          verbose = 1,
                          restore_best_weights = True) # mantiene los mejores pesos una vez detenido


# ### **Otra devolución de llamada útil es Reducir nuestra tasa de aprendizaje en Plateau**
#
# Podemos evitar que nuestro sistema oscile alrededor del mínimo global al intentar reducir la tasa de aprendizaje por
# un hecho determinado. Si no se ve una mejora en nuestra métrica monitoreada (normalmente, val_loss), esperamos un
# cierto número de épocas (paciencia), luego esta devolución de llamada reduce la tasa de aprendizaje en un factor.


from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)


# colocamos nuestras devoluciones de llamadas en una lista de devoluciones de llamadas
callbacks = [earlystop, checkpoint, reduce_lr]


epochs = 10

history = model.fit(
    train_generator, 
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = 5000//batch_size,
    steps_per_epoch = 20000//batch_size,
)





