#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Uso de modelos previamente entrenados en Keras**
# ### **Cargaremos los pesos de modelos preentrenados avanzados como:**
#
# ---
#
#
# 1. VGG16
# 2. ResNet
# h. Inicio vs
#4. Mobilina F
#5. Electricidad t201
# 6. NASNet móvil
# 7. EficienteNetB7
#
# https://keras.io/api/applications/

# En[ ]:


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')
model.summary()


# En[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')
get_ipython().system('rm rf images/class1/.DS_Store')


# En[ ]:


import cv2
from os import listdir
from os.path import isfile, join

# Obtener imágenes ubicadas en la carpeta ./images
mypath = "./images/class1/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_names


# En[ ]:


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# # **2. ResNet50**

# En[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


## **h. Inicio Bz**

# En[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

model = InceptionV3(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    # tenga en cuenta el cambio en el tamaño de la imagen de entrada a 299,299
    img = image.load_img(mypath+file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# # **4. red móvil**

# En[ ]:


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

model = MobileNetV2(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    # from keras.preprocessing import image # Necesidad de recargar ya que opencv2 parece tener un conflicto
    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# # **5. Denseno t201**

# En[ ]:


from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np

model = DenseNet201(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# # **6. NASNet móvil**

# En[ ]:


from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input
import numpy as np

model = NASNetMobile(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


### **7. EficienteNetB7**
#
# Pruebe otras EfficientNets B0 a B7 - https://keras.io/api/applications/ficientnet/

# En[ ]:


from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

model = EfficientNetB7(weights='imagenet')
model.summary()


# En[ ]:


fig=plt.figure(figsize=(16,16))

# Bucle a través de las imágenes, páselas a través de nuestro clasificador
for (i,file) in enumerate(file_names):

    # El tamaño de entrada de la nota ha aumentado a 600,600
    img = image.load_img(mypath+file, target_size=(600, 600))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # cargar imagen usando opencv
    img2 = cv2.imread(mypath+file)
    # imageL = cv2.resize(img2, Ninguno, fx=.5, fy=.5, interpolación = cv2.INTER_CUBIC)
    
    # Obtener predicciones
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    print(preditions)
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions[0][1])}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# En[ ]:




