#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Keras Cats vs Dogs - Extracción de funciones**
#
# ---
#
# En esta lección, aprenderemos a usar una red preentrenada como extractor de funciones. Luego usaremos esas características como entrada para nuestro clasificador de regresión logística.
# 1. Descarga y explora nuestros datos
# 2. Cargue nuestro modelo VGG16 preentrenado
# 3. Extraiga nuestras características usando VGG16
# 4. Entrena un Clasificador LR usando esas características
# 5. Prueba algunas inferencias
#
# ### **Necesitará usar High-RAM y GPU (para aumentar la velocidad).**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)

### **1. Descarga y explora nuestros datos**

# En 1]:


from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications import VGG16, imagenet_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random
import tqdm
import os


# En 2]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/dogs-vs-cats.zip')
get_ipython().system('unzip -q gatos_perros.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# ### **Cargando nuestros datos y sus etiquetas en un marco de datos**
#
# Hay muchas formas en que podemos hacer esto, esta forma es relativamente simple de seguir.

# En 3]:


filenames = os.listdir("./train")

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


### **2. Cargue nuestro modelo VGG16 preentrenado**

# En[4]:


model = VGG16(weights="imagenet", include_top=False)


# En[5]:


model.summary()


# ## **¿Qué estamos haciendo exactamente?**
#
# Estamos tomando la salida de la última capa CONV-POOL (ver abajo).
#
# La forma de salida en esta capa es **7 x 7 x 512**
#
# ![feat_extraction](https://appliedmachinelearning.files.wordpress.com/2021/05/ef54e-vgg16.png?w=612&zoom=2)
# Imagen referenciada desde [aquí](https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/)

# ### **Almacenar nuestras rutas de imágenes y nombres de etiquetas**

### **3. Extrae nuestras características usando VGG16**

# En[28]:


batch_size = 32
image_features = []
image_labels = []

# bucle sobre cada lote
for i in range(0, len(image_paths)//batch_size):
  # extraer nuestros lotes
  batch_paths = image_paths[i:i + batch_size]
  batch_labels = labels[i:i + batch_size]
  batch_images = []

  # iterar sobre cada imagen y extraer las características de nuestra imagen
  for image_path in batch_paths:
    # cargar imágenes usando load_img() de Keras y cambiar el tamaño a 224 x 244
    image = load_img(image_path, target_size = (224, 224))
    image = img_to_array(image)

    # Expandimos las dimensiones y luego restamos la intensidad media de píxeles RGB de ImageNet
    # usando la función imagenet_utils.preprocess_input()
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # agregar nuestras características de imagen a nuestra lista de lotes
    batch_images.append(image)

  # tomamos nuestro lote de imágenes y las colocamos en el formato correcto con vstack
  batch_images = np.vstack(batch_images)

  # luego usamos ese lote y lo ejecutamos a través de nuestra función de predicción
  features = model.predict(batch_images, batch_size = batch_size)

  # luego tomamos la forma de salida 7x7x512 y la aplanamos
  features = np.reshape(features,(-1, 7*7*512))

  # almacenar nuestras características y etiquetas correspondientes
  image_features.append(features)
  image_labels.append(batch_labels)


# En[29]:


# veamos la imagen imageFeatures
print(image_features[0].shape)
image_features[0]


# En[30]:


image_labels


### **4. Entrene a un clasificador LR usando esas características**
#
# Primero, almacenemos nuestra información de función extraída en un formato que sklearn pueda cargar directamente.

# En[ ]:


# tome nuestra lista de lotes y reduzca la dimensión para que ahora sea una lista de 25088 funciones x 25000 filas (25000 x 1 para nuestras etiquetas)
imageLabels_data =  [lb for label_batch in image_labels for lb in label_batch]
imageFeatures_data = [feature for feature_batch in image_features for feature in feature_batch]

# Convertir a matrices numpy
image_labels_data = np.array(imageLabels_data)
image_features_data = np.array(imageFeatures_data)


# En[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

y = image_labels_data 

# Divida nuestro modelo en un conjunto de datos de prueba y entrenamiento para entrenar nuestro clasificador LR
X_train, X_test, y_train, y_test = train_test_split(image_features_data, y, test_size=0.2, random_state = 7)

glm = LogisticRegression(C=0.1)
glm.fit(X_train,y_train)


# En[ ]:


# Obtenga precisión en el 20 % que dividimos de su conjunto de datos de entrenamiento
accuracy = glm.score(X_test, y_test)
print(f'Accuracy on validation set using Logistic Regression: {accuracy*100}%')


### **5. Pruebe algunas inferencias**

# En[ ]:


image_names_test = os.listdir("./test1")
image_paths_test = ["./test1/"+ x for x in image_names_test]


# En[ ]:


import random 

test_sample = random.sample(image_paths_test, 12)

def test_img():
    result_lst = []
    for path in test_sample:
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        features = model.predict(image)
        features = np.reshape(features,(-1,7*7*512))
        result = glm.predict(features)
        result = 'dog' if float(result) >0.5 else 'cat'
        result_lst.append(result)
    return result_lst


# En[ ]:


# obtener predicciones de prueba de todos los modelos
pred_results = test_img()
pred_results


# En[ ]:


plt.figure(figsize=(15, 15))

for i in range(0, 12):
    plt.subplot(4, 3, i+1)
    result = pred_results[i]
    img = test_sample[i]
    image = load_img(img, target_size=(256,256))
    plt.text(72, 248, f'Feature Extractor CNN: {result}', color='lightgreen',fontsize= 12, bbox=dict(facecolor='black', alpha=0.9))
    plt.imshow(image)

plt.tight_layout()
plt.show()


# ## **¿Cómo nos comparamos con los 10 mejores de Kaggle?**
# https://www.kaggle.com/c/dogs-vs-cats/leaderboard
#
# ¡Obtuvimos 98.34%, segundo lugar! No está nada mal :)
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%208.09.25%20pm.png)

# En[ ]:




