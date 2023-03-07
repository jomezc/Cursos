#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Uso de modelos preentrenados en Keras para obtener precisión de rango 1 y rango 5**
# 1. Primero cargaremos el modelo de ImageNet pre-entrenado MobileNetV2
# 2. Obtendremos las 5 mejores clases a partir de una sola inferencia de imagen
# 3. A continuación, construiremos una función que nos dé la precisión de rango N usando algunas imágenes de prueba.
#
# ---# El rango es una forma de dar al clasificador como precisión un poco más de margen.
# # Entonces, en lugar de regresar,  una clase particular, lo que haría de forma natural, Cuando consideramos la
# # inexactitud, miramos los cinco primeros o el árbol superior, si la clase correcta pertenece a las cinco clases de
# # mayor probabilidad que se generan desde CNN, entonces consideramos que está correctamente identificado.
# # 
#

# En[ ]:


# Cargue nuestro modelo MobileNetV2 preentrenado

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

model = MobileNetV2(weights='imagenet')
model.summary()


# Obtenga los nombres de las etiquetas de clase de imageNet y las imágenes de prueba
'''wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip'
get_ipython().system('unzip imagesDLCV.zip')
get_ipython().system('rm -rf ./images/class1/.DS_Store'
'''


import cv2
from os import listdir
from os.path import isfile, join

# Obtener imágenes ubicadas en la carpeta ./images
mypath = "images/class1/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_names




import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,16))
all_top_classes = []

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
    preditions = decode_predictions(preds, top=10)[0]
    all_top_classes.append([x[1] for x in preditions])
    # Trazar imagen
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions)}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()



print(preditions)




print(all_top_classes)



# Crea nuestras etiquetas de verdad en el suelo
ground_truth = ['basketball',
                'German shepherd',
                'limousine, limo',
                'spider_web',
                'burrito',
                'beer_glass',
                'doormat',
                'Christmas_stocking',
                'collie']



def getScore(all_top_classes, ground_truth, N):
  # Calcular rango-Y puntuación
  in_labels = 0
  for (i,labels) in enumerate(all_top_classes):
    if ground_truth[i] in labels[:N]:
      in_labels += 1
  return f'Rank-{N} Accuracy = {in_labels/len(all_top_classes)*100:.2f}%'


# ## **Obtener precisión de rango 5**

getScore(all_top_classes, ground_truth, 5)


# ## **Obtener precisión de rango 1**

getScore(all_top_classes, ground_truth, 1)



getScore(all_top_classes, ground_truth, 10)






