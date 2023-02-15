#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Análisis de rendimiento del modelo Keras**
# ---
#
#
#
# ---
#
#
#
# En esta lección, aprendemos a usar el modelo MNIST que entrenamos en la lección anterior y analizamos su desempeño, hacemos:
# 1. Cargue nuestro modelo y datos de Keras
# 2. Ver las imágenes que clasificamos mal
# 3. Crea una matriz de confusión
# 4. Crear informe de clasificación
#

### **1. Cargue nuestro modelo Keras y el conjunto de datos MNIST**
#
# **Descargue nuestro modelo anterior y cárguelo con load_model**

# En[5]:


get_ipython().system('gdown --id 1jW5aHd7_fAi3UrbT9MRTDbKyxjwfQ3WC')


# En[6]:


# Necesitamos importar nuestra función load_model
from tensorflow.keras.models import load_model

model = load_model('mnist_simple_cnn_10_Epochs.h5')


# **Cargar nuestro conjunto de datos MNIST**
#
# Técnicamente, solo necesitamos cargar el conjunto de datos de prueba, ya que estamos analizando el rendimiento en ese segmento de datos.

# En[7]:


# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test)  = mnist.load_data()


# # **2. Ver nuestras clasificaciones erróneas**
# #### **Primero obtengamos nuestras predicciones de prueba**

# En[8]:


import numpy as np

# Reformamos nuestros datos de prueba
print(x_test.shape)
x_test = x_test.reshape(10000,28,28,1) 
print(x_test.shape)

# Obtenga las predicciones para todas las muestras de 10K en nuestros datos de prueba
print("Predicting classes for all 10,000 test images...")
pred = np.argmax(model.predict(x_test), axis=-1)
print("Completed.\n")


# En[11]:


import cv2
import numpy as np

# Use numpy para crear una matriz que almacene un valor de 1 cuando ocurra una clasificación incorrecta
result = np.absolute(y_test - pred)
misclassified_indices = np.nonzero(result > 0)

# Mostrar los índices de clasificaciones erróneas
print(f"Indices of misclassifed data are: \n{misclassified_indices}")
print(len(misclassified_indices[0]))


# ### **Visualización de las imágenes clasificadas incorrectamente por nuestro modelo**

# En[18]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imshow(title="", image = None, size = 6):
    if image.any():
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()
    else:
      print("Image not found")


# En 19]:


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
    input_im = x_test[misclassified_indices[0][i]]
    # Cree una imagen redimensionada más grande para contener nuestro texto y permitir una visualización más grande
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    # Reformar nuestros datos para que podamos ingresarlos (hacia adelante propagarlos) a nuestra red
    input_im = input_im.reshape(1,28,28,1) 
    
    # Obtener predicción, use [0] para acceder al valor en la matriz numpy ya que está almacenada como una matriz
    res = str(np.argmax(model.predict(input_im), axis=-1)[0])

    # Coloque la etiqueta en la imagen de nuestra muestra de datos de prueba
    draw_test("Misclassified Prediction", res,  np.uint8(imageL)) 


# ### **Una forma más elegante de trazar esto**

# En 20]:


L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L * W):  
    input_im = x_test[misclassified_indices[0][i]]
    ind = misclassified_indices[0][i]
    predicted_class = str(np.argmax(model.predict(input_im.reshape(1,28,28,1)), axis=-1)[0])
    axes[i].imshow(input_im.reshape(28,28), cmap='gray_r')
    axes[i].set_title(f"Prediction Class = {predicted_class}\n Original Class = {y_test[ind]}")
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)


## **3. Creando nuestra Matriz de Confusión**
#
# Usamos la herramienta Confusion Matrix de Sklean para crearlo. Todo lo que necesitamos es:
# 1. Las verdaderas etiquetas
# 2. Las etiquetas predichas
#

# En[21]:


from sklearn.metrics import confusion_matrix
import numpy as np

x_test = x_test.reshape(10000,28,28,1) 
y_pred = np.argmax(model.predict(x_test), axis=-1)

print(confusion_matrix(y_test, y_pred))


# #### **Interpretación de la matriz de confusión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2010.46.45.png)
#
# ### **Creando una trama más presentable**
#
# Reutilizaremos esta función muy bien hecha de la documentación de sklearn sobre el trazado de una matriz de confusión usando gradientes de color y etiquetas.

# En[22]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """dada una matriz de confusión de sklearn (cm), haga una buena trama

Argumentos
---------
cm: matriz de confusión de sklearn.metrics.confusion_matrix

target_names: clases de clasificación dadas como [0, 1, 2]
los nombres de las clases, por ejemplo: ['high', 'medium', 'low']

título: el texto que se mostrará en la parte superior de la matriz

cmap: el gradiente de los valores mostrados desde matplotlib.pyplot.cm
consulte http://matplotlib.org/examples/color/colormaps_reference.html
plt.get_cmap('jet') o plt.cm.Blues

normalizar: si es falso, graficar los números sin procesar
Si es Verdadero, traza las proporciones

Uso
-----
plot_confusion_matrix(cm           = cm,                  # matriz de confusión creada por
# sklearn.metrics.confusion_matrix
normalize    = True,                # mostrar proporciones
target_names = y_labels_vals,       # lista de nombres de las clases
title        = best_estimator_name) # título del gráfico

Citación
---------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# En[23]:


target_names = list(range(0,10))
conf_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat, target_names)


# ## **Veamos nuestra precisión por clase**

# En[24]:


# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

for (i,ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')


# # **4. Ahora veamos el Informe de Clasificación**

# En[25]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### **4.1 El soporte es la suma total de esa clase en el conjunto de datos**
#
# ### **4.2 Revisión del retiro del mercado**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.12.png)
#
# ### **4.3 Revisión de la precisión**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.22.png)
#
# ### **4.4 Alta recuperación (o sensibilidad) con baja precisión.**
# Esto nos dice que la mayoría de los ejemplos positivos se reconocen correctamente (falsos negativos bajos), pero hay muchos falsos positivos, es decir, otras clases se predicen como nuestra clase en cuestión.
#
# ### **4.5 Baja recuperación (o sensibilidad) con alta precisión.**
#
# A nuestro clasificador le faltan muchos ejemplos positivos (FN alto), pero aquellos que predecimos como positivos son realmente positivos (Falsos positivos bajos)
#

# En[ ]:




