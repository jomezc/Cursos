#!/usr/bin/env python
# codificación: utf-8
##################################################################################
# 03 Clasificaciones erróneas de Keras y análisis de rendimiento del modelo######
##################################################################################
# # **Análisis de rendimiento del modelo Keras**
# En esta lección, aprendemos a usar el modelo MNIST que entrenamos en la lección anterior y analizamos su desempeño,
# hacemos:
# 1. Cargue nuestro modelo y datos de Keras
# 2. Ver las imágenes que clasificamos mal
# 3. Crea una matriz de confusión
# 4. Crear informe de clasificación

# ## **1. Cargue nuestro modelo Keras y el conjunto de datos MNIST**
# **Descargue nuestro modelo anterior (02) y cárguelo con load_model**

# Necesitamos importar nuestra función load_model
from tensorflow.keras.models import load_model

model = load_model('models/mnist_simple_cnn_10_Epochs.h5')


# **Cargar nuestro conjunto de datos MNIST**
# Técnicamente, solo necesitamos cargar el conjunto de datos de prueba, ya que estamos analizando el rendimiento en ese
# segmento de datos.


# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# # **2. Ver nuestras clasificaciones erróneas**
# #### **Primero obtengamos nuestras predicciones de prueba**

import numpy as np

# Reformamos nuestros datos de prueba
print(x_test.shape)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_test.shape)

# Obtenga las predicciones para todas las muestras de 10K en nuestros datos de prueba
print("Predicting classes for all 10,000 test images...")
pred = np.argmax(model.predict(x_test), axis=-1)  # 313/313 [==============================] - 3s 1ms/step
print("Completed.\n")  # Completed.


import cv2
import numpy as np

# Use numpy para crear una matriz que almacene un valor de 1 cuando ocurra una clasificación incorrecta
result = np.absolute(y_test - pred)
misclassified_indices = np.nonzero(result > 0)

# Mostrar los índices de clasificaciones erróneas
print(f"Indices of misclassifed data are: \n{misclassified_indices}")
'''Indices of misclassifed data are: 
(array([   8,   62,  195,  241,  247,  259,  290,  300,  318,  320,  321,
        340,  341,  352,  362,  412,  445,  448,  449,  478,  502,  507,
        515,  531,  551,  565,  578,  591,  613,  619,  628,  659,  684,
        689,  691,  707,  717,  720,  740,  791,  795,  839,  898,  938,
        939,  947,  950,  951,  965,  975,  990, 1003, 1014, 1032, 1039,
       1044, 1062, 1068, 1073, 1082, 1107, 1112, 1114, 1191, 1198, 1204,
       1206, 1226, 1232, 1234, 1242, 1243, 1247, 1260, 1270, 1283, 1289,
       1299, 1319, 1326, 1328, 1337, 1364, 1378, 1393, 1410, 1433, 1440,
       1444, 1453, 1494, 1500, 1522, 1527, 1530, 1549, 1553, 1581, 1609,
       1634, 1640, 1671, 1681, 1709, 1717, 1718, 1737, 1754, 1774, 1790,
       1800, 1850, 1865, 1878, 1883, 1901, 1911, 1938, 1940, 1952, 1970,
       1981, 1984, 2016, 2024, 2035, 2037, 2040, 2043, 2044, 2053, 2070,
       2098, 2109, 2118, 2125, 2129, 2130, 2135, 2182, 2185, 2186, 2189,
       2215, 2224, 2266, 2272, 2293, 2299, 2325, 2369, 2371, 2381, 2387,
       2394, 2395, 2406, 2408, 2414, 2422, 2425, 2433, 2488, 2545, 2573,
       2574, 2607, 2610, 2648, 2654, 2751, 2760, 2771, 2780, 2810, 2832,
       2836, 2863, 2896, 2914, 2925, 2927, 2930, 2945, 2953, 2986, 2990,
       2995, 3005, 3060, 3073, 3106, 3110, 3117, 3130, 3136, 3145, 3157,
       3167, 3189, 3206, 3240, 3269, 3284, 3330, 3333, 3336, 3376, 3384,
       3410, 3422, 3503, 3520, 3547, 3549, 3550, 3558, 3565, 3567, 3573,
       3597, 3604, 3629, 3664, 3681, 3702, 3716, 3718, 3751, 3757, 3763,
       3767, 3776, 3780, 3796, 3806, 3808, 3811, 3838, 3846, 3848, 3853,
       3855, 3862, 3869, 3876, 3893, 3902, 3906, 3926, 3941, 3946, 3968,
       3976, 3985, 4000, 4007, 4017, 4063, 4065, 4068, 4072, 4075, 4076,
       4078, 4093, 4131, 4145, 4152, 4154, 4163, 4176, 4180, 4199, 4205,
       4211, 4212, 4224, 4238, 4248, 4256, 4271, 4289, 4300, 4306, 4313,
       4315, 4341, 4355, 4356, 4369, 4374, 4425, 4433, 4435, 4449, 4451,
       4477, 4497, 4498, 4500, 4523, 4536, 4540, 4571, 4575, 4578, 4601,
       4615, 4633, 4639, 4671, 4740, 4751, 4761, 4785, 4807, 4808, 4814,
       4823, 4837, 4863, 4876, 4879, 4880, 4886, 4910, 4939, 4950, 4956,
       4966, 4981, 4990, 4997, 5009, 5068, 5135, 5140, 5210, 5331, 5457,
       5600, 5611, 5642, 5654, 5714, 5734, 5749, 5757, 5835, 5842, 5887,
       5888, 5891, 5912, 5913, 5936, 5937, 5955, 5972, 5973, 5985, 6035,
       6042, 6043, 6045, 6059, 6065, 6071, 6081, 6091, 6093, 6112, 6157,
       6166, 6168, 6172, 6173, 6400, 6421, 6505, 6555, 6560, 6568, 6569,
       6571, 6572, 6574, 6597, 6598, 6603, 6641, 6642, 6651, 6706, 6740,
       6746, 6765, 6783, 6817, 6847, 6906, 6926, 7043, 7094, 7121, 7130,
       7338, 7432, 7434, 7451, 7459, 7492, 7498, 7539, 7580, 7812, 7847,
       7849, 7886, 7899, 7921, 7928, 7945, 7990, 8020, 8062, 8072, 8091,
       8094, 8095, 8183, 8246, 8272, 8277, 8279, 8311, 8332, 8339, 8406,
       8408, 8444, 8520, 8522, 9009, 9013, 9015, 9016, 9019, 9024, 9026,
       9036, 9045, 9245, 9280, 9427, 9465, 9482, 9544, 9587, 9624, 9634,
       9642, 9643, 9662, 9679, 9698, 9719, 9729, 9741, 9744, 9745, 9749,
       9752, 9768, 9770, 9779, 9808, 9811, 9832, 9839, 9856, 9858, 9867,
       9879, 9883, 9888, 9893, 9905, 9941, 9944, 9970, 9980, 9982, 9986]),)'''
print(len(misclassified_indices[0]))  # 495


# ### **Visualización de las imágenes clasificadas incorrectamente por nuestro modelo**


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
    res = str(np.argmax(model.predict(input_im), axis=-1)[0])  # 1/1 [==============================] - 0s 28ms/step

    # Coloque la etiqueta en la imagen de nuestra muestra de datos de prueba
    draw_test("Misclassified Prediction", res,  np.uint8(imageL))  # imprime la imagen mal predicha y la etiqueta real


# ### **Una forma más elegante de trazar esto**
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

plt.subplots_adjust(wspace=0.5)  # lo imprime todo en una sola imagen con subplots


## **3. Creando nuestra Matriz de Confusión**
#
# Usamos la herramienta Confusion Matrix de Sklean para crearlo. Todo lo que necesitamos es:
# 1. Las verdaderas etiquetas
# 2. Las etiquetas predichas


from sklearn.metrics import confusion_matrix
import numpy as np

x_test = x_test.reshape(10000,28,28,1) 
y_pred = np.argmax(model.predict(x_test), axis=-1)

print(confusion_matrix(y_test, y_pred))
'''
[[ 968    0    0    2    0    3    3    2    2    0]
 [   0 1116    3    2    0    0    4    1    9    0]
 [   7    1  969   19    5    0    7    7   16    1]
 [   2    2    6  961    0    7    2    8   14    8]
 [   1    0    4    0  943    0    7    3    2   22]
 [   9    2    0   27    6  805   15    5   14    9]
 [   9    2    2    3    7    9  922    2    2    0]
 [   1    6   16    9    2    0    0  969    4   21]
 [   5    2    1   13    8    7    6    6  919    7]
 [   8    6    1   12   28    1    1   13    6  933]]'''

# #### **Interpretación de la matriz de confusión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2010.46.45.png)
#
# ### **Creando una trama más presentable**
#
# Reutilizaremos esta función muy bien hecha de la documentación de sklearn sobre el trazado de una matriz de confusión
# usando gradientes de color y etiquetas.


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
plot_confusion_matrix(cm = cm,  # matriz de confusión creada por
# sklearn.metrics.confusion_matrix
normalize = True,                # mostrar proporciones
target_names = y_labels_vals,    # lista de nombres de las clases
title = best_estimator_name) # título del gráfico

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

target_names = list(range(0,10))
conf_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat, target_names)


# ## **Veamos nuestra precisión por clase**

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

for (i, ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')

'''Accuracy for 0 : 98.776%
Accuracy for 1 : 98.326%
Accuracy for 2 : 93.895%
Accuracy for 3 : 95.149%
Accuracy for 4 : 96.029%
Accuracy for 5 : 90.247%
Accuracy for 6 : 96.242%
Accuracy for 7 : 94.261%
Accuracy for 8 : 94.353%
Accuracy for 9 : 92.468%'''

# # **4. Ahora veamos el Informe de Clasificación**

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
'''              precision    recall  f1-score   support

           0       0.96      0.99      0.97       980
           1       0.98      0.98      0.98      1135
           2       0.97      0.94      0.95      1032
           3       0.92      0.95      0.93      1010
           4       0.94      0.96      0.95       982
           5       0.97      0.90      0.93       892
           6       0.95      0.96      0.96       958
           7       0.95      0.94      0.95      1028
           8       0.93      0.94      0.94       974
           9       0.93      0.92      0.93      1009

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

Process finished with exit code 0
'''

# ### **4.1 El soporte es la suma total de esa clase en el conjunto de datos**
# ### **4.2 Revisión del retiro del mercado**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.12.png)
#
# ### **4.3 Revisión de la precisión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.22.png)
#
# ### **4.4 Alta recuperación (o sensibilidad) con baja precisión.**
# Esto nos dice que la mayoría de los ejemplos positivos se reconocen correctamente (falsos negativos bajos), pero hay
# falsos positivos, es decir, otras clases se predicen como nuestra clase en cuestión.
#
# ### **4.5 Baja recuperación (o sensibilidad) con alta precisión.**
#
# A nuestro clasificador le faltan muchos ejemplos positivos (FN alto), pero aquellos que predecimos como positivos son
# realmente positivos (Falsos positivos bajos)
#



