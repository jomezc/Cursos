#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Análisis de rendimiento del modelo PyTorch**
# ---
#
#
#
# ---
#
#
#
# En esta lección, aprendemos a usar el modelo MNIST que entrenamos en la lección anterior y analizamos su desempeño, hacemos:
# 1. Configure nuestro modelo y datos de PyTorch
#2. Cargar el modelo previamente entrenado
# 3. Ver las imágenes que clasificamos mal
# 4. Crea una Matriz de Confusión
# 5. Crear informe de clasificación
#

# # **1. Configure nuestras importaciones de PyTorch, modele y cargue el conjunto de datos MNIST**
#
# Solo necesitamos cargar el conjunto de datos de prueba ya que estamos analizando el rendimiento en ese segmento de datos.

# En[ ]:


# Importar PyTorch
import torch

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))

# Establecer dispositivo en cuda
device = 'cuda'


# #### **Nuestra función de trazado de imágenes**

# En[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imgshow(title, image = None, size = 6):
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()


# ### **Cargando nuestro conjunto de datos de prueba MNIST**

# En[ ]:


# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, )) ])

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.MNIST('mnist', 
                                     train = False,
                                     download = True,
                                     transform = transform)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size = 128,
                                          shuffle = False,
                                          num_workers = 0)


# ### **Creando nuestra clase de definición de modelo**

# En[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # **2. Cargando modelo**
#
# He subido el modelo a mi Google Drive - https://drive.google.com/file/d/1yj01iUbYL8ZXHiYRE5Xd639tddSAkzKs/view?usp=sharing
#
# Usamos gdown en nuestra terminal para descargar el archivo modelo que entrenamos en la última lección.
#
# Actualización del 4 de marzo de 2022: archivo movido a S3

# En[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/mnist_cnn_net.pth')


# #### **NOTA**
#
# Al cargar nuestro modelo, necesitamos crear la instancia del modelo, es decir, ```net = Net()``` y luego, dado que lo entrenamos usando nuestra GPU en Colab, lo movemos a la GPU usando ```net.to( dispositivo``` donde dispositivo = 'cuda'.
#
# Entonces podemos cargar los pesos de nuestro modelo descargado.

# En[ ]:


# Crear una instancia del modelo
net = Net()
net.to(device)

# Cargar pesos desde la ruta especificada
net.load_state_dict(torch.load('models/mnist_cnn_net.pth'))


# Modelo cargado con éxito si se muestra ```Todas las claves coincidieron correctamente```.
#
# ### **Ahora calculemos su precisión (hecho en la lección anterior, así que esto es solo un resumen) en los datos de prueba**
#
#

# En[ ]:


correct = 0 
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.3}%')


## **3. Mostrando nuestras imágenes mal clasificadas** ##
#
# De 10.000 imágenes, nuestro modelo predijo que el 98,7 % era correcto. Esto es bueno para un primer intento con un modelo tan simple. (hay modelos mucho mejores).
#
# **¡Una buena práctica!**
#
# Es un buen hábito al crear clasificadores de imágenes inspeccionar visualmente las imágenes que están mal clasificadas.
# 1. Podemos detectar qué tipos de imágenes son un desafío para nuestro modelo
# 2. Podemos detectar cualquier imagen etiquetada incorrectamente
#3. Si a veces no podemos identificar correctamente la clase, ver tu lucha en CNN duele menos :)
#
# **Recordatorio** de por qué usamos ```net.eval()``` y ```torch.no_grad()```
#
# [Tomado de Stackoverflow:](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
#
# **model.eval()** es una especie de interruptor para algunas capas/partes específicas del modelo que se comportan de manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, capas de **abandonos**, capas BatchNorm, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes.
#
# Entonces, aunque no usamos Dropouts o BatchNorm en nuestro modelo, es una buena práctica usarlo al hacer inferencias.

# En[ ]:


# Establecer el modelo en modo de evaluación o inferencia
net.eval()

# No necesitamos gradientes para la validación, así que envuélvalos
# no_grad para ahorrar memoria
with torch.no_grad():
    for data in testloader:
        images, labels = data

        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)

        # Obtener nuestros resultados
        outputs = net(images)

        # use torch.argmax() para obtener las predicciones, argmax se usa para long_tensors
        predictions = torch.argmax(outputs, dim=1)

        # Para los datos de prueba en cada lote, identificamos qué predicciones no coincidieron con la etiqueta
        # luego imprimimos la verdad del terreno real
        for i in range(data[0].shape[0]):
            pred = predictions[i].item()
            label = labels[i]
            if(label != pred):
                print(f'Actual Label: {label}, Predicted Label: {pred}')       
                img = np.reshape(images[i].cpu().numpy(),[28,28])
                imgshow("", np.uint8(img), size = 1)


# # **4. Creando nuestra Matriz de Confusión**
#
# Usamos la herramienta Confusion Matrix de Sklean para crearlo. Todo lo que necesitamos es:
# 1. Las verdaderas etiquetas
# 2. Las etiquetas predichas
#

# En[ ]:


from sklearn.metrics import confusion_matrix


# Inicializar tensores en blanco para almacenar nuestras predicciones y listas de etiquetas (tensores)
pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
label_list = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes
        pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
        label_list = torch.cat([label_list, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
print(conf_mat)


# #### **Interpretación de la matriz de confusión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2010.46.45.png)

# ### **Creando una trama más presentable**
#
# Reutilizaremos esta función muy bien hecha de la documentación de sklearn sobre el trazado de una matriz de confusión usando gradientes de color y etiquetas.

# En[ ]:


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


# En[ ]:


target_names = list(range(0,10))
plot_confusion_matrix(conf_mat, target_names)


# ## **Veamos nuestra precisión por clase**

# En[ ]:


# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

for (i,ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')


# # **5. Ahora veamos el Informe de Clasificación**

# En[ ]:


from sklearn.metrics import classification_report

print(classification_report(label_list.numpy(), pred_list.numpy()))


# ### **5.1 El soporte es la suma total de esa clase en el conjunto de datos**
#
# ### **5.2 Revisión del retiro del mercado**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.12.png)
#
# ### **5.3 Revisión de la precisión**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.22.png)
#
# ### **5.4 Alta recuperación (o sensibilidad) con baja precisión.**
# Esto nos dice que la mayoría de los ejemplos positivos se reconocen correctamente (falsos negativos bajos), pero hay muchos falsos positivos, es decir, otras clases se predicen como nuestra clase en cuestión.
#
# ### **5.5 Baja recuperación (o sensibilidad) con alta precisión.**
#
# A nuestro clasificador le faltan muchos ejemplos positivos (FN alto), pero aquellos que predecimos como positivos son realmente positivos (Falsos positivos bajos)
#
