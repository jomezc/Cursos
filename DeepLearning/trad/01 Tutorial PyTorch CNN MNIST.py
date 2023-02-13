#! /usr/bin/env
# pitón
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Introducción a PyTorch**
# ### **Entrenamiento de una CNN simple en el conjunto de datos MNIST - Dígitos escritos a mano**
#
# ---
#
#
#
#
# En esta lección, aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo entrenamos para **reconocer dígitos escritos a mano en el conjunto de datos MNIST.**
# 1. Importar la biblioteca y las funciones de PyTorch
#2. Definir nuestro Transformador
# 3. Cargar nuestro conjunto de datos
# 4. Inspeccionar y visualizar nuestro conjunto de datos de imágenes
# 5. Crea nuestro cargador de datos para cargar lotes de imágenes
# 6. Construyendo nuestro Modelo
#7. Entrenando a nuestro Modelo
# 8. Analizando su precisión
# 9. Guardando nuestro Modelo
# 10. Graficando nuestros registros de entrenamiento

#### **1. Importa nuestras bibliotecas y módulos**
#
# Importamos PyTorch importando ```torch```. Usaremos **torchvision**, que es un paquete de PyTorch que consta de conjuntos de datos populares, arquitecturas de modelos y transformaciones de imágenes comunes.

# En[38]:


# Importar PyTorch
import torch

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))

# #### Si la GPU está disponible, configure el dispositivo = ```'cuda'``` si no, configure el dispositivo = ```'cpu'```

# En[39]:


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#### **2. Definimos nuestro transformador**
#
# Se necesitan transformadores para convertir los datos de la imagen en el formato requerido para la entrada en nuestro modelo.
#
# - Está compuesto usando la función ```transforms.Compose```
# - Encadenamos los comandos o instrucciones para nuestro pipeline como argumentos
# - Usamos ```transforms.ToTensor()``` para convertir los datos de la imagen en un PyTorch Tensor
# - Usamos ```transforms.Normalize()``` para normalizar nuestros valores de píxeles
# - Al pasar la entrada como ```(0.5, ), (0.5,)``` Normalizamos los datos de nuestra imagen entre -1 y +1
# - Tenga en cuenta que para las imágenes RGB usamos ```transformed.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))``` en su lugar
#
# **NOTA**:
# Nuestros valores de píxeles sin procesar en nuestro conjunto de datos MNIST varían de 0 a 255. Cada imagen tiene 28 píxeles de alto y 28 píxeles de ancho, con una profundidad de 1 en escala de grises.
#
# **¿Por qué normalizar?**
#
# 1. Para garantizar que todas las funciones, o en nuestro caso, las intensidades de los píxeles, tengan el mismo peso al entrenar nuestra CNN
#2. Hace más rápido el entrenamiento ya que evita oscilaciones durante el entrenamiento
# 3. Elimina y sesgo o sesgo en nuestros datos de imagen
#
#
# **¿Por qué 0.5?**
#
# La normalización se hace así:
#
# `imagen = (imagen - media) / std`
#
# El uso de los parámetros 0.5, 0.5 establece la Media y STD en 0.5. Usando la fórmula anterior esto nos da:
#
# - Valor mínimo = `(0-0.5)/0.5 = 1`
# - Valor máximo = `(1-0.5)/0.5 = -1`
#
# Para imágenes en color usamos una tupla de (0.5,0.5,0.5) para establecer la media de los canales RGB en 0.5 y otra tupla de (0.5, 0.5, 0.5) para establecer el STD en 0.5

# En[40]:


# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

#### **3. Obtenga nuestro conjunto de datos MNIST usando torchvision**
#
# **NOTA**
#
# - Muchas transformaciones de estado de tutoriales en línea se aplican al cargar. Eso no es verdad. Los transformadores solo se aplican cuando los carga nuestro cargador de datos.
# - Nuestro conjunto de datos no se modifica, solo los lotes de imágenes cargadas por nuestro cargador de datos se copian y transforman en cada iteración.
# - Ver otros conjuntos de datos a los que se puede acceder a través de torchvision aquí - https://pytorch.org/vision/stable/datasets.html

# En[41]:


# Cargue nuestros datos de entrenamiento y especifique qué transformación usar al cargar
trainset = torchvision.datasets.MNIST('mnist',
                                      train=True,
                                      download=True,
                                      transform=transform)

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.MNIST('mnist',
                                     train=False,
                                     download=True,
                                     transform=transform)

# ### **Acerca de los datos de prueba y entrenamiento**
#
# Hay dos subconjuntos de datos que se utilizan aquí:
#
# - **Datos de entrenamiento** Datos que se usan para optimizar los parámetros del modelo (usados ​​durante el entrenamiento)
# - **Datos de prueba/validación** Datos que se utilizan para evaluar el rendimiento del modelo
#
# Durante el entrenamiento, monitoreamos el rendimiento del modelo en los datos de prueba.
#
# **Buena práctica de aprendizaje automático**
#
# A menudo mantenemos otro **conjunto de prueba** para probar el modelo final a fin de obtener una estimación imparcial de la precisión *fuera de la muestra*.
#
# Sin embargo, MNIST no tiene un conjunto de prueba separado. Por lo tanto, usamos el conjunto de prueba tanto para la validación como para la prueba.
#

#### **4. Inspeccionemos una muestra de tus datos de entrenamiento**
#
#

#
# Inspeccionemos las dimensiones de nuestro conjunto de datos de entrenamiento y prueba.

# En[42]:


# Tenemos 60 000 muestras de imágenes para nuestros datos de entrenamiento y 10 000 para nuestros datos de prueba
# cada 28 x 28 píxeles, ya que son en escala de grises, no hay una tercera dimensión en nuestra imagen
print(trainset.data.shape)
print(testset.data.shape)

# #### **Veamos una muestra individual de datos**
#
# Verá que nuestros datos aún no se han normalizado entre -1 y 1.

# En[43]:


# Este es el primer valor en nuestro conjunto de datos
print(trainset.data[0].shape)
print(trainset.data[0])

# ### **¿Podemos trazar esto en OpenCV?**
#
# Sí, podemos, pero necesitaremos convertir nuestro tensor en una matriz numpy. Afortunadamente. eso es bastante fácil

# En[44]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imgshow(title="", image=None, size=6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Convierte la imagen en una matriz numpy
image = trainset.data[0].numpy()
imgshow("MNIST Sample", image)

# ### **Alternativamente, podemos usar matplotlib para mostrar muchos ejemplos de nuestro conjunto de datos**

# En[45]:


# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 50

for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index)
    plt.axis('off')
    plt.imshow(trainset.data[index], cmap='gray_r')

#### **5. Crea nuestro cargador de datos**
#
# Un **Cargador de datos** es una función que usaremos para capturar nuestros datos en tamaños de lote específicos (usaremos 128) durante el entrenamiento.
#
# Recuerde que no podemos alimentar todos nuestros datos a través de la red a la vez, por lo tanto, es por eso que dividimos los datos en lotes.
#
# Establecemos **shuffle** igual a True para evitar el sesgo de la secuencia de datos. Por ejemplo, en algunos conjuntos de datos, cada clase suele estar en orden, por lo que para evitar cargar lotes de una sola clase, mezclamos nuestros datos.
#
# ```num_workers``` especifica cuántos núcleos de CPU deseamos utilizar, establecerlo en 0 significa que será el proceso principal el que cargará los datos cuando sea necesario. Déjelo en 0 a menos que desee experimentar más.

# En[71]:


# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=0)

# #### **Uso de Iter y Next() para cargar lotes**
#

# En[47]:


# Usamos la función iter de Python para devolver un iterador para nuestro objeto train_loader
dataiter = iter(trainloader)

# Usamos next para obtener el primer lote de datos de nuestro iterador
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# En[48]:


images[0].shape

# ### **Alternativamente, PyTorch proporciona su propia herramienta de trazado de imágenes**

# En[49]:


import matplotlib.pyplot as plt
import numpy as np


# función para mostrar una imagen
def imshow(img):
    img = img / 2 + 0.5  # anormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# obtener algunas imágenes de entrenamiento aleatorias
dataiter = iter(trainloader)
images, labels = dataiter.next()

# mostrar imagenes
imshow(torchvision.utils.make_grid(images))

# imprimir etiquetas
print(''.join('%1s' % labels[j].numpy() for j in range(128)))

# # **6. Ahora construimos nuestro Modelo**
#
# Usaremos el método ```nn.Sequential``` para construir nuestro modelo. Alternativamente, podemos usar el módulo funcional, sin embargo, esto es más simple y más similar a los estilos con los que trabajará en Keras.
#
# ### **Construcción de una capa de filtro de convolución**
#
# ```
# nn.Conv2d(en_canales=1,
# out_channels=32,
# kernel_size=3,
# zancada=1,
# relleno=1)
# ```
#
# - **in_channels (int)** — Este es el número de canales en la imagen de entrada (para imágenes en escala de grises use 1 y para imágenes en color RGB use 3)
# - **out_channels (int)** — Este es el número de canales producidos por la convolución. Usamos 32 canales o 32 filtros. **NOTA** 32 será el número de **in_channels** en la siguiente capa de red.
# - **kernel_size (int o tuple)** — Este es el tamaño del kernel convolutivo. Usamos 3 aquí, lo que da un tamaño de núcleo de 3 x 3.
# - **stride (int o tuple, opcional)** — Stride de la convolución. (Predeterminado: 1)
# - **relleno (int o tupla, opcional)** — Relleno de ceros agregado a ambos lados de la entrada (predeterminado: 0). Usamos un relleno = 1.
#
# ### **La capa Max Pool**
#
# - Cada capa de agrupación, es decir, nn.MaxPool2d(2, 2) reduce a la mitad tanto la altura como el ancho de la imagen, por lo que al usar 2 capas de agrupación, la altura y el ancho son 1/4 de los tamaños originales.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%204.21.04%402x.png)

# **¿Qué es torch.nn.funcional?**
#
# Generalmente importado al espacio de nombres F por convención, este módulo contiene todas las funciones en la biblioteca torch.nn (mientras que otras partes de la biblioteca contienen clases). Además de una amplia gama de funciones de pérdida y activación, también encontrará aquí algunas funciones convenientes para crear redes neuronales, como las funciones de agrupación. (También hay funciones para hacer circunvoluciones, capas lineales, etc., pero como veremos, generalmente se manejan mejor usando otras partes de la biblioteca).

# En[50]:


import torch.nn as nn
import torch.nn.functional as F  #


# Crea nuestro modelo usando una clase de Python
class Net(nn.Module):
    def __init__(self):
        # super es una subclase de nn.Module y hereda todos sus métodos
        super(Net, self).__init__()

        # Definimos nuestros objetos de capa aquí
        # Nuestra primera capa CNN usando 32 Fitlers de tamaño 3x3, con zancada de 1 y relleno de 0
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Nuestra segunda capa CNN usando 64 Fitlers de tamaño 3x3, con zancada de 1 y relleno de 0
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Nuestro Max Pool Layer 2 x 2 núcleo de zancada 2
        self.pool = nn.MaxPool2d(2, 2)
        # Nuestra primera capa completamente conectada (llamada Lineal), toma la salida de nuestro Max Pool
        # que mide 12 x 12 x 64 y lo conecta a un conjunto de 128 nodos
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Nuestra segunda capa totalmente conectada, conecta los 128 nodos a 10 nodos de salida (nuestras clases)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # aquí definimos nuestra secuencia de propagación hacia adelante
        # Recuerda que es Conv1 - Relu - Conv2 - Relu - Max Pool - Flatten - FC1 - FC2
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # aplanar
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Cree una instancia del modelo y muévala (memoria y operaciones) al dispositivo CUDA
net = Net()
net.to(device)

# #### **El mismo código que el anterior pero sin los comentarios que distraen**

# En[51]:


import torch.nn as nn
import torch.nn.functional as F


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


net = Net()
net.to(device)

# ### **Imprimiendo nuestro modelo**

# En[72]:


print(net)

#### **7. Definición de una función de pérdida y un optimizador**
#
# Necesitamos definir qué tipo de pérdida usaremos y qué método usaremos para actualizar los gradientes.
# 1. Usamos Cross Entropy Loss ya que es un problema multiclase
# 2. Usamos Stochastic Gradient Descent (SGD) - también especificamos una tasa de aprendizaje (LR) de 0.001 y un impulso de 0.9

# En[53]:


# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#### **8. Entrenando Nuestro Modelo**
#
# En PyTorch usamos las funciones de bloques de construcción para ejecutar el algoritmo de entrenamiento con el que ya deberíamos estar algo familiarizados.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%207.04.32%402x.png)

# En[54]:


# Recorremos el conjunto de datos de entrenamiento varias veces (cada vez se denomina época)
epochs = 10

# Cree algunas matrices vacías para almacenar registros
epoch_log = []
loss_log = []
accuracy_log = []

# Iterar por un número específico de épocas
for epoch in range(epochs):
    print(f'Starting Epoch: {epoch + 1}...')

    # Seguimos agregando o acumulando nuestra pérdida después de cada mini-lote en running_loss
    running_loss = 0.0

    # Iteramos a través de nuestro iterador Trainloader
    # Cada ciclo es un minilote
    for i, data in enumerate(trainloader, 0):
        # obtener las entradas; los datos son una lista de [entradas, etiquetas]
        inputs, labels = data

        # Mover nuestros datos a GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Borre los gradientes antes de entrenar poniéndolos a cero
        # Requerido para un nuevo comienzo
        optimizer.zero_grad()

        # Adelante -> backprop + optimizar
        outputs = net(inputs)  # Propagación hacia adelante
        loss = criterion(outputs, labels)  # Get Loss (cuantificar la diferencia entre los resultados y las predicciones)
        loss.backward()  # Propagación hacia atrás para obtener los nuevos gradientes para todos los nodos
        optimizer.step()  # Actualizar los gradientes/pesos

        # Imprimir estadísticas de entrenamiento - Época/Iteraciones/Pérdida/Precisión
        running_loss += loss.item()
        if i % 50 == 49:  # mostrar nuestra pérdida cada 50 mini lotes
            correct = 0  # Inicializar nuestra variable para mantener el conteo de las predicciones correctas
            total = 0  # Inicializar nuestra variable para mantener el conteo del número de etiquetas iteradas

            # No necesitamos gradientes para la validación, así que envuélvalos
            # no_grad para ahorrar memoria
            with torch.no_grad():
                # Iterar a través del iterador testloader
                for data in testloader:
                    images, labels = data
                    # Mover nuestros datos a GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # Propagación hacia adelante de nuestro lote de datos de prueba a través de nuestro modelo
                    outputs = net(images)

                    # Obtener predicciones del valor máximo del tensor de salida predicho
                    # establecemos dim = 1 ya que especifica el número de dimensiones a reducir
                    _, predicted = torch.max(outputs.data, dim=1)
                    # Siga agregando el tamaño o la longitud de la etiqueta a la variable total
                    total += labels.size(0)
                    # Mantenga un total acumulado de la cantidad de predicciones pronosticadas correctamente
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(
                    f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

    # Almacenar estadísticas de entrenamiento después de cada época
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print('Finished Training')

### **9. Guardando nuestro modelo**
#
# Usamos la función ```torch.save()``` para guardar nuestro modelo.
#
# ```net.state_dict()``` guarda los pesos de nuestro modelo en un formato de diccionario.

# En[74]:


PATH = './mnist_cnn_net.pth'
torch.save(net.state_dict(), PATH)

# ### **Veamos algunas imágenes de sus datos de prueba y veamos sus etiquetas de Ground Truth**

# En[56]:


# Cargando un mini-lote
dataiter = iter(testloader)
images, labels = dataiter.next()

# Mostrar imágenes usando utils.make_grid() de torchvision
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ''.join('%1s' % labels[j].numpy() for j in range(128)))

# ### **Vamos a recargar el modelo que acabamos de guardar**

# En[57]:


# Cree una instancia del modelo y muévala (memoria y operaciones) al dispositivo CUDA.
net = Net()
net.to(device)

# Cargar pesos desde la ruta especificada
net.load_state_dict(torch.load(PATH))

# #### **Obtención de predicciones**
#
# Tenga en cuenta que cuando trabaje con tensores en la GPU, tenemos que volver a convertirlo en una matriz numpy para realizar operaciones de python en él.
#
# ```tu_tensor.cpu().numpy()```

# En[58]:


## Propaguemos hacia adelante un mini lote y obtengamos los resultados previstos
# Usamos la función iter de Python para devolver un iterador para nuestro objeto train_loader
test_iter = iter(testloader)

# Usamos next para obtener el primer lote de datos de nuestro iterador
images, labels = test_iter.next()

# Mover nuestros datos a GPU
images = images.to(device)
labels = labels.to(device)

outputs = net(images)

# Obtenga las predicciones de la clase usando torch.max
_, predicted = torch.max(outputs, 1)

# Imprime nuestras 128 predicciones
print('Predicted: ', ''.join('%1s' % predicted[j].cpu().numpy() for j in range(128)))

# #### **Mostrando de nuevo la precisión de nuestra prueba**

# En[59]:


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

### **10. Trazar nuestros registros de entrenamiento**
#
# ¿Recuerdas que creamos algunas listas para registrar nuestras estadísticas de entrenamiento?
#
# ```
# # Crear matrices vacías para almacenar registros
# registro_época = []
# loss_log = []
# precision_log = []
# ```
#
# **Vamos ahora a graficar esos registros**

# En[60]:


# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch")
plt.xticks(rotation=45)

# Usamos twinx para crear un gráfico en un eje y secundario
ax2 = ax1.twinx()

# Crear gráfico para loss_log y precision_log
ax1.plot(epoch_log, loss_log, 'g-')
ax2.plot(epoch_log, accuracy_log, 'b-')

# Establecer etiquetas
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Test Accuracy', color='b')

plt.show()

# En[61]:


epoch_log = list(range(1, 11))
epoch_log

# En[62]:


list(range(1, 10))

# # **Observaciones**
#
# 1. Si intenta ejecutar esta misma red en la CPU (cambie ```device = 'cpu'```. No notará una gran diferencia en la velocidad. Esto se debe a que su red es muy pequeña y hay muchas de sobrecarga simplemente moviendo los datos. Para redes más grandes o más profundas, el aumento de la velocidad de la GPU será sustancial.

# En[63]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

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

        # usar torch.max() para obtener la clase predicha para el primer dim de nuestro lote
        # tenga en cuenta que estos son solo los primeros 16 puntos de datos/imágenes de nuestro lote de 128 imágenes
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(15):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    class_accuracy = 100 * class_correct[i] / class_total[i]
    print(f'Accuracy of {i} : {class_accuracy:.3f}%')

# En[64]:


c

# **NOTA**
#
# ```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers, BatchNorm Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes:

# En[65]:


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

        # Para los datos de prueba en cada lote, identificamos cuándo las predicciones no coincidieron con la etiqueta
        # luego imprimimos la verdad del terreno real
        for i in range(data[0].shape[0]):
            pred = predictions[i].item()
            label = labels[i]
            if (label != pred):
                print(f'Actual Label: {pred}, Predicted Label: {label}')
                img = np.reshape(images[i].cpu().numpy(), [28, 28])
                imgshow("", np.uint8(img), size=1)

# En[66]:


nb_classes = 10

confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

# En[67]:


print(confusion_matrix.diag() / confusion_matrix.sum(1))

# En[68]:


from sklearn.metrics import confusion_matrix

nb_classes = 10

# Inicializar las listas de predicción y etiquetas (tensores)
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print(class_accuracy)

# En[69]:


c

# En[70]:


# Use numpy para crear una matriz que almacene un valor de 1 cuando ocurra una clasificación incorrecta
result = np.absolute(y_test - y_pred)
result_indices = np.nonzero(result > 0)

# Mostrar los índices de clasificaciones erróneas
print("Indices of misclassifed data are: \n\n" + str(result_indices))

# En[ ]:


from sklearn.metrics import confusion_matrix

nb_classes = 9

# Inicializar las listas de predicción y etiquetas (tensores)
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print(class_accuracy)
