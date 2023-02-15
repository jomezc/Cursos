#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Regularización en PyTorch - Parte 2**
# ### **Ahora usamos algunos métodos de regularización en nuestro Fashion-MNIST CNN**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)
# ---
#
#
#
# ---
#
#
# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, ahora **CON** el uso de cualquier regularización métodos.
# 1. Importe bibliotecas de PyTorch, defina nuestros transformadores, cargue nuestro conjunto de datos y visualice nuestras imágenes.
# 2. Cree una CNN simple con los siguientes métodos de **regularización**:
# - Regularización L2
# - Aumento de datos
#   - Abandonar
# - Norma de lote
#3. Capacitar a nuestra CNN con Regularización
#

# # **1. Importe bibliotecas de PyTorch, defina transformadores y cargue y visualice conjuntos de datos**

# En[ ]:


# Importar PyTorch
import torch
import PIL
import numpy as np

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))


# En[ ]:


device = 'cuda' # 'cpu' si no hay GPU disponible


# # **2. Construyendo una CNN con Regulación**
# ## **Implementación del aumento de datos**
#
# #### **Comprender lo que hacen nuestras transformaciones**
#
# 1. Nuestras transformaciones se aplican a una imagen o lote de imágenes cada vez que se carga.
# 2. Estas nuevas imágenes no se guardan, se generan o se 'alteran' cada vez que se carga un lote
#
# #### **NOTA**
#
# No aplicamos los mismos aumentos a nuestros conjuntos de datos de prueba o validación. Por lo tanto, mantenemos funciones de transformación separadas (ver más abajo) para nuestros datos de Entrenamiento y Validación/Prueba.

# En[ ]:


data_transforms = {
    'train': transforms.Compose([
        # Tenga en cuenta que estos se ejecutan en el orden en que se llaman aquí
        # Algunas de estas transformaciones devuelven una imagen en color, por lo que necesitamos convertir
        # la imagen vuelve a escala de grises
        transforms.RandomAffine(degrees = 10, translate = (0.05,0.05), shear = 5), 
        transforms.ColorJitter(hue = .05, saturation = .05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, resample = PIL.Image.BILINEAR),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
}


# ### **Obtener y crear nuestros cargadores de datos**

# En[ ]:


# Cargue nuestros datos de tren y especifique qué transformación usar al cargar
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=data_transforms['train'])

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=data_transforms['val'])

# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


# ### **Agregando Abandono**
#
# En las redes neuronales convolucionales, el abandono se agrega comúnmente después de las capas CONV-RELU.
#
# P.ej. CONV->RELU->**CAÍDA**
#
# #### **Recomendaciones de abandono**
#
# En los valores de CNN de 0.1 a 0.3 se ha encontrado que funcionan bien.
#
# ### **Añadiendo BatchNorm**
#
# En las CNN, **BatchNorm** se usa mejor entre la capa de conversión y la capa de función de activación (ReLU)
# Cuando se usa con Dropout, el orden recomendado es:
#
# CONV_1 -> **BatchNorm** -> ReLU -> Abandono - CONV_2
#
# **NOTA** El argumento de entrada de BatchNorm es el tamaño de **salida** de la capa anterior.

# En[ ]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Agregando BatchNorm, usando 32 como entrada ya que 32 fue la salida de nuestra primera capa Conv
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Agregando BatchNorm, usando 64 como entrada ya que 64 fue la salida de nuestra primera capa Conv
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        # Definiendo nuestra función de abandono con una tasa de 0.2
        # Podemos aplicar esto después de cualquier capa, pero es más adecuado después de ReLU
        self.dropOut = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropOut(x)
        x = self.dropOut(F.relu(self.conv2_bn(self.conv2(x))))

        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to(device)


# ### **Añadir regularización L2**
#
#
# La regularización L2 sobre los parámetros/pesos del modelo se incluye directamente en la mayoría de los optimizadores, incluido optim.SGD.
#
# Se puede controlar con el parámetro **weight_decay** como se puede ver en la [documentación SGD] (http://pytorch.org/docs/optim.html#torch.optim.SGD).
#
# ```weight_decay``` (**flotante**, opcional) – disminución del peso *(penalización L2) (predeterminado: 0)*
#
# **Buenos valores de L2 oscilan entre 0,1 y 0,0001**
#
# **NOTA:**
#
# La regularización L1 no está incluida por defecto en los optimizadores, pero podría añadirse incluyendo un extra loss nn.L1Loss en los pesos del modelo.
#
#

# En[ ]:


# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)


## **3. Entrenamiento de nuestro modelo utilizando métodos de regulación: aumento de datos, abandono, BatchNorm y regularización L2**

# En[ ]:


# Recorremos el conjunto de datos de entrenamiento varias veces (cada vez se denomina época)
epochs = 15

# Cree algunas matrices vacías para almacenar registros
epoch_log = []
loss_log = []
accuracy_log = []

# Iterar por un número específico de épocas
for epoch in range(epochs):  
    print(f'Starting Epoch: {epoch+1}...')

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
        outputs = net(inputs) # Propagación hacia adelante
        loss = criterion(outputs, labels) # Get Loss (cuantificar la diferencia entre los resultados y las predicciones)
        loss.backward() # Propagación hacia atrás para obtener los nuevos gradientes para todos los nodos
        optimizer.step() # Actualizar los gradientes/pesos

        # Imprimir estadísticas de entrenamiento - Época/Iteraciones/Pérdida/Precisión
        running_loss += loss.item()
        if i % 100 == 99:    # mostrar nuestra pérdida cada 50 mini lotes
            correct = 0 # Inicializar nuestra variable para mantener el conteo de las predicciones correctas
            total = 0 # Inicializar nuestra variable para mantener el conteo del número de etiquetas iteradas

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

                     # Obtenga predicciones del valor máximo
                    _, predicted = torch.max(outputs.data, 1)
                    # Siga agregando el tamaño o la longitud de la etiqueta a la variable total
                    total += labels.size(0)
                    # Mantenga un total acumulado de la cantidad de predicciones pronosticadas correctamente
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(f'Epoch: {epoch_num}, Mini-Batches Completed: {(i+1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

    # Almacenar estadísticas de entrenamiento después de cada época
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print('Finished Training')


# ### **Precisión de nuestros modelos**

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
print(f'Accuracy of the network on the 10000 test images: {accuracy:.4}%')


# ### **Parcelas de entrenamiento**

# En[ ]:


import matplotlib.pyplot as plt

# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch Mini-Batches")
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


# #### **Detención anticipada en PyTorch**
#
# https://github.com/Bjarten/early-stopping-pytorch
