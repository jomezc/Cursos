#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Regularización en PyTorch - Parte 1**
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
# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, sin el uso de ningún método de regularización.
# 1. Importe bibliotecas de PyTorch, defina nuestros transformadores, cargue nuestro conjunto de datos y visualice nuestras imágenes.
# 2. Construya una CNN simple sin regularización
# 3. Capacitar a nuestra CNN
# 3. Eche un vistazo al aumento de datos
#
#

# # **Importar bibliotecas de PyTorch, definir transformadores y cargar y visualizar conjuntos de datos**

# En[17]:


# Importar PyTorch
import torch
import PIL

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))
device = 'cuda' # 'cpu' si no hay GPU disponible


# ### **Nuestra transformación de datos**

# En[18]:


# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, )) ])


# En[ ]:


# Cargue nuestros datos de entrenamiento y especifique qué transformación usar al cargar
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)


# En 19]:


# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Crear una lista con los nombres de nuestras clases
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


# En 20]:


# Tenemos 60 000 muestras de imágenes para nuestros datos de entrenamiento y 10 000 para nuestros datos de prueba
# cada 28 x 28 píxeles, ya que son en escala de grises, no hay una tercera dimensión en nuestra imagen
print(trainset.data.shape)
print(testset.data.shape)


# ### **Visualización de nuestros datos**

# En[21]:


# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 50 

for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index)
    plt.axis('off')
    plt.imshow(trainset.data[index], cmap='gray_r')
  


# En[22]:


import matplotlib.pyplot as plt
import numpy as np

# función para mostrar una imagen
def imshow(img):
    img = img / 2 + 0.5     # anormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# obtener algunas imágenes de entrenamiento aleatorias
dataiter = iter(trainloader)
images, labels = dataiter.next()

# mostrar imagenes
imshow(torchvision.utils.make_grid(images))

# imprimir etiquetas
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))


# # **2. Construyendo y entrenando nuestra CNN simple sin regularización**
#
# #### **Definiendo nuestro modelo**

# En[23]:


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


# #### **Definición de nuestras funciones de pérdida y optimización**

# En[24]:


# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


## **3. Entrenando Nuestro Modelo**

# En[25]:


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


# #### **Precisión de nuestro modelo**

# En[27]:


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


# #### **Nuestras Parcelas de Entrenamiento**

# En[28]:


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


# #### **Guardando los pesos de nuestro modelo**

# En[29]:


PATH = './fashion_mnist_cnn_net.pth'
torch.save(net.state_dict(), PATH)


# # **4. Aumento de datos**
#
# Para introducir el aumento de datos en nuestros datos de entrenamiento, simplemente creamos nuevas funciones de transformación.
#
# **Recuerda nuestra función de transformación anterior**
#
# ```transform = transforma.Compose([transforma.ToTensor(),
# transforma.Normalizar((0.5, ), (0.5, )) ])```
#
# ### **Primero vamos a demostrar cómo el aumento de datos afecta nuestras imágenes**

# En[30]:


# Importamos PIL, una biblioteca de procesamiento de imágenes para implementar rotaciones aleatorias
import PIL

data_aug_transform = transforms.Compose([
        transforms.RandomAffine(degrees = 10, translate = (0.05,0.05), shear = 5), 
        transforms.ColorJitter(hue = .05, saturation = .05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, resample = PIL.Image.BILINEAR),
        transforms.Grayscale(num_output_channels = 1)
])


# #### **Realice el aumento de datos en una sola imagen usando la función a continuación para visualizar los efectos**

# En[33]:


from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

def showAugmentations(image, augmentations = 6):
    fig = figure()
    for i in range(augmentations):
        a = fig.add_subplot(1,augmentations,i+1)
        img = data_aug_transform(image)
        imshow(img ,cmap='Greys_r')
        axis('off')


# En[34]:


# Cargue la primera imagen de nuestros datos de entrenamiento como una matriz numpy
image = trainset.data[0].numpy()

# Convertirlo a formato de imagen PIL
img_pil = PIL.Image.fromarray(image)

showAugmentations(img_pil, 8)


# En[ ]:




