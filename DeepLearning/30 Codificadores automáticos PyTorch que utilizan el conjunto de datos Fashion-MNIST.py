#!/usr/bin/env python
# codificación: utf-8

# # **Codificadores automáticos de PyTorch que utilizan el conjunto de datos Fashion-MNIST**
#
# ---
#
#
# En esta lección, implementaremos un **Codificador automático en el conjunto de datos MNIST de moda** usando PyTorch
#
#
# Un codificador automático es un algoritmo de aprendizaje automático no supervisado. En nuestro ejemplo, toma una
# imagen como entrada y luego intenta reconstruir esa imagen usando menos información.
#
# Hacen esto proyectando datos dimensionales más altos a una dimensión más baja (similar al Análisis de Componentes
# Principales) manteniendo las características de mayor importancia. Esto se llama el espacio latente.
#
# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522830223/AutoEncoder_kfqad1.png)
#
# ### **Un Autoencoder se compone de estas dos Redes**
#
# **Codificador**: comprime/reduce la muestra de la imagen de entrada en un número menor de bits. Este menor número de
# bits se denomina espacio latente o cuello de botella.
#
# **Decodificador**: intenta reconstruir la entrada usando solo la codificación de la entrada. Si el decodificador
# puede reconstruir la imagen con precisión a partir de la salida del codificador, tiene un codificador que funciona
# correctamente (capaz de producir buenas codificaciones) y un sistema decodificador.
#
# **Crédito del tutorial:**
#
# https://debuggercafe.com/implementing-deep-autoencoder-in-pytorch/

### **1. Cargue y preprocese nuestro conjunto de datos F-MNIST**
#


import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


### **2. Cargue nuestros datos, cree sus transformaciones, defina nuestras constantes y haga nuestros cargadores de#
# datos**

# constantes
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# transformaciones de imagen
transform = transforms.Compose([transforms.ToTensor(),])


# Cargue nuestro conjunto de datos FashionMNIST
trainset = datasets.FashionMNIST(
    root='./data',
    train=True, 
    download=True,
    transform=transform)

testset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform)


# Crear nuestros cargadores de datos
trainloader = DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True)

testloader = DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=True)


### **3. Crea nuestro modelo de autocodificador**
#
# **En primer lugar, algunas funciones de utilidad**


# Primero haga algunas funciones de utilidad
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def make_dir():
    image_dir = 'images/FashionMNIST_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, './FashionMNIST_Images/linear_ae_image{}.png'.format(epoch))


# ### **Crear nuestra clase modelo**


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # codificador
        # nn.Linear recomendado para imágenes pequeñas en escala de grises, grandes a color CNN
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decodificador
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

net = Autoencoder()
print(net)


# ### **Definir nuestra función de pérdida y optimizador**



criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


### **4. Entrena a nuestro modelo**
#
# #### **Definir nuestras funciones de Entrenamiento y Prueba**


def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))

        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)

    return train_loss

def test_image_reconstruction(net, testloader):
     for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'fashionmnist_reconstruction.png')
        break


# ### **Entrenando nuestro Modelo**

#

# obtener el dispositivo de cómputo
device = get_device()
print(device)
# cargar la red neuronal en el dispositivo
net.to(device)

make_dir()

# entrenar la red
train_loss = train(net, trainloader, NUM_EPOCHS)

# Trazar resultados de entrenamiento
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('deep_ae_fashionmnist_loss.png')

# probar la red
test_image_reconstruction(net, testloader)


### **5. Mostrar nuestros resultados**
#
# Ver nuestras imágenes reconstruidas.


from IPython.display import Image

Image('fashionmnist_reconstruction.png')


# ### **Ver el original**


import matplotlib.pyplot as plt
import numpy as np

# función para mostrar una imagen
def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize=(8, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# obtener algunas imágenes de entrenamiento aleatorias
dataiter = iter(trainloader)
images, labels = next(dataiter)

# mostrar imagenes
imshow(torchvision.utils.make_grid(images))





