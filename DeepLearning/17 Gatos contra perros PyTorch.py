#!/usr/bin/env python
# codificación: utf-8

# En 3]:


import numpy as np 
import pandas as pd 
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
import time
import copy
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
from PIL import Image

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## **Los gatos contra los perros de Kaggle**
# Descargue nuestro conjunto de datos de gatos contra perros. Debe hablar alrededor de 15-20 segundos.
# Fuente: https://www.kaggle.com/c/dogs-vs-cats/data

# En 2]:


get_ipython().system('gdown --id 1Dvw0UpvItjig0JbnzbTgYKB-ibMrXdxk')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# En[4]:


# Establecer rutas de directorio para nuestros archivos
train_dir = './train'
test_dir = './test1'

# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)


# En[5]:


print(f'Number of images in {train_dir} is {len(train_files)}')
print(f'Number of images in {test_dir} is {len(test_files)}')


# En[7]:


imgpath = os.path.join(train_dir, train_files[0])
print(imgpath)


# #### **Crear nuestras transformaciones**

# En[8]:


transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])


# ## **Crear una clase de conjunto de datos que almacene la información de nuestro conjunto de datos (rutas, etiquetas y transformaciones**)
#
# Este objeto puede ser utilizado por funciones de antorcha como `torch.utils.data.random_split`
#
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

# En[9]:


class Dataset():
    def __init__(self, filelist, filepath, transform = None):
        self.filelist = filelist
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return int(len(self.filelist))

    def __getitem__(self, index):
        imgpath = os.path.join(self.filepath, self.filelist[index])
        img = Image.open(imgpath)

        if "dog" in imgpath:
            label = 1
        else:
            label = 0 

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)


# En[10]:


# Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)


# ### **Usando nuestro objeto de conjunto de datos**

# En[11]:


# Obtener una entrada de datos
train.__getitem__(0)


# En[12]:


# Obtener la forma de una sola imagen
print(val.__getitem__(0)[0].shape)
print(train.__getitem__(0)[0].shape)


# ## **Usando nuestro objeto de conjunto de datos para crear nuestro tren, división de validación**

# En[13]:


train, val = torch.utils.data.random_split(train,[20000,5000]) 


# En[14]:


# Obtener un tamaño de nuestro
print(len(train))
print(len(val))


# En[15]:


# Vamos a crear una matriz de nuestras etiquetas
val_set_class_count = [val.__getitem__(x)[1] for x in range(len(val)) ]


# En[16]:


import seaborn as sns

sns.countplot(val_set_class_count)


# ## **Cargadores de datos: creemos nuestro iterable sobre un conjunto de datos**

# En[18]:


train_dataset = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_dataset = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)


# ### **Úselo para obtener algunas imágenes de muestra**

# En 19]:


samples, labels = iter(train_dataset).next()
plt.figure(figsize=(16,24))
grid_imgs = torchvision.utils.make_grid(samples[:24])
np_grid_imgs = grid_imgs.numpy()

# en tensor, la imagen es (lote, ancho, alto), por lo que debe transponerla a (ancho, alto, lote) en números para mostrarla.
plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))


# **Ahora construimos nuestro Modelo**
#
# Usaremos el método ```nn.Sequential``` para construir nuestro modelo. Alternativamente, podemos usar el módulo funcional, sin embargo, esto es más simple y más similar a los estilos con los que trabajará en Keras.

# En 20]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            ) 
            
        self.conv2 =   nn.Sequential(
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ) 
        self.conv3 =   nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ) 

        self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64*5*5,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU()
        )
            
        self.fc2 = nn.Sequential(
        nn.Linear(128,2),
        )
                
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,dim = 1) 


# ### **Uso de TorchSummary para mostrar un resultado de resumen de estilo Keras**
#
# `summary(your_model, input_size=(channels, H, W))`

# En[21]:


model = CNN()
model.cuda()
summary(model,(3,60,60))


# ### **Definición de una función de pérdida y un optimizador**
#
# Necesitamos definir qué tipo de pérdida usaremos y qué método usaremos para actualizar los gradientes.
# 1. Usamos pérdida de entropía cruzada
# 2. Usamos el algoritmo de descenso de optimización de Adam; también especificamos una tasa de aprendizaje (LR) de 0.0005.
# 3. Establecer nuestras épocas en 50

# En[22]:


criterion = nn.CrossEntropyLoss().to(device)
optimiser = optim.Adam(model.parameters(),lr=0.0005)
epochs = 10


# ## **Entrena a nuestro modelo**
#
# **Usamos TQDM para realizar un entrenamiento estilo keras**
#

# En[23]:


type(train_dataset)


# En[ ]:


train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

for epoch in range(epochs):
    model.train()
    total = 0
    correct = 0
    counter = 0
    train_running_loss = 0

    # Establecemos nuestra unidad para tqdm y el número de iteraciones, es decir, len(train_dataset) sin necesidad de len ya que train_dataset es iterable
    # tepoch se convierte
    with tqdm(train_dataset, unit="batch") as tepoch:
        # nuestras etiquetas de la barra de progreso
        tepoch.set_description(f'Epoch {epoch+1}/{epochs}')

        for data,label in tepoch:
            data,label = data.to(device), label.to(device)
            optimiser.zero_grad()
            output = model(data)
            loss = criterion(output,label)
            loss.backward()
            optimiser.step() 

            train_running_loss += loss.item() * data.size(0)

            _, pred = torch.max(output.data, 1)

            # Mantenga un registro de cuántas imágenes se han propagado hacia adelante
            total += label.size(0)
            # Mantenga un registro de cuántos se predijeron que eran correctos
            correct += (pred == label).sum().item()

        train_accuracy.append(correct/total)
        train_loss.append(train_running_loss/len(train_dataset))
        print(f'Epoch {epoch+1} Training Accuracy = {correct/total}')
        print(f'Epoch {epoch+1} Training Loss = {train_running_loss/len(train_dataset)}')

    # Obtenga nuestra precisión de validación y puntajes de pérdida
    if epoch %1 == 0:
        model.eval()
        total = 0
        correct = 0
        val_running_loss = 0

        # No necesitamos gradientes para la validación, así que ajuste no_grad para ahorrar memoria
        with torch.no_grad():
            for val_data, val_label in val_dataset:
                val_data, val_label = val_data.to(device), val_label.to(device)
                val_output = model(val_data)
                loss_val = criterion(val_output, val_label)

                # Calcule la pérdida corriente multiplicando el valor de la pérdida por el tamaño del lote
                val_running_loss += loss_val.item() * val_data.size(0)
                _, pred = torch.max(val_output.data, 1)    
                total += val_label.size(0)
                correct += (pred == val_label).sum().item()

            val_accuracy.append(correct/total)
            # Calcule la pérdida por época dividiendo la pérdida en ejecución por el número de elementos en el conjunto de validación
            val_loss.append(val_running_loss/len(val_dataset))

            # print(val_running_loss)
            print(f'Epoch {epoch+1} Validation Accuracy = {correct/total}')
            print(f'Epoch {epoch+1} Validation Loss = {val_running_loss/len(val_dataset)}')


# En[ ]:


train_loss


# En[ ]:


epoch_log = [*range(epochs)]

# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch")
plt.xticks(rotation=45)

# Usamos twinx para crear un gráfico en un eje y secundario
ax2 = ax1.twinx()

# Crear gráfico para loss_log y precision_log
ax1.plot(epoch_log, train_loss, 'r-')
ax2.plot(epoch_log, train_accuracy, 'b-')

# Crear gráfico para loss_log y precision_log
ax1.plot(epoch_log, val_loss, 'r-')
ax2.plot(epoch_log, val_accuarcy, 'b-')

# Establecer etiquetas
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='r')
ax2.set_ylabel('Test Accuracy', color='b')

plt.show()


# En[ ]:


PATH = './cats_vs_dogs_10_epochs.pth'
torch.save(model.state_dict(), PATH)


# En[ ]:


# función para mostrar una imagen
def imshow(img):
    img = img / 2 + 0.5     # anormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Cargando un mini-lote
dataiter = iter(val_dataset)
images, labels = dataiter.next()

# Mostrar imágenes usando utils.make_grid() de torchvision
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ',''.join('%1s' % labels[j].numpy() for j in range(32)))


# En[ ]:


# Cree una instancia del modelo y muévala (memoria y operaciones) al dispositivo CUDA.
model = CNN()
model.to(device)

# Cargar pesos desde la ruta especificada
model.load_state_dict(torch.load(PATH))


# En[ ]:


samples, _ = iter(val_dataset).next()
samples = samples.to(device)

fig = plt.figure(figsize=(12, 8))
fig.tight_layout()

output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}

for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))


# En[ ]:




