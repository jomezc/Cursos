#!/usr/bin/env python
# codificación: utf-8

#
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **PyTorch - Similitud facial con redes siamesas en Pytorch**
#
# ---
#
#
# En esta lección, implementaremos **Similitud facial con redes siamesas en Pytorch** usando PyTorch.
#
# El objetivo es enseñar a una red siamesa a poder distinguir pares de imágenes. El esquema de esta lección es el siguiente:
#
# 1. Cargue nuestros módulos, datos y defina algunas funciones de utilidad
# 2. Configure nuestro procesamiento de datos: cree nuestros pares de imágenes
# 3. Construyendo nuestra Red Siamés
# 4. Comienza a entrenar
# 5. Ver los resultados de nuestras SOCKET
#
# **Créditos**:
#
# Puede leer el artículo adjunto en https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
#
# Fuente: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
#
# Nota:
# Se puede usar cualquier conjunto de datos. Cada clase debe estar en su propia carpeta. Esta es la misma estructura que
# utiliza el propio conjunto de datos de carpetas de imágenes de PyTorch.

### **1. Cargue nuestros módulos, datos y defina algunas funciones de utilidad**



import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# ### **Cree una función simple de visualización y trazado de imágenes**

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# ### **Descargue nuestro conjunto de datos - El conjunto de datos de AT&T Faces**
# Fuente: https://www.kaggle.com/kasikrit/att-database-of-faces

# Descarga y descomprime nuestros datos
'''wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/face_data.zip'''
'''unzip -q face_data.zip'''


class Config():
    training_dir = "images/data/faces/training/"
    testing_dir = "images/data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100


### **2. Configure nuestro procesamiento de datos: cree nuestros pares de imágenes**

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # necesitamos asegurarnos de que aproximadamente el 50% de las imágenes estén en la misma clase
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                # seguir recorriendo hasta que se encuentre la misma imagen de clase
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                # seguir repitiendo hasta que se encuentre una imagen de clase diferente
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# ### **Establecer carpetas de imágenes**

folder_dataset = dset.ImageFolder(root=Config.training_dir)


# ### **Crear nuestros transformadores**

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()]),
                                                                      should_invert=False)


# ### **Cree nuestro cargador de datos y vea algunas imágenes de muestra**


vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=4,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())


### **3. Construyendo nuestra red siamesa**


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),  # Rellena el tensor de entrada utilizando la reflexión del límite de entrada.


            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),)

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# ## **Definir nuestra función de pérdida de contraste**


class ContrastiveLoss(torch.nn.Module):
    """Función de pérdida de contraste.
Basado en: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# ### **Hacer nuestro cargador de datos de entrenamiento**


train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)


# ### **Definir nuestro Loss y Optimizer antes del entrenamiento**


net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )


### **4. Empezar a entrenar**

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
            
show_plot(counter,loss_history)


### **5. Ver los resultados de nuestras SOCKET**
#
# Los últimos 3 temas quedaron fuera del entrenamiento y se utilizarán para la prueba. La Distancia entre cada par de
# imágenes indica el grado de similitud que el modelo encontró entre las dos imágenes. Menos significa que encontró
# más similares, mientras que los valores más altos indican que los encontró diferentes.

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))



