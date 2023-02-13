#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **PyTorch Lightning - Transferencia de aprendizaje**
# ---
# https://pytorch-lightning.readthedocs.io/en/1.2.0/advanced/transfer_learning.html
# ---
#
# En esta lección, aprendemos a usar la increíble biblioteca **PyTorch Lightning**. Es una excelente manera de organizarse. su código PyTorch y obtenga muchas funciones excelentes y beneficios adicionales. Haremos lo siguiente en esta guía:
# 1. Configurar e instalar Lightning
# 2. Cree nuestra clase de modelo Lightning y use un modelo pre-entrenado
# 3. Entrena a nuestro modelo

### **1. Configurar e instalar Lightning**

# En 1]:


# Primero instalamos PyTorch Lightning y TorchMetricsß
get_ipython().system('pip install pytorch-lightning --quiet')
get_ipython().system('pip install torchmetrics')


# En 2]:


# Importar todos los paquetes que usaremos
import os
import torch
import torchmetrics
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image


# #### **Descargue nuestros conjuntos de datos**

# En 3]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/dogs-vs-cats.zip')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# ## **Configurar nuestros cargadores de datos**

# En[4]:


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


# Establecer rutas de directorio para nuestros archivos
train_dir = './train'
test_dir = './test1'

# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])

# Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

train, val = torch.utils.data.random_split(train,[20000,5000]) 

train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)


# ## **2.Creación de nuestro módulo Lightning usando un modelo preentrenado (ImageNet)**
#
# ¡Usar modelos previamente entrenados para Transfer Learning es simple!
#
# Todo lo que hacemos es cargar los pesos del modelo en la función init. Aquí usamos **resNet50**.

# En[5]:


import torchvision.models as models

class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.accuracy = torchmetrics.Accuracy()

        # iniciar un resnet preentrenado
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        # usar el modelo preentrenado
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # Cambiar nuestra función de reenvío para incluir las 4 líneas a continuación
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return F.softmax(x,dim = 1) 

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output,label)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(output, label))
        return {'loss': loss, 'log': self.log}

    def training_epoch_end(self, outs):
        # métrica de época de registro
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_acc_step', self.accuracy(val_output, val_label))
        self.log('val_loss', val_loss)

    def validation_epoch_end(self, outs):
        # métrica de época de registro
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        # optimizador = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ## **Empezar a entrenar - Exactamente igual que antes**

# En[6]:


model = ImagenetTransferLearning()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=10)

trainer.fit(model)


# En[ ]:


# Iniciar tensorboard.
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')

