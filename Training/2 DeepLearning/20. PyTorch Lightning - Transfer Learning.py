#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **PyTorch Lightning - Transfer Learning**
# ---
# https://pytorch-lightning.readthedocs.io/en/1.2.0/advanced/transfer_learning.html
# ---
# 
# In this lesson, we learn to use the amazing library **PyTorch Lightning**. It's a great way to organize. your PyTorch code and get many great features and added benefits. We'll be doing the following in this guide:
# 1. Setup and Install Lightning
# 2. Create our Lightning Model Class and use a pre-trained model
# 3. Train our model

# ## **1. Setup and Install Lightning**

# In[1]:


# First we install PyTorch Lightning and TorchMetricsß
get_ipython().system('pip install pytorch-lightning --quiet')
get_ipython().system('pip install torchmetrics')


# In[2]:


# Import all packages we'll be using
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


# #### **Download our datasets**

# In[3]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/dogs-vs-cats.zip')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# ## **Setup our Dataloaders**

# In[4]:


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


# Set directory paths for our files
train_dir = './train'
test_dir = './test1'

# Get files in our directories
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])

# Create our train and test dataset objects
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

train, val = torch.utils.data.random_split(train,[20000,5000]) 

train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)


# ## **2.Creating our Lightning Module using a Pre-trained (ImageNet) Model**
# 
# Using pre-trained models for Transfer Learning is simple! 
# 
# All we do is load the model weights in the init function. Here we use **resNet50**.

# In[5]:


import torchvision.models as models

class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.accuracy = torchmetrics.Accuracy()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        # use the pretrained model
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # Change our forward function to include the 4 lines below
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
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_acc_step', self.accuracy(val_output, val_label))
        self.log('val_loss', val_loss)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ## **Start Training - Exactly the same as before**

# In[6]:


model = ImagenetTransferLearning()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=10)

trainer.fit(model)


# In[ ]:


# Start tensorboard.
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')

