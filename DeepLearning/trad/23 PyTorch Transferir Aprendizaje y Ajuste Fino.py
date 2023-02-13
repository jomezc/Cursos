#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **PyTorch - Transferir aprendizaje con hormigas vs abejas**
#
# ---
#
# En esta lección, aprendemos cómo configurar generadores de datos para cargar nuestro propio conjunto de datos y entrenar un clasificador usando Keras.
# 1. Configurar nuestros datos
# 2. Construyendo nuestro modelo para Transfer Learning
# 3. Ajuste fino de Convnet
# 4. ConvNet como extractor de funciones fijas

# En 1]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   #  modo interactivo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# En 2]:


# Descarga nuestro conjunto de datos
get_ipython().system('wget https://download.pytorch.org/tutorial/hymenoptera_data.zip')
get_ipython().system('unzip hymenoptera_data.zip')


# ### **Establecer nuestras transformaciones de datos**

# En 3]:


# Aumento y normalización de datos para entrenamiento
# Solo normalización para validación
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Resta la media de cada valor y luego lo divide por la desviación estándar.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # cambia el tamaño de las imágenes para que el lado más corto tenga una longitud de 256 píxeles.
        # El otro lado se escala para mantener la relación de aspecto de la imagen.
        transforms.Resize(256), 
        # recorta el centro de la imagen para que sea una imagen cuadrada de 224 x 224 píxeles.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ### **Crear nuestros cargadores de datos**
# El comando **datasets.ImageFolder()** espera que nuestros datos estén organizados de la siguiente manera: `root/label/image.jpg`. En otras palabras, las imágenes deben ordenarse en carpetas. Por ejemplo, todas las imágenes de abejas deben estar en una carpeta, todas las imágenes de hormigas deben estar en otra, así:
# - nombre_del_conjunto_de_datos/
# - /hormigas/
# - nombre_del_conjunto_de_datos/
#   - /abejas/

# En[4]:


# Establecer en la ruta de su imagen
data_dir = './hymenoptera_data'

# Use ImageFolder para apuntar a nuestro conjunto de datos completo
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Crear nuestros cargadores de datos
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Obtener nuestros tamaños de conjuntos de datos
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].classes
print(class_names)


# ### **Visualizar algunas Imágenes**
#

# En[5]:


def imshow(inp, title=None):
    """ Imshow para tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pausa un poco para que se actualicen las tramas


# Obtenga un lote de datos de entrenamiento
inputs, classes = next(iter(dataloaders['train']))

# Hacer una grilla por lotes
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# ### **Crear función para entrenar modelo**
#
# También usaremos el nativo de PyTorch:
#
# - programador de tasa de aprendizaje
# - Checkpoint - guardando el mejor modelo
#
# A continuación, el parámetro ``scheduler`` es un objeto del programador LR de
# ``torch.optim.lr_scheduler``.

# En[6]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Cada época tiene una fase de entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Establecer modelo en modo de entrenamiento
            else:
                model.eval()   # Establecer el modelo para evaluar el modo

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre los datos.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # poner a cero los gradientes de parámetros
                optimizer.zero_grad()

                #  adelante
                # historial de seguimiento si solo está en tren
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # hacia atrás + optimizar solo si está en fase de entrenamiento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #  Estadísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copia profunda del modelo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # carga los pesos de los mejores modelos
    model.load_state_dict(best_model_wts)
    return model


# ### **Crear función para visualizar las predicciones de nuestro modelo**

# En[7]:


def visualize_predictions(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


### **3. Ajuste fino de Convnet**
# Cargamos un modelo **resnet18** preentrenado y cambiamos la capa final completamente conectada para generar el tamaño de nuestra clase (2).

# En[8]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Aquí el tamaño de cada muestra de salida se establece en 2.
# Alternativamente, se puede generalizar a nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observa que todos los parámetros están siendo optimizados
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decae LR por un factor de 0.1 cada 7 épocas
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# ### **Entrenar y Evaluar**

# En[10]:


model_ft = train_model(model_ft,
                       criterion,
                       optimizer_ft,
                       exp_lr_scheduler,
                       num_epochs=5)


# En[11]:


visualize_predictions(model_ft)


### **4. ConvNet como extractor de características fijas**
# ----------------------------------
#
# Aquí, necesitamos **congelar** todas las capas de red excepto la capa final.
#
# Nosotros necesitamos
# para establecer ``requires_grad == False`` para congelar los parámetros para que el
# los gradientes no se calculan en ``hacia atrás()``.
#
# Puedes leer más sobre esto en la documentación
# aquí https://pytorch.org/docs/notes/autograd.html#excluyendo-subgraphs-from-backward.
#
#
#

# En[12]:


model_conv = torchvision.models.resnet18(pretrained=True)

# Aquí congelamos capas
for param in model_conv.parameters():
    param.requires_grad = False

# Los parámetros de los módulos recién construidos tienen require_grad=True por defecto
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe que solo los parámetros de la capa final se están optimizando como
# opuesto a antes.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decae LR por un factor de 0.1 cada 7 épocas
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# ### **Entrenar y Evaluar**

# En[13]:


model_conv = train_model(model_conv,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         num_epochs=5)


# En[14]:


visualize_predictions(model_conv)

