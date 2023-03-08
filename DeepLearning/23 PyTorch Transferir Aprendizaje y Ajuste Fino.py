#!/usr/bin/env python
# codificación: utf-8

# # **PyTorch - Transferir aprendizaje con hormigas vs abejas**
#
# ---
#
# En esta lección, aprendemos cómo configurar generadores de datos para cargar nuestro propio conjunto de datos y
# entrenar un clasificador usando pytorch.
# 1. Configurar nuestros datos
# 2. Construyendo nuestro modelo para Transfer Learning
# 3. Ajuste fino de Convnet
# 4. ConvNet como extractor de funciones fijas

# Explicación de programador de tasa de aprendizaje.
'''
Encontrados Varios programadores explicados en:
https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863

Las redes neuronales tienen muchos hiperparámetros que afectan el rendimiento del modelo. Uno de los hiperparámetros
esenciales es la tasa de aprendizaje (LR), que determina cuánto cambian los pesos del modelo entre los pasos de
entrenamiento. En el caso más simple, el valor LR es un valor fijo entre 0 y 1.

Un programador de tasa de aprendizaje ajusta la tasa de aprendizaje de acuerdo con un programa predefinido durante el
proceso de capacitación. Entre ellos está step LR

El StepLRreduce la tasa de aprendizaje por un factor multiplicativo después de cada número predefinido de pasos de
entrenamiento.

from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer,
                   step_size = 4 , # Período de disminución de la tasa de aprendizaje
                    gamma = 0.5 ) # Factor multiplicativo de disminución de la tasa de aprendizaje
'''


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


# Descarga nuestro conjunto de datos
'''wget https://download.pytorch.org/tutorial/hymenoptera_data.zip'''


# ### **Establecer nuestras transformaciones de datos**


# Aumento y normalización de datos para entrenamiento
# Solo normalización para validación
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Dado que la red original fue entrenado con un conjuntop de datos en un formato determinado, tenemos que
        # normalizar el nuestro para que sea como con el que se entrenó.
        # Resta la media de cada valor y luego lo divide por la desviación estándar.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # Cambia el tamaño de las imágenes para que el lado más corto tenga una longitud de 256 píxeles.
        # El otro lado se escala para mantener la relación de aspecto de la imagen.
        transforms.Resize(256), 
        # recorta el centro de la imagen para que sea una imagen cuadrada de 224 x 224 píxeles.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ### **Crear nuestros cargadores de datos**
# El comando **datasets.ImageFolder()** espera que nuestros datos estén organizados de la siguiente manera:
# `root/label/image.jpg`. En otras palabras, las imágenes deben ordenarse en carpetas. Por ejemplo, todas las imágenes
# de abejas deben estar en una carpeta, todas las imágenes de hormigas deben estar en otra, así:
# - nombre_del_conjunto_de_datos/
# - /hormigas/
# - nombre_del_conjunto_de_datos/
#   - /abejas/
# Esto tanto pata train como para val, ejemplo completo. la ruta es hymenoptera_data, y es la que se le pasa. dentro
# tenemos
#   - train/ants(etiqueta)/imágenes
#   - train/bees(etiqueta)/imágenes
#   - val/ants(etiqueta)/imágenes
#   - val/bees(etiqueta)/imágenes


# Establecer en la ruta de su imagen
data_dir = 'images/hymenoptera_data'

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

def imshow(inp, title=None):
    """ Imshow para tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # tiempo para monitorizacion

    best_model_wts = copy.deepcopy(model.state_dict())  # realizamos una copia para salvar el estado original del modelo
    best_acc = 0.0  # monitorización del accuracy

    for epoch in range(num_epochs):  # para el número de épocas que establezcamos
        # pintar información de la época que está realizando
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

            # copia profunda del modelo, hacemos un seguimiento del mejor modelo guardando una copia del mejor y su
            # Accuracy
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
# en ajuste fino bajamos un modelo preentrenado y no congelamos ninguna capa o congelamos alguna, en este caso ninguna.
# Cargamos un modelo **resnet18** preentrenado y cambiamos la capa final completamente conectada para generar el tamaño
# de nuestra clase (2).

model_ft = models.resnet18(weights=True)  # cargamos el modelo
num_ftrs = model_ft.fc.in_features  # obtenemos la cantidad de características aquí del modelo que acabamos de cargar

# Aquí el tamaño de cada muestra de salida se establece en 2,
# estamos cambiando la capa final totalmente conectada adaptándola a la salida
# Alternativamente, se puede generalizar a nn.Linear(num_ftrs, len(class_names)).
'''Comentamos que se establecería nuestro tamaño de muestra solo para establecer dos clases como la salida final, 
la capa lineal para decirme que las características extraidas del modelo son la entrada para esa capa y  el numero de 
clase es la salida, y esta entrada  de características procedía venía de la última capa completamente conectada , son 
solo una serie de características, ya sabes, que simplemente las envían todas al a la cabecera.'''
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()  # establecemos el criterio , en este caso de entropía cruzada

# Observa que todos los parámetros están siendo optimizados
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  # definimos el optimizador

# Decae LR por un factor de 0.1 cada 7 épocas
# explicación de lo que es un programador de tasa de aprendizaje arriba,
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                       step_size=7,  # Período de disminución de la tasa de aprendizaje
                                       gamma=0.1)   # Factor multiplicativo de disminución de la tasa de aprendizaje
# en estos dos pasos anteriores definimos un optimizador cuyo valor de LR ( tasa de aprendizaje) vamos modificando
# automáticamente en función de un programa cuando se cumplen unas condiciones o características durante la ejecución
# del entrenamiento, cada 7 pasos se reduce un 10% en este caso

# ### **Entrenar y Evaluar**

model_ft = train_model(model_ft,
                       criterion,
                       optimizer_ft,
                       exp_lr_scheduler,
                       num_epochs=5)

visualize_predictions(model_ft)


# ## **4. ConvNet como extractor de características fijas**
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
'''Lo que vamos a hacer ahora es usar transferencia de aprendizaje por extracción de características,
lo que significa que todo lo que hacemos es congelar todas las capas convolucionales, todo el primer bloque de capas
y la red que cargamos y luego entrenamos a los mejores jugadores.'''
#

model_conv = torchvision.models.resnet18(pretrained=True)

# Aquí congelamos capas
# usamos param.requires_grad para congeglar cada convolución del modelo
for param in model_conv.parameters():
    param.requires_grad = False


# tomamos las funciones totalmente conectadas aquí, las funciones de la última capa, y luego las agregamos a nuestra
# clase de salida aquí, estos módulos recién construidos si van a ser entrenables, ya que Los parámetros de los módulos
# recién construidos tienen requires_grad=True por defecto
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe que solo los parámetros de la capa final se están optimizando como
# opuesto a antes.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decae LR por un factor de 0.1 cada 7 épocas
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv,
                                       step_size=7,  # Período de disminución de la tasa de aprendizaje
                                       gamma=0.1)  # Factor multiplicativo de disminución de la tasa de aprendizaje


# ### **Entrenar y Evaluar**

model_conv = train_model(model_conv,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         num_epochs=5)


visualize_predictions(model_conv)

