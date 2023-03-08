#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **PyTorch Cats vs Dogs - Extracción de funciones**
'''
En lugar de tener una red neuronal completamente conectada en la parte superior, usaremos una red lineal
como una regresión logística para usar las entradas de nuestra capa anterior como entradas para ese modelo.
Y luego y luego crear un modelo a partir de eso.
'''

# En esta lección, aprenderemos a usar una red preentrenada como extractor de funciones. Luego usaremos esas
# características como entrada para nuestro clasificador de regresión logística.
# 1. Cargue nuestro modelo VGG16 preentrenado
# 2. Descarga nuestros datos y configura nuestras transformaciones
# 3. Extraiga nuestras características usando VGG16
# 4. Entrena un Clasificador LR usando esas características
# 5. Ejecute algunas inferencias en nuestros datos de prueba
# ---
# ### **Necesitará usar High-RAM y GPU (para aumentar la velocidad).**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)
#
# ---

### **1. Descarga nuestros Modelos Pre-entrenados (VGG16)**


import torch
import os
import tqdm
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchsummary import summary 
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
model = model.to(device)

summary(model, input_size = (3,224,224))


# ### **Eliminar las capas superiores densas totalmente conectadas**


# eliminar las últimas capas completamente conectadas ( siempre eliminamos todas las conectadas para extraer
# funciones en este paso)
# convertimos en una lista y eliminamos las últimas capas
new_classifier = nn.Sequential(*list(model.classifier.children())[:-7])
# a partir de ese punto creamos un nuevo clasificador
model.classifier = new_classifier


# `pitón
# secuencial(
# (0): Lineal (in_features=25088, out_features=4096, bias=True)
# (1): ReLU(inplace=True)
# (2): abandono (p = 0,5, en el lugar = falso)
# (3): Lineal (in_features=4096, out_features=4096, bias=True)
# (4): ReLU(en lugar=Verdadero)
# (5): abandono (p = 0.5, en el lugar = falso)
# (6): Lineal(in_features=4096, out_features=1000, bias=True)`


summary(model, input_size = (3,224,224))


### **2. Descargue nuestros datos y configure nuestros transformadores**


# Establecer rutas de directorio para nuestros archivos
train_dir = 'images/gatos_perros/train'
test_dir = 'images/gatos_perros/test1'

# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

print(f'Number of images in {train_dir} is {len(train_files)}')
print(f'Number of images in {test_dir} is {len(test_files)}')

# al ser el origen vgg no necesitamos muchas transformaciones, solo cambiar el tamaño a 224x224 y pasarlo a tensor
transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])

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

# Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

# Crear nuestros cargadores de datos
train_dataset = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_dataset = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=True)


### **3. Extrae nuestras características usando VGG16**

image_names = os.listdir('images/gatos_perros/train')
image_paths = ['images/gatos_perros/train/' + x for x in image_names]

# ponemos el modelo en modo de evaluación porque no lo estamos entrenando sólo estamos usando el modelo para extraer
# características.
# Los filtros CNN son básicamente detectores de características para dar efecto a bordes, complejos patrones complejos,
# y todas esas cosas que podemos y podríamos incluir en cualquier imagen en términos de esas características.
# características y luego utilizar esas características como entradas en otro modelo.

model.eval() 
model = model.cuda()

with torch.no_grad():
    features = None
    image_labels = None

    # recorrer cada lote y pasar nuestros tensores de entrada al modelo hte
    for data, label in tqdm.tqdm(train_dataset):
        x = data.cuda()
        output = model(x)
        
        if features is not None:
            # Concatena la secuencia dada de tensores en la dimensión dada.
            # cat necesita al menos dos tensores, por lo que solo comenzamos a cat después del primer ciclo
            features = torch.cat((features, output), 0)
            image_labels = torch.cat((image_labels, label), 0)
        else:
            features = output
            image_labels = label

    # reformar nuestro tensor a 25000 x 25088 , debido a la salida de la última capa 512 * 7 * 7
    features = features.view(features.size(0), -1)

# Compruebe que tenemos funciones para todas las 25000 imágenes
print(features.size(0))


# Compruebe que tenemos etiquetas para todas las 25000 imágenes
print(image_labels.shape)


# Verifique la forma para asegurarse de que nuestras características sean una matriz aplanada de 512 * 7 * 7
print(features.shape)


### **4. Entrene a un clasificador LR usando esas características**

# Convertir nuestros tensores en matrices numpy
features_np = features.cpu().numpy()
image_labels_np = image_labels.cpu().numpy()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Divida nuestro modelo en un conjunto de datos de prueba y entrenamiento para entrenar nuestro clasificador LR
X_train, X_test, y_train, y_test = train_test_split(features_np, image_labels_np, test_size=0.2, random_state = 7)
# random state es un número aleatorio puede ponerse el que sea

'''Creamos un objeto de regresión logística '''
glm = LogisticRegression(C=0.1)
glm.fit(X_train, y_train)  # y ajustamos el modelo


# Obtener Precisión
accuracy = glm.score(X_test, y_test)
print(f'Accuracy on validation set using Logistic Regression: {accuracy*100}%')


### **5. Ejecute algunas inferencias en nuestros datos de prueba**



image_names_test = os.listdir("images/gatos_perros/test1")
image_paths_test = ["images/gatos_perros/test1/" + x for x in image_names_test]


from torch.autograd import Variable

imsize = 224

loader = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


import random 

test_sample = random.sample(image_paths_test, 12)
model.eval() 

def test_img():
    result_lst = []
    for path in test_sample:
      image = image_loader(loader, path)
      output = model(image.to(device))
      output = output.cpu().detach().numpy() 
      result = glm.predict(output)
      # El modelo de regresión logística va a devolver un numero entre 0 y 1. se establece que si es mayor de 0,5 es un
      # perro y si no es un gato
      result = 'dog' if float(result) >0.5 else 'cat'
      result_lst.append(result)
    return result_lst

# obtener predicciones de prueba de todos los modelos
pred_results = test_img()
pred_results



import cv2

plt.figure(figsize=(15, 15))

for i in range(0, 12):
    plt.subplot(4, 3, i+1)
    result = pred_results[i]
    img = test_sample[i]
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.text(72, 248, f'Feature Extractor CNN: {result}', color='lightgreen',fontsize= 12, bbox=dict(facecolor='black', alpha=0.9))
    plt.imshow(image)

plt.tight_layout()
plt.show()




