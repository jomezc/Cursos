#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Uso de modelos preentrenados en PyTorch para obtener precisión de rango 1 y rango 5**
# 1. Primero cargaremos el modelo ImageNet VGG16 previamente entrenado
# 2. Obtendremos las 5 mejores clases a partir de una sola inferencia de imagen
# 3. A continuación, construiremos una función que nos dé la precisión de rango N usando algunas imágenes de prueba.
#
# ---
#

# En[ ]:


# Cargue nuestro VGG16 pre-entrenado
import torchvision.models as models

model = models.vgg16(pretrained=True)


# ## **Normalización**
#
# Todos los modelos preentrenados esperan imágenes de entrada **normalizadas** de la misma manera, es decir, mini lotes de imágenes RGB de 3 canales de forma (3 x H x W), donde se espera que H y W sean al menos 224 Las imágenes deben cargarse en un rango de [0, 1] y luego normalizarse usando la media = [0,485, 0,456, 0,406] y std = [0,229, 0,224, 0,225]. Puede usar la siguiente transformación para normalizar:
#
# `normalizar = transforma.Normalizar(media=[0.485, 0.456, 0.406],
# estándar=[0.229, 0.224, 0.225])`

# En[ ]:


from torchvision import datasets, transforms, models

data_dir = '/images'

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),])


# **NOTA**
#
# ```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers, BatchNorm Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes:

# En[ ]:


model.eval()


# ### **Descargue nuestro nombre de clase de ImageNet y nuestras imágenes de prueba**

# En[ ]:


# Obtenga los nombres de las etiquetas de clase de imageNet y las imágenes de prueba
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/imageNetclasses.json')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')
get_ipython().system('rm -rf ./images/class1/.DS_Store')


# ## **Importar nuestros módulos**

# En[ ]:


import torch
import json
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('imageNetclasses.json') as f:
  class_names = json.load(f)


# # **Cargar y ejecutar una sola imagen a través de nuestro modelo pre-entrenado**

# En[ ]:


from PIL import Image
import numpy as np

image = Image.open('./images/class1/1539714414867.jpg')

# Convertir a Tensor
image_tensor = test_transforms(image).float()
image_tensor = image_tensor.unsqueeze_(0)
input = Variable(image_tensor)
input = input.to(device)
output = model(input)
index = output.data.cpu().numpy().argmax()
name = class_names[str(index)]

# Trazar imagen
fig=plt.figure(figsize=(8,8))
plt.axis('off')
plt.title(f'Predicted {name}')
plt.imshow(image)
plt.show()


# ## **Obtén nuestras probabilidades de clase**

# En[ ]:


import torch.nn.functional as nnf

prob = nnf.softmax(output, dim=1)

top_p, top_class = prob.topk(5, dim = 1)
print(top_p, top_class)


# En[ ]:


# Convertir a matriz numpy
top_class_np = top_class.cpu().data.numpy()[0]
top_class_np


# ## **Crear una clase que nos dé los nombres de nuestras clases**

# En[ ]:


def getClassNames(top_classes):
  top_classes = top_classes.cpu().data.numpy()[0]
  all_classes = []
  for top_class in top_classes:
    all_classes.append(class_names[str(top_class)])
  return all_classes


# En[ ]:


getClassNames(top_class)


# # **Construye nuestra función para darnos nuestra Precisión de Rango-N**

# En[ ]:


from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,16))

def getRankN(model, directory, ground_truth, N, show_images = True):
  # Obtener nombres de imágenes en el directorio
  onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

  # Almacenaremos aquí los nombres de las N mejores clases
  all_top_classes = []

  # Iterar a través de nuestras imágenes de prueba
  for (i,image_filename) in enumerate(onlyfiles):
    image = Image.open(directory+image_filename)

    # Convertir a Tensor
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    # Obtener nuestras probabilidades y nombres de clase top-N
    prob = nnf.softmax(output, dim=1)
    top_p, top_class = prob.topk(N, dim = 1)
    top_class_names = getClassNames(top_class)
    all_top_classes.append(top_class_names)

    if show_images:
      # Trazar imagen
      sub = fig.add_subplot(len(onlyfiles),1, i+1)
      x = " ,".join(top_class_names)
      print(f'Top {N} Predicted Classes {x}')
      plt.axis('off')
      plt.imshow(image)
      plt.show()

  return getScore(all_top_classes, ground_truth, N)

def getScore(all_top_classes, ground_truth, N):
  # Calcular rango-Y puntuación
  in_labels = 0
  for (i,labels) in enumerate(all_top_classes):
    if ground_truth[i] in labels:
      in_labels += 1
  return f'Rank-{N} Accuracy = {in_labels/len(all_top_classes)*100:.2f}%'


# En[ ]:


# Crea nuestras etiquetas de verdad en el suelo
ground_truth = ['basketball',
                'German shepherd, German shepherd dog, German police dog, alsatian',
                'limousine, limo',
                "spider web, spider's web",
                'burrito',
                'beer_glass',
                'doormat, welcome mat',
                'Christmas stocking',
               'collie']


# ## **Obtener precisión de rango 5**

# En[ ]:


getRankN(model,'./images/class1/', ground_truth, N=5)


# ## **Obtener precisión de rango 1**

# En[ ]:


getRankN(model,'./images/class1/', ground_truth, N=1)


# ## **Obtener precisión de rango 10**

# En[ ]:


getRankN(model,'./images/class1/', ground_truth, N=10)


# En[ ]:




