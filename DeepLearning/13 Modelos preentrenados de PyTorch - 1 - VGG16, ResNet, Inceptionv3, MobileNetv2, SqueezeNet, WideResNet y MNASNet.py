#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Uso de modelos previamente entrenados en PyTorch**
# ### **Cargaremos los pesos de modelos preentrenados avanzados como:**
#
# ---
#
#
# 1. VGG16
# 2. ResNet
# h. Inicio vs
#4. Mobilina F
# 5. Squeeze Net
# 6. ResNet amplio
# h. cortesía
#
# **Ver todos los modelos disponibles en PyTorch aquí** - https://pytorch.org/vision/main/models.html

# # **1. Cargando VGG16**

# En[ ]:


import torchvision.models as models

model = models.vgg16(pretrained=True)


# ### **Echemos un vistazo a sus capas**

# En[ ]:


# Mira las capas del modelo
model


# ### **Compruebe el número de parámetros**

# En[ ]:


from torchsummary import summary 

summary(model, input_size = (3,224,224))


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
                                      transforms.ToTensor(),
                                      # transforma.Normalizar([0.485, 0.456, 0.406],
                                      #                      [0.229, 0.224, 0.225])
                                     ])


# **NOTA**
#
# ```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers, BatchNorm Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes:

# En[ ]:


model.eval()


# ## **Haz algunas inferencias**

# En[ ]:


# Obtener los nombres de las etiquetas de clase imageNet
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/imageNetclasses.json')


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


def predict_image(images, class_names):
    to_pil = transforms.ToPILImage()
    fig=plt.figure(figsize=(16,16))

    for (i,image) in enumerate(images):
      # Convertir a imagen y tensor
      image = to_pil(image)
      image_tensor = test_transforms(image).float()
      image_tensor = image_tensor.unsqueeze_(0)
      input = Variable(image_tensor)
      input = input.to(device)
      output = model(input)
      index = output.data.cpu().numpy().argmax()
      name = class_names[str(index)]
      
      # Trazar imagen
      sub = fig.add_subplot(len(images),1, i+1)
      sub.set_title(f'Predicted {str(name)}')
      plt.axis('off')
      plt.imshow(image)
    plt.show()

def get_images(directory='./images'):
    data = datasets.ImageFolder(directory, transform=test_transforms)
    num_images = len(data)
    loader = torch.utils.data.DataLoader(data, batch_size=num_images)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images


# En[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **2. Cargando ResNet**

# En[ ]:


import torchvision.models as models

model = models.resnet18(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


## **3. Inicio de carga**

# En[ ]:


import torchvision.models as models

model = models.inception_v3(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

# Tenga en cuenta que se espera el tamaño de entrada de diferencia con Inception
summary(model, input_size = (3,299,299))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **4. Cargando MobileNet**
#

# En[ ]:


import torchvision.models as models

model = models.mobilenet_v2(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **5. Cargando SqueezeNet**
#

# En[ ]:


import torchvision.models as models

model = models.squeezenet1_0(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **6. Carga amplia ResNet**
#

# En[ ]:


import torchvision.models as models

model = models.wide_resnet50_2(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **7. Cargando Wide MNASNet**
#

# En[ ]:


import torchvision.models as models

model = models.mnasnet1_0(pretrained=True)


# En[ ]:


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# En[ ]:


# Establecer en Eval y mirar las capas del modelo
model.eval()


# En[ ]:


images = get_images('./images')
predict_image(images, class_names)


# En[ ]:




