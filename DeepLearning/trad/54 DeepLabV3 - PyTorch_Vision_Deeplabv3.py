#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **DeepLabV3 - PyTorch_Vision_Deeplabv3**
#
# Este portátil se acelera opcionalmente con un tiempo de ejecución de GPU.
# Si desea utilizar esta aceleración, seleccione la opción de menú "Tiempo de ejecución" -> "Cambiar tipo de tiempo de ejecución", seleccione "Acelerador de hardware" -> "GPU" y haga clic en "GUARDAR"
#
# ----------------------------------------------------------------------
#
# ## **DeepLabV3**
#
# *Autor: Equipo Pytorch*
#
# **Modelos DeepLabV3 con redes troncales ResNet-50, ResNet-101 y MobileNet-V3**
#
# _ | _
# - | -
# ![alt](https://pytorch.org/assets/images/deeplab1.png) | ![alt](https://pytorch.org/assets/images/deeplab2.png)

# En[7]:


import torch

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# o cualquiera de estas variantes
# modelo = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', preentrenado=Verdadero)
# modelo = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', preentrenado=Verdadero)
model.eval()


# Todos los modelos pre-entrenados esperan imágenes de entrada normalizadas de la misma manera,
# es decir, mini-lotes de imágenes RGB de 3 canales de forma `(N, 3, H, W)`, donde `N` es el número de imágenes, `H` y `W` se espera que sean al menos `224 ` píxeles.
# Las imágenes deben cargarse en un rango de `[0, 1]` y luego normalizarse usando `mean = [0.485, 0.456, 0.406]`
# y `std = [0.229, 0.224, 0.225]`.
#
# El modelo devuelve un `OrderedDict` con dos tensores que tienen la misma altura y anchura que el tensor de entrada, pero con 21 clases.
# `output['out']` contiene las máscaras semánticas, y `output['aux']` contiene los valores de pérdida auxiliar por píxel. En el modo de inferencia, `output['aux']` no es útil.
# Entonces, `output['out']` tiene la forma `(N, 21, H, W)`. Se puede encontrar más documentación [aquí] (https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

# En[8]:


# Descargue una imagen de ejemplo del sitio web de pytorch
import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# En[10]:


# ejemplo de ejecución (requiere torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # crear un mini-lote como lo espera el modelo

# mueva la entrada y el modelo a GPU para obtener velocidad si está disponible
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)
print(output_predictions)


# La salida aquí tiene la forma `(21, H, W)`, y en cada ubicación, hay probabilidades no normalizadas correspondientes a la predicción de cada clase.
# Para obtener la predicción máxima de cada clase y luego usarla para una tarea posterior, puede hacer `output_predictions = output.argmax(0)`.
#
# Aquí hay un pequeño fragmento que traza las predicciones, con cada color asignado a cada clase (vea la imagen visualizada a la izquierda).

# En[6]:


from IPython.display import Image
Image('deeplab1.png')


# En[5]:


# crear una paleta de colores, seleccionando un color para cada clase
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# trazar las predicciones de segmentación semántica de 21 clases en cada color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt

plt.imshow('deeplab1.png')
plt.imshow(r)


# ### Descripcion del modelo
#
# Deeplabv3-ResNet está construido por un modelo Deeplabv3 utilizando una red troncal ResNet-50 o ResNet-101.
# Deeplabv3-MobileNetV3-Large está construido por un modelo Deeplabv3 utilizando la red troncal grande MobileNetV3.
# El modelo preentrenado se entrenó en un subconjunto de COCO train2017, en las 20 categorías que están presentes en el conjunto de datos Pascal VOC.
#
# Sus precisiones de los modelos preentrenados evaluados en el conjunto de datos COCO val2017 se enumeran a continuación.
#
# | Estructura del modelo | Pagaré medio | Precisión global de píxeles |
# | ---------------------------- | ----------- | --------------------------|
# | deeplabv3_resnet50 | 66.4 | 92,4 |
# | deeplabv3_resnet101 | 67.4 | 92,4 |
# | deeplabv3_mobilenet_v3_large | 60.3 | 91.2 |
#
# ### Recursos
#
# - [Repensar la convolución de Atrous para la segmentación semántica de imágenes](https://arxiv.org/abs/1706.05587)
