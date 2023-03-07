#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Uso de modelos previamente entrenados en PyTorch**
# ### **Cargaremos los pesos de modelos preentrenados avanzados como:**
#
#     VGG16
#     ResNet
#     Inception v3
#     MobileNet v2
#     SqueezeNet
#     Wide ResNet
#     MNASNet
#
# **Ver todos los modelos disponibles en PyTorch aquí** - https://pytorch.org/vision/main/models.html
'''pip install torch-summary # torchsummary deprecado consultado en foro oficial!!'''


# ## **Normalización**
#
# Todos los modelos preentrenados esperan imágenes de entrada **normalizadas** de la misma manera, es decir, mini lotes
# de imágenes RGB de 3 canales de forma (3 x H x W), donde se espera que H y W sean al menos 224 Las imágenes deben
# cargarse en un rango de [0, 1] y luego normalizarse usando la media = [0,485, 0,456, 0,406] y
# std = [0,229, 0,224, 0,225]. Puede usar la siguiente transformación para normalizar:

# `normalizar = transforma.Normalizar(media=[0.485, 0.456, 0.406],
# estándar=[0.229, 0.224, 0.225])`


from torchvision import datasets, transforms, models

data_dir = 'images'
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      # transforma.Normalizar([0.485, 0.456, 0.406],
                                      #                      [0.229, 0.224, 0.225])
                                     ])

import torch
import json
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

print("GPU available: {}".format(torch.cuda.is_available()))  # GPU available: True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('models/imageNetclasses.json') as f:
    class_names = json.load(f)


def predict_image(images, class_names):
    to_pil = transforms.ToPILImage()
    fig = plt.figure(figsize=(16, 16))

    for (i, image) in enumerate(images):
        # Convert to image and tensor
        image = to_pil(image)
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        # input = input.to("cpu")  # error en gpu por los pesos por defecto del modelo, hay que forzar .cuda()
        # ejemplo -> models.vgg16(weights=True).cuda()
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        name = class_names[str(index)]

        # Plot image
        sub = fig.add_subplot(len(images), 1, i + 1)
        sub.set_title(f'Predicted {str(name)}')
        plt.axis('off')
        plt.imshow(image)
    plt.show()


def get_images(directory='images'):
    data = datasets.ImageFolder(directory, transform=test_transforms)
    num_images = len(data)
    loader = torch.utils.data.DataLoader(data, batch_size=num_images)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images


'''wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip'''

# SE ESTABLECEN EN EVAL PORQUE  LOS MODELOS SE COMPORTAN DE FORMA DIFERENTE


# *******************************************+
# # **1. Cargando VGG16**

import torchvision.models as models

model = models.vgg16(weights=True).cuda()


# ### **Echemos un vistazo a sus capas**
# Mira las capas del modelo
print(model)
'''VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)'''

# ### **Compruebe el número de parámetros**

from torchsummary import summary
print(summary(model, input_size = (3,224,224)))
'''---------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-32            [-1, 512, 7, 7]               0
           Linear-33                 [-1, 4096]     102,764,544
             ReLU-34                 [-1, 4096]               0
          Dropout-35                 [-1, 4096]               0
           Linear-36                 [-1, 4096]      16,781,312
             ReLU-37                 [-1, 4096]               0
          Dropout-38                 [-1, 4096]               0
           Linear-39                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.78
Params size (MB): 527.79
Estimated Total Size (MB): 747.15
----------------------------------------------------------------'''
# **NOTA**
#
# ```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de
# manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers,
# BatchNorm Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la
# práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de
# gradientes:

model.eval()

images = get_images()
predict_image(images, class_names)

# *******************************************+
# *** Cargando Resnet

import torchvision.models as models

model = models.resnet18(weights=True).cuda()


# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# Establecer en Eval y mirar las capas del modelo
model.eval()
'''=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            9,408
├─BatchNorm2d: 1-2                       128
├─ReLU: 1-3                              --
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─BasicBlock: 2-1                   --
|    |    └─Conv2d: 3-1                  36,864
|    |    └─BatchNorm2d: 3-2             128
|    |    └─ReLU: 3-3                    --
|    |    └─Conv2d: 3-4                  36,864
|    |    └─BatchNorm2d: 3-5             128
|    └─BasicBlock: 2-2                   --
|    |    └─Conv2d: 3-6                  36,864
|    |    └─BatchNorm2d: 3-7             128
|    |    └─ReLU: 3-8                    --
|    |    └─Conv2d: 3-9                  36,864
|    |    └─BatchNorm2d: 3-10            128
├─Sequential: 1-6                        --
|    └─BasicBlock: 2-3                   --
|    |    └─Conv2d: 3-11                 73,728
|    |    └─BatchNorm2d: 3-12            256
|    |    └─ReLU: 3-13                   --
|    |    └─Conv2d: 3-14                 147,456
|    |    └─BatchNorm2d: 3-15            256
|    |    └─Sequential: 3-16             8,448
|    └─BasicBlock: 2-4                   --
|    |    └─Conv2d: 3-17                 147,456
|    |    └─BatchNorm2d: 3-18            256
|    |    └─ReLU: 3-19                   --
|    |    └─Conv2d: 3-20                 147,456
|    |    └─BatchNorm2d: 3-21            256
├─Sequential: 1-7                        --
|    └─BasicBlock: 2-5                   --
|    |    └─Conv2d: 3-22                 294,912
|    |    └─BatchNorm2d: 3-23            512
|    |    └─ReLU: 3-24                   --
|    |    └─Conv2d: 3-25                 589,824
|    |    └─BatchNorm2d: 3-26            512
|    |    └─Sequential: 3-27             33,280
|    └─BasicBlock: 2-6                   --
|    |    └─Conv2d: 3-28                 589,824
|    |    └─BatchNorm2d: 3-29            512
|    |    └─ReLU: 3-30                   --
|    |    └─Conv2d: 3-31                 589,824
|    |    └─BatchNorm2d: 3-32            512
├─Sequential: 1-8                        --
|    └─BasicBlock: 2-7                   --
|    |    └─Conv2d: 3-33                 1,179,648
|    |    └─BatchNorm2d: 3-34            1,024
|    |    └─ReLU: 3-35                   --
|    |    └─Conv2d: 3-36                 2,359,296
|    |    └─BatchNorm2d: 3-37            1,024
|    |    └─Sequential: 3-38             132,096
|    └─BasicBlock: 2-8                   --
|    |    └─Conv2d: 3-39                 2,359,296
|    |    └─BatchNorm2d: 3-40            1,024
|    |    └─ReLU: 3-41                   --
|    |    └─Conv2d: 3-42                 2,359,296
|    |    └─BatchNorm2d: 3-43            1,024
├─AdaptiveAvgPool2d: 1-9                 --
├─Linear: 1-10                           513,000
=================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
================================================================='''

images = get_images()
predict_image(images, class_names)


# *******************************************+
# # **3. Inicio de carga inception**

import torchvision.models as models

model = models.inception_v3(weights=True).cuda()


# Mostrar parámetros del modelo
from torchsummary import summary 

# Tenga en cuenta que se espera el tamaño de entrada de diferencia con Inception
summary(model, input_size = (3,299,299))

# Establecer en Eval y mirar las capas del modelo
model.eval()
'''=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─BasicConv2d: 1-1                       --
|    └─Conv2d: 2-1                       864
|    └─BatchNorm2d: 2-2                  64
├─BasicConv2d: 1-2                       --
|    └─Conv2d: 2-3                       9,216
|    └─BatchNorm2d: 2-4                  64
├─BasicConv2d: 1-3                       --
|    └─Conv2d: 2-5                       18,432
|    └─BatchNorm2d: 2-6                  128
├─MaxPool2d: 1-4                         --
├─BasicConv2d: 1-5                       --
|    └─Conv2d: 2-7                       5,120
|    └─BatchNorm2d: 2-8                  160
├─BasicConv2d: 1-6                       --
|    └─Conv2d: 2-9                       138,240
|    └─BatchNorm2d: 2-10                 384
├─MaxPool2d: 1-7                         --
├─InceptionA: 1-8                        --
|    └─BasicConv2d: 2-11                 --
|    |    └─Conv2d: 3-1                  12,288
|    |    └─BatchNorm2d: 3-2             128
|    └─BasicConv2d: 2-12                 --
|    |    └─Conv2d: 3-3                  9,216
|    |    └─BatchNorm2d: 3-4             96
|    └─BasicConv2d: 2-13                 --
|    |    └─Conv2d: 3-5                  76,800
|    |    └─BatchNorm2d: 3-6             128
|    └─BasicConv2d: 2-14                 --
|    |    └─Conv2d: 3-7                  12,288
|    |    └─BatchNorm2d: 3-8             128
|    └─BasicConv2d: 2-15                 --
|    |    └─Conv2d: 3-9                  55,296
|    |    └─BatchNorm2d: 3-10            192
|    └─BasicConv2d: 2-16                 --
|    |    └─Conv2d: 3-11                 82,944
|    |    └─BatchNorm2d: 3-12            192
|    └─BasicConv2d: 2-17                 --
|    |    └─Conv2d: 3-13                 6,144
|    |    └─BatchNorm2d: 3-14            64
├─InceptionA: 1-9                        --
|    └─BasicConv2d: 2-18                 --
|    |    └─Conv2d: 3-15                 16,384
|    |    └─BatchNorm2d: 3-16            128
|    └─BasicConv2d: 2-19                 --
|    |    └─Conv2d: 3-17                 12,288
|    |    └─BatchNorm2d: 3-18            96
|    └─BasicConv2d: 2-20                 --
|    |    └─Conv2d: 3-19                 76,800
|    |    └─BatchNorm2d: 3-20            128
|    └─BasicConv2d: 2-21                 --
|    |    └─Conv2d: 3-21                 16,384
|    |    └─BatchNorm2d: 3-22            128
|    └─BasicConv2d: 2-22                 --
|    |    └─Conv2d: 3-23                 55,296
|    |    └─BatchNorm2d: 3-24            192
|    └─BasicConv2d: 2-23                 --
|    |    └─Conv2d: 3-25                 82,944
|    |    └─BatchNorm2d: 3-26            192
|    └─BasicConv2d: 2-24                 --
|    |    └─Conv2d: 3-27                 16,384
|    |    └─BatchNorm2d: 3-28            128
├─InceptionA: 1-10                       --
|    └─BasicConv2d: 2-25                 --
|    |    └─Conv2d: 3-29                 18,432
|    |    └─BatchNorm2d: 3-30            128
|    └─BasicConv2d: 2-26                 --
|    |    └─Conv2d: 3-31                 13,824
|    |    └─BatchNorm2d: 3-32            96
|    └─BasicConv2d: 2-27                 --
|    |    └─Conv2d: 3-33                 76,800
|    |    └─BatchNorm2d: 3-34            128
|    └─BasicConv2d: 2-28                 --
|    |    └─Conv2d: 3-35                 18,432
|    |    └─BatchNorm2d: 3-36            128
|    └─BasicConv2d: 2-29                 --
|    |    └─Conv2d: 3-37                 55,296
|    |    └─BatchNorm2d: 3-38            192
|    └─BasicConv2d: 2-30                 --
|    |    └─Conv2d: 3-39                 82,944
|    |    └─BatchNorm2d: 3-40            192
|    └─BasicConv2d: 2-31                 --
|    |    └─Conv2d: 3-41                 18,432
|    |    └─BatchNorm2d: 3-42            128
├─InceptionB: 1-11                       --
|    └─BasicConv2d: 2-32                 --
|    |    └─Conv2d: 3-43                 995,328
|    |    └─BatchNorm2d: 3-44            768
|    └─BasicConv2d: 2-33                 --
|    |    └─Conv2d: 3-45                 18,432
|    |    └─BatchNorm2d: 3-46            128
|    └─BasicConv2d: 2-34                 --
|    |    └─Conv2d: 3-47                 55,296
|    |    └─BatchNorm2d: 3-48            192
|    └─BasicConv2d: 2-35                 --
|    |    └─Conv2d: 3-49                 82,944
|    |    └─BatchNorm2d: 3-50            192
├─InceptionC: 1-12                       --
|    └─BasicConv2d: 2-36                 --
|    |    └─Conv2d: 3-51                 147,456
|    |    └─BatchNorm2d: 3-52            384
|    └─BasicConv2d: 2-37                 --
|    |    └─Conv2d: 3-53                 98,304
|    |    └─BatchNorm2d: 3-54            256
|    └─BasicConv2d: 2-38                 --
|    |    └─Conv2d: 3-55                 114,688
|    |    └─BatchNorm2d: 3-56            256
|    └─BasicConv2d: 2-39                 --
|    |    └─Conv2d: 3-57                 172,032
|    |    └─BatchNorm2d: 3-58            384
|    └─BasicConv2d: 2-40                 --
|    |    └─Conv2d: 3-59                 98,304
|    |    └─BatchNorm2d: 3-60            256
|    └─BasicConv2d: 2-41                 --
|    |    └─Conv2d: 3-61                 114,688
|    |    └─BatchNorm2d: 3-62            256
|    └─BasicConv2d: 2-42                 --
|    |    └─Conv2d: 3-63                 114,688
|    |    └─BatchNorm2d: 3-64            256
|    └─BasicConv2d: 2-43                 --
|    |    └─Conv2d: 3-65                 114,688
|    |    └─BatchNorm2d: 3-66            256
|    └─BasicConv2d: 2-44                 --
|    |    └─Conv2d: 3-67                 172,032
|    |    └─BatchNorm2d: 3-68            384
|    └─BasicConv2d: 2-45                 --
|    |    └─Conv2d: 3-69                 147,456
|    |    └─BatchNorm2d: 3-70            384
├─InceptionC: 1-13                       --
|    └─BasicConv2d: 2-46                 --
|    |    └─Conv2d: 3-71                 147,456
|    |    └─BatchNorm2d: 3-72            384
|    └─BasicConv2d: 2-47                 --
|    |    └─Conv2d: 3-73                 122,880
|    |    └─BatchNorm2d: 3-74            320
|    └─BasicConv2d: 2-48                 --
|    |    └─Conv2d: 3-75                 179,200
|    |    └─BatchNorm2d: 3-76            320
|    └─BasicConv2d: 2-49                 --
|    |    └─Conv2d: 3-77                 215,040
|    |    └─BatchNorm2d: 3-78            384
|    └─BasicConv2d: 2-50                 --
|    |    └─Conv2d: 3-79                 122,880
|    |    └─BatchNorm2d: 3-80            320
|    └─BasicConv2d: 2-51                 --
|    |    └─Conv2d: 3-81                 179,200
|    |    └─BatchNorm2d: 3-82            320
|    └─BasicConv2d: 2-52                 --
|    |    └─Conv2d: 3-83                 179,200
|    |    └─BatchNorm2d: 3-84            320
|    └─BasicConv2d: 2-53                 --
|    |    └─Conv2d: 3-85                 179,200
|    |    └─BatchNorm2d: 3-86            320
|    └─BasicConv2d: 2-54                 --
|    |    └─Conv2d: 3-87                 215,040
|    |    └─BatchNorm2d: 3-88            384
|    └─BasicConv2d: 2-55                 --
|    |    └─Conv2d: 3-89                 147,456
|    |    └─BatchNorm2d: 3-90            384
├─InceptionC: 1-14                       --
|    └─BasicConv2d: 2-56                 --
|    |    └─Conv2d: 3-91                 147,456
|    |    └─BatchNorm2d: 3-92            384
|    └─BasicConv2d: 2-57                 --
|    |    └─Conv2d: 3-93                 122,880
|    |    └─BatchNorm2d: 3-94            320
|    └─BasicConv2d: 2-58                 --
|    |    └─Conv2d: 3-95                 179,200
|    |    └─BatchNorm2d: 3-96            320
|    └─BasicConv2d: 2-59                 --
|    |    └─Conv2d: 3-97                 215,040
|    |    └─BatchNorm2d: 3-98            384
|    └─BasicConv2d: 2-60                 --
|    |    └─Conv2d: 3-99                 122,880
|    |    └─BatchNorm2d: 3-100           320
|    └─BasicConv2d: 2-61                 --
|    |    └─Conv2d: 3-101                179,200
|    |    └─BatchNorm2d: 3-102           320
|    └─BasicConv2d: 2-62                 --
|    |    └─Conv2d: 3-103                179,200
|    |    └─BatchNorm2d: 3-104           320
|    └─BasicConv2d: 2-63                 --
|    |    └─Conv2d: 3-105                179,200
|    |    └─BatchNorm2d: 3-106           320
|    └─BasicConv2d: 2-64                 --
|    |    └─Conv2d: 3-107                215,040
|    |    └─BatchNorm2d: 3-108           384
|    └─BasicConv2d: 2-65                 --
|    |    └─Conv2d: 3-109                147,456
|    |    └─BatchNorm2d: 3-110           384
├─InceptionC: 1-15                       --
|    └─BasicConv2d: 2-66                 --
|    |    └─Conv2d: 3-111                147,456
|    |    └─BatchNorm2d: 3-112           384
|    └─BasicConv2d: 2-67                 --
|    |    └─Conv2d: 3-113                147,456
|    |    └─BatchNorm2d: 3-114           384
|    └─BasicConv2d: 2-68                 --
|    |    └─Conv2d: 3-115                258,048
|    |    └─BatchNorm2d: 3-116           384
|    └─BasicConv2d: 2-69                 --
|    |    └─Conv2d: 3-117                258,048
|    |    └─BatchNorm2d: 3-118           384
|    └─BasicConv2d: 2-70                 --
|    |    └─Conv2d: 3-119                147,456
|    |    └─BatchNorm2d: 3-120           384
|    └─BasicConv2d: 2-71                 --
|    |    └─Conv2d: 3-121                258,048
|    |    └─BatchNorm2d: 3-122           384
|    └─BasicConv2d: 2-72                 --
|    |    └─Conv2d: 3-123                258,048
|    |    └─BatchNorm2d: 3-124           384
|    └─BasicConv2d: 2-73                 --
|    |    └─Conv2d: 3-125                258,048
|    |    └─BatchNorm2d: 3-126           384
|    └─BasicConv2d: 2-74                 --
|    |    └─Conv2d: 3-127                258,048
|    |    └─BatchNorm2d: 3-128           384
|    └─BasicConv2d: 2-75                 --
|    |    └─Conv2d: 3-129                147,456
|    |    └─BatchNorm2d: 3-130           384
├─InceptionAux: 1-16                     --
|    └─BasicConv2d: 2-76                 --
|    |    └─Conv2d: 3-131                98,304
|    |    └─BatchNorm2d: 3-132           256
|    └─BasicConv2d: 2-77                 --
|    |    └─Conv2d: 3-133                2,457,600
|    |    └─BatchNorm2d: 3-134           1,536
|    └─Linear: 2-78                      769,000
├─InceptionD: 1-17                       --
|    └─BasicConv2d: 2-79                 --
|    |    └─Conv2d: 3-135                147,456
|    |    └─BatchNorm2d: 3-136           384
|    └─BasicConv2d: 2-80                 --
|    |    └─Conv2d: 3-137                552,960
|    |    └─BatchNorm2d: 3-138           640
|    └─BasicConv2d: 2-81                 --
|    |    └─Conv2d: 3-139                147,456
|    |    └─BatchNorm2d: 3-140           384
|    └─BasicConv2d: 2-82                 --
|    |    └─Conv2d: 3-141                258,048
|    |    └─BatchNorm2d: 3-142           384
|    └─BasicConv2d: 2-83                 --
|    |    └─Conv2d: 3-143                258,048
|    |    └─BatchNorm2d: 3-144           384
|    └─BasicConv2d: 2-84                 --
|    |    └─Conv2d: 3-145                331,776
|    |    └─BatchNorm2d: 3-146           384
├─InceptionE: 1-18                       --
|    └─BasicConv2d: 2-85                 --
|    |    └─Conv2d: 3-147                409,600
|    |    └─BatchNorm2d: 3-148           640
|    └─BasicConv2d: 2-86                 --
|    |    └─Conv2d: 3-149                491,520
|    |    └─BatchNorm2d: 3-150           768
|    └─BasicConv2d: 2-87                 --
|    |    └─Conv2d: 3-151                442,368
|    |    └─BatchNorm2d: 3-152           768
|    └─BasicConv2d: 2-88                 --
|    |    └─Conv2d: 3-153                442,368
|    |    └─BatchNorm2d: 3-154           768
|    └─BasicConv2d: 2-89                 --
|    |    └─Conv2d: 3-155                573,440
|    |    └─BatchNorm2d: 3-156           896
|    └─BasicConv2d: 2-90                 --
|    |    └─Conv2d: 3-157                1,548,288
|    |    └─BatchNorm2d: 3-158           768
|    └─BasicConv2d: 2-91                 --
|    |    └─Conv2d: 3-159                442,368
|    |    └─BatchNorm2d: 3-160           768
|    └─BasicConv2d: 2-92                 --
|    |    └─Conv2d: 3-161                442,368
|    |    └─BatchNorm2d: 3-162           768
|    └─BasicConv2d: 2-93                 --
|    |    └─Conv2d: 3-163                245,760
|    |    └─BatchNorm2d: 3-164           384
├─InceptionE: 1-19                       --
|    └─BasicConv2d: 2-94                 --
|    |    └─Conv2d: 3-165                655,360
|    |    └─BatchNorm2d: 3-166           640
|    └─BasicConv2d: 2-95                 --
|    |    └─Conv2d: 3-167                786,432
|    |    └─BatchNorm2d: 3-168           768
|    └─BasicConv2d: 2-96                 --
|    |    └─Conv2d: 3-169                442,368
|    |    └─BatchNorm2d: 3-170           768
|    └─BasicConv2d: 2-97                 --
|    |    └─Conv2d: 3-171                442,368
|    |    └─BatchNorm2d: 3-172           768
|    └─BasicConv2d: 2-98                 --
|    |    └─Conv2d: 3-173                917,504
|    |    └─BatchNorm2d: 3-174           896
|    └─BasicConv2d: 2-99                 --
|    |    └─Conv2d: 3-175                1,548,288
|    |    └─BatchNorm2d: 3-176           768
|    └─BasicConv2d: 2-100                --
|    |    └─Conv2d: 3-177                442,368
|    |    └─BatchNorm2d: 3-178           768
|    └─BasicConv2d: 2-101                --
|    |    └─Conv2d: 3-179                442,368
|    |    └─BatchNorm2d: 3-180           768
|    └─BasicConv2d: 2-102                --
|    |    └─Conv2d: 3-181                393,216
|    |    └─BatchNorm2d: 3-182           384
├─AdaptiveAvgPool2d: 1-20                --
├─Dropout: 1-21                          --
├─Linear: 1-22                           2,049,000
=================================================================
Total params: 27,161,264
Trainable params: 27,161,264
Non-trainable params: 0
================================================================='''

images = get_images()
predict_image(images, class_names)

# *******************************************+
# # **4. Cargando MobileNet**
import torchvision.models as models

model = models.mobilenet_v2(pretrained=True).cuda()

# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))

# Establecer en Eval y mirar las capas del modelo
model.eval()
'''===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
├─Sequential: 1-1                                  --
|    └─Conv2dNormActivation: 2-1                   --
|    |    └─Conv2d: 3-1                            864
|    |    └─BatchNorm2d: 3-2                       64
|    |    └─ReLU6: 3-3                             --
|    └─InvertedResidual: 2-2                       --
|    |    └─Sequential: 3-4                        896
|    └─InvertedResidual: 2-3                       --
|    |    └─Sequential: 3-5                        5,136
|    └─InvertedResidual: 2-4                       --
|    |    └─Sequential: 3-6                        8,832
|    └─InvertedResidual: 2-5                       --
|    |    └─Sequential: 3-7                        10,000
|    └─InvertedResidual: 2-6                       --
|    |    └─Sequential: 3-8                        14,848
|    └─InvertedResidual: 2-7                       --
|    |    └─Sequential: 3-9                        14,848
|    └─InvertedResidual: 2-8                       --
|    |    └─Sequential: 3-10                       21,056
|    └─InvertedResidual: 2-9                       --
|    |    └─Sequential: 3-11                       54,272
|    └─InvertedResidual: 2-10                      --
|    |    └─Sequential: 3-12                       54,272
|    └─InvertedResidual: 2-11                      --
|    |    └─Sequential: 3-13                       54,272
|    └─InvertedResidual: 2-12                      --
|    |    └─Sequential: 3-14                       66,624
|    └─InvertedResidual: 2-13                      --
|    |    └─Sequential: 3-15                       118,272
|    └─InvertedResidual: 2-14                      --
|    |    └─Sequential: 3-16                       118,272
|    └─InvertedResidual: 2-15                      --
|    |    └─Sequential: 3-17                       155,264
|    └─InvertedResidual: 2-16                      --
|    |    └─Sequential: 3-18                       320,000
|    └─InvertedResidual: 2-17                      --
|    |    └─Sequential: 3-19                       320,000
|    └─InvertedResidual: 2-18                      --
|    |    └─Sequential: 3-20                       473,920
|    └─Conv2dNormActivation: 2-19                  --
|    |    └─Conv2d: 3-21                           409,600
|    |    └─BatchNorm2d: 3-22                      2,560
|    |    └─ReLU6: 3-23                            --
├─Sequential: 1-2                                  --
|    └─Dropout: 2-20                               --
|    └─Linear: 2-21                                1,281,000
===========================================================================
Total params: 3,504,872
Trainable params: 3,504,872
Non-trainable params: 0
==========================================================================='''


images = get_images()
predict_image(images, class_names)


# *******************************************+
# # **5. Cargando SqueezeNet**

import torchvision.models as models

model = models.squeezenet1_0(weights=True).cuda()

# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))

# Establecer en Eval y mirar las capas del modelo
model.eval()
'''=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       14,208
|    └─ReLU: 2-2                         --
|    └─MaxPool2d: 2-3                    --
|    └─Fire: 2-4                         --
|    |    └─Conv2d: 3-1                  1,552
|    |    └─ReLU: 3-2                    --
|    |    └─Conv2d: 3-3                  1,088
|    |    └─ReLU: 3-4                    --
|    |    └─Conv2d: 3-5                  9,280
|    |    └─ReLU: 3-6                    --
|    └─Fire: 2-5                         --
|    |    └─Conv2d: 3-7                  2,064
|    |    └─ReLU: 3-8                    --
|    |    └─Conv2d: 3-9                  1,088
|    |    └─ReLU: 3-10                   --
|    |    └─Conv2d: 3-11                 9,280
|    |    └─ReLU: 3-12                   --
|    └─Fire: 2-6                         --
|    |    └─Conv2d: 3-13                 4,128
|    |    └─ReLU: 3-14                   --
|    |    └─Conv2d: 3-15                 4,224
|    |    └─ReLU: 3-16                   --
|    |    └─Conv2d: 3-17                 36,992
|    |    └─ReLU: 3-18                   --
|    └─MaxPool2d: 2-7                    --
|    └─Fire: 2-8                         --
|    |    └─Conv2d: 3-19                 8,224
|    |    └─ReLU: 3-20                   --
|    |    └─Conv2d: 3-21                 4,224
|    |    └─ReLU: 3-22                   --
|    |    └─Conv2d: 3-23                 36,992
|    |    └─ReLU: 3-24                   --
|    └─Fire: 2-9                         --
|    |    └─Conv2d: 3-25                 12,336
|    |    └─ReLU: 3-26                   --
|    |    └─Conv2d: 3-27                 9,408
|    |    └─ReLU: 3-28                   --
|    |    └─Conv2d: 3-29                 83,136
|    |    └─ReLU: 3-30                   --
|    └─Fire: 2-10                        --
|    |    └─Conv2d: 3-31                 18,480
|    |    └─ReLU: 3-32                   --
|    |    └─Conv2d: 3-33                 9,408
|    |    └─ReLU: 3-34                   --
|    |    └─Conv2d: 3-35                 83,136
|    |    └─ReLU: 3-36                   --
|    └─Fire: 2-11                        --
|    |    └─Conv2d: 3-37                 24,640
|    |    └─ReLU: 3-38                   --
|    |    └─Conv2d: 3-39                 16,640
|    |    └─ReLU: 3-40                   --
|    |    └─Conv2d: 3-41                 147,712
|    |    └─ReLU: 3-42                   --
|    └─MaxPool2d: 2-12                   --
|    └─Fire: 2-13                        --
|    |    └─Conv2d: 3-43                 32,832
|    |    └─ReLU: 3-44                   --
|    |    └─Conv2d: 3-45                 16,640
|    |    └─ReLU: 3-46                   --
|    |    └─Conv2d: 3-47                 147,712
|    |    └─ReLU: 3-48                   --
├─Sequential: 1-2                        --
|    └─Dropout: 2-14                     --
|    └─Conv2d: 2-15                      513,000
|    └─ReLU: 2-16                        --
|    └─AdaptiveAvgPool2d: 2-17           --
=================================================================
Total params: 1,248,424
Trainable params: 1,248,424
Non-trainable params: 0
================================================================='''

images = get_images()
predict_image(images, class_names)


# *******************************************+
# # **6. Carga amplia ResNet**


import torchvision.models as models

model = models.wide_resnet50_2(weights=True).cuda()

# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# Establecer en Eval y mirar las capas del modelo
model.eval()
'''=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            9,408
├─BatchNorm2d: 1-2                       128
├─ReLU: 1-3                              --
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─Bottleneck: 2-1                   --
|    |    └─Conv2d: 3-1                  8,192
|    |    └─BatchNorm2d: 3-2             256
|    |    └─Conv2d: 3-3                  147,456
|    |    └─BatchNorm2d: 3-4             256
|    |    └─Conv2d: 3-5                  32,768
|    |    └─BatchNorm2d: 3-6             512
|    |    └─ReLU: 3-7                    --
|    |    └─Sequential: 3-8              16,896
|    └─Bottleneck: 2-2                   --
|    |    └─Conv2d: 3-9                  32,768
|    |    └─BatchNorm2d: 3-10            256
|    |    └─Conv2d: 3-11                 147,456
|    |    └─BatchNorm2d: 3-12            256
|    |    └─Conv2d: 3-13                 32,768
|    |    └─BatchNorm2d: 3-14            512
|    |    └─ReLU: 3-15                   --
|    └─Bottleneck: 2-3                   --
|    |    └─Conv2d: 3-16                 32,768
|    |    └─BatchNorm2d: 3-17            256
|    |    └─Conv2d: 3-18                 147,456
|    |    └─BatchNorm2d: 3-19            256
|    |    └─Conv2d: 3-20                 32,768
|    |    └─BatchNorm2d: 3-21            512
|    |    └─ReLU: 3-22                   --
├─Sequential: 1-6                        --
|    └─Bottleneck: 2-4                   --
|    |    └─Conv2d: 3-23                 65,536
|    |    └─BatchNorm2d: 3-24            512
|    |    └─Conv2d: 3-25                 589,824
|    |    └─BatchNorm2d: 3-26            512
|    |    └─Conv2d: 3-27                 131,072
|    |    └─BatchNorm2d: 3-28            1,024
|    |    └─ReLU: 3-29                   --
|    |    └─Sequential: 3-30             132,096
|    └─Bottleneck: 2-5                   --
|    |    └─Conv2d: 3-31                 131,072
|    |    └─BatchNorm2d: 3-32            512
|    |    └─Conv2d: 3-33                 589,824
|    |    └─BatchNorm2d: 3-34            512
|    |    └─Conv2d: 3-35                 131,072
|    |    └─BatchNorm2d: 3-36            1,024
|    |    └─ReLU: 3-37                   --
|    └─Bottleneck: 2-6                   --
|    |    └─Conv2d: 3-38                 131,072
|    |    └─BatchNorm2d: 3-39            512
|    |    └─Conv2d: 3-40                 589,824
|    |    └─BatchNorm2d: 3-41            512
|    |    └─Conv2d: 3-42                 131,072
|    |    └─BatchNorm2d: 3-43            1,024
|    |    └─ReLU: 3-44                   --
|    └─Bottleneck: 2-7                   --
|    |    └─Conv2d: 3-45                 131,072
|    |    └─BatchNorm2d: 3-46            512
|    |    └─Conv2d: 3-47                 589,824
|    |    └─BatchNorm2d: 3-48            512
|    |    └─Conv2d: 3-49                 131,072
|    |    └─BatchNorm2d: 3-50            1,024
|    |    └─ReLU: 3-51                   --
├─Sequential: 1-7                        --
|    └─Bottleneck: 2-8                   --
|    |    └─Conv2d: 3-52                 262,144
|    |    └─BatchNorm2d: 3-53            1,024
|    |    └─Conv2d: 3-54                 2,359,296
|    |    └─BatchNorm2d: 3-55            1,024
|    |    └─Conv2d: 3-56                 524,288
|    |    └─BatchNorm2d: 3-57            2,048
|    |    └─ReLU: 3-58                   --
|    |    └─Sequential: 3-59             526,336
|    └─Bottleneck: 2-9                   --
|    |    └─Conv2d: 3-60                 524,288
|    |    └─BatchNorm2d: 3-61            1,024
|    |    └─Conv2d: 3-62                 2,359,296
|    |    └─BatchNorm2d: 3-63            1,024
|    |    └─Conv2d: 3-64                 524,288
|    |    └─BatchNorm2d: 3-65            2,048
|    |    └─ReLU: 3-66                   --
|    └─Bottleneck: 2-10                  --
|    |    └─Conv2d: 3-67                 524,288
|    |    └─BatchNorm2d: 3-68            1,024
|    |    └─Conv2d: 3-69                 2,359,296
|    |    └─BatchNorm2d: 3-70            1,024
|    |    └─Conv2d: 3-71                 524,288
|    |    └─BatchNorm2d: 3-72            2,048
|    |    └─ReLU: 3-73                   --
|    └─Bottleneck: 2-11                  --
|    |    └─Conv2d: 3-74                 524,288
|    |    └─BatchNorm2d: 3-75            1,024
|    |    └─Conv2d: 3-76                 2,359,296
|    |    └─BatchNorm2d: 3-77            1,024
|    |    └─Conv2d: 3-78                 524,288
|    |    └─BatchNorm2d: 3-79            2,048
|    |    └─ReLU: 3-80                   --
|    └─Bottleneck: 2-12                  --
|    |    └─Conv2d: 3-81                 524,288
|    |    └─BatchNorm2d: 3-82            1,024
|    |    └─Conv2d: 3-83                 2,359,296
|    |    └─BatchNorm2d: 3-84            1,024
|    |    └─Conv2d: 3-85                 524,288
|    |    └─BatchNorm2d: 3-86            2,048
|    |    └─ReLU: 3-87                   --
|    └─Bottleneck: 2-13                  --
|    |    └─Conv2d: 3-88                 524,288
|    |    └─BatchNorm2d: 3-89            1,024
|    |    └─Conv2d: 3-90                 2,359,296
|    |    └─BatchNorm2d: 3-91            1,024
|    |    └─Conv2d: 3-92                 524,288
|    |    └─BatchNorm2d: 3-93            2,048
|    |    └─ReLU: 3-94                   --
├─Sequential: 1-8                        --
|    └─Bottleneck: 2-14                  --
|    |    └─Conv2d: 3-95                 1,048,576
|    |    └─BatchNorm2d: 3-96            2,048
|    |    └─Conv2d: 3-97                 9,437,184
|    |    └─BatchNorm2d: 3-98            2,048
|    |    └─Conv2d: 3-99                 2,097,152
|    |    └─BatchNorm2d: 3-100           4,096
|    |    └─ReLU: 3-101                  --
|    |    └─Sequential: 3-102            2,101,248
|    └─Bottleneck: 2-15                  --
|    |    └─Conv2d: 3-103                2,097,152
|    |    └─BatchNorm2d: 3-104           2,048
|    |    └─Conv2d: 3-105                9,437,184
|    |    └─BatchNorm2d: 3-106           2,048
|    |    └─Conv2d: 3-107                2,097,152
|    |    └─BatchNorm2d: 3-108           4,096
|    |    └─ReLU: 3-109                  --
|    └─Bottleneck: 2-16                  --
|    |    └─Conv2d: 3-110                2,097,152
|    |    └─BatchNorm2d: 3-111           2,048
|    |    └─Conv2d: 3-112                9,437,184
|    |    └─BatchNorm2d: 3-113           2,048
|    |    └─Conv2d: 3-114                2,097,152
|    |    └─BatchNorm2d: 3-115           4,096
|    |    └─ReLU: 3-116                  --
├─AdaptiveAvgPool2d: 1-9                 --
├─Linear: 1-10                           2,049,000
=================================================================
Total params: 68,883,240
Trainable params: 68,883,240
Non-trainable params: 0
================================================================='''

images = get_images()
predict_image(images, class_names)

# *******************************************+
# # **7. Cargando Wide MNASNet**


import torchvision.models as models

model = models.mnasnet1_0(weights=True).cuda()



# Mostrar parámetros del modelo
from torchsummary import summary 

summary(model, input_size = (3,224,224))



# Establecer en Eval y mirar las capas del modelo
model.eval()
'''=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       864
|    └─BatchNorm2d: 2-2                  64
|    └─ReLU: 2-3                         --
|    └─Conv2d: 2-4                       288
|    └─BatchNorm2d: 2-5                  64
|    └─ReLU: 2-6                         --
|    └─Conv2d: 2-7                       512
|    └─BatchNorm2d: 2-8                  32
|    └─Sequential: 2-9                   --
|    |    └─_InvertedResidual: 3-1       2,592
|    |    └─_InvertedResidual: 3-2       4,440
|    |    └─_InvertedResidual: 3-3       4,440
|    └─Sequential: 2-10                  --
|    |    └─_InvertedResidual: 3-4       6,776
|    |    └─_InvertedResidual: 3-5       13,160
|    |    └─_InvertedResidual: 3-6       13,160
|    └─Sequential: 2-11                  --
|    |    └─_InvertedResidual: 3-7       35,920
|    |    └─_InvertedResidual: 3-8       90,880
|    |    └─_InvertedResidual: 3-9       90,880
|    └─Sequential: 2-12                  --
|    |    └─_InvertedResidual: 3-10      90,912
|    |    └─_InvertedResidual: 3-11      118,272
|    └─Sequential: 2-13                  --
|    |    └─_InvertedResidual: 3-12      182,976
|    |    └─_InvertedResidual: 3-13      476,160
|    |    └─_InvertedResidual: 3-14      476,160
|    |    └─_InvertedResidual: 3-15      476,160
|    └─Sequential: 2-14                  --
|    |    └─_InvertedResidual: 3-16      605,440
|    └─Conv2d: 2-15                      409,600
|    └─BatchNorm2d: 2-16                 2,560
|    └─ReLU: 2-17                        --
├─Sequential: 1-2                        --
|    └─Dropout: 2-18                     --
|    └─Linear: 2-19                      1,281,000
=================================================================
Total params: 4,383,312
Trainable params: 4,383,312
Non-trainable params: 0
=================================================================
'''

images = get_images()
predict_image(images, class_names)





