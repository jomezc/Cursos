#!/usr/bin/env python
# codificación: utf-8

# # **Transferencia de estilo neuronal en PyTorch**
#
# En esta lección, primero aprenderemos a implementar el **Algoritmo de transferencia de estilo neuronal** con PyTorch.
#
# Aplicamos la técnica conocida como *transferencia de estilo neuronal* que se muestra en la investigación publicada aquí
# <a href="https://arxiv.org/abs/1508.06576" class="external">Un algoritmo neuronal de estilo artístico</a > (Gatys et al.).
#
# En este tutorial demostramos el algoritmo de transferencia de estilo original. Optimiza el contenido de la imagen a
# un estilo particular. Los enfoques modernos entrenan un modelo para generar la imagen estilizada directamente
# (similar a [cyclegan](cyclegan.ipynb)). Este enfoque es mucho más rápido (hasta 1000x).
#
# 1. Importación de paquetes y selección de un dispositivo
# 2. Cargando las Imágenes
# 3. Funciones de pérdida
# 4. Pérdida de estilo y matriz Gram
# 5. Importación del modelo
# 6. Descenso de gradiente
# 7. Algoritmo de ejecución

# # **Principio subyacente**
# --------------------
#
'''
El principio es sencillo: definimos dos distancias, una para el contenido (DC) y otra para el estilo (DS).
La DC mide la diferencia de contenido entre dos imágenes, mientras que la DS mide la diferencia de estilo entre dos
imágenes. A continuación, tomamos una tercera imagen, la de entrada, y la transformamos para minimizar tanto su
distancia de contenido con la imagen de contenido como su distancia de estilo con la imagen de estilo. Ahora podemos
importar los paquetes necesarios y comenzar la transferencia neuronal.

Traducción realizada con la versión gratuita del traductor www.DeepL.com/Translator'''
#
### **1. Importación de paquetes y selección de un dispositivo**
#
# A continuación se muestra una lista de los paquetes necesarios para implementar la transferencia neuronal.
#
# - ``torch``, ``torch.nn``, ``numpy`` (paquetes indispensables para redes neuronales con PyTorch)
# - ``torch.optim`` (descensos de gradiente eficientes)
# - ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (cargar y mostrar imágenes)
# - ``torchvision.transforms`` (transforma imágenes PIL en tensores)
# - ``torchvision.models`` (entrenar o cargar modelos pre-entrenados)
# - ``copiar`` (para copiar en profundidad los modelos; paquete del sistema)

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# A continuación, debemos elegir en qué dispositivo ejecutar la red e importar el
# imágenes de contenido y estilo. Ejecutar el algoritmo de transferencia neuronal en grandes
# imágenes tardan más y se reproducirán mucho más rápido cuando se ejecutan en una GPU. Podemos
# use ``torch.cuda.is_available()`` para detectar si hay una GPU disponible.
# A continuación, configuramos ``torch.device`` para su uso a lo largo del tutorial. También el ``.to(device)``
# El método se usa para mover tensores o módulos a un dispositivo deseado.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

### **2. Cargando las imágenes**
#
# Ahora importaremos las imágenes de estilo y contenido. Las imágenes PIL originales tienen valores entre 0 y 255, pero
# cuando transformados en tensores, sus valores se convierten para estar entre
# 0 y 1. También es necesario cambiar el tamaño de las imágenes para que tengan las mismas dimensiones.
# Un detalle importante a tener en cuenta es que las redes neuronales del
# La biblioteca torch está entrenada con valores de tensor que van de 0 a 1. Si
# intente alimentar las redes con 0 a 255 imágenes de tensor, luego las activará
# los mapas de características no podrán detectar el contenido y el estilo previstos.
# Sin embargo, las redes preentrenadas de la biblioteca Caffe están entrenadas con 0
# a 255 imágenes de tensor.
#
#
# .. Nota::
# Aquí hay enlaces para descargar las imágenes necesarias para ejecutar el tutorial:
# `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ y
# `bailando.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
# Descargue estas dos imágenes y agréguelas a un directorio
# con el nombre ``images`` en su directorio de trabajo actual.

# tamaño deseado de la imagen de salida
imsize = 512 if torch.cuda.is_available() else 128  # usar tamaño pequeño si no hay gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # escalar imagen importada
    transforms.ToTensor()])  # transformarlo en un tensor de antorcha

# Parece que .unsqueeze(0) añade una dimensión falsa
# .squeeze(0) elimina dicha dimensión
def image_loader(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    # imagen = Imagen.open(nombre_imagen)
    # se requiere una dimensión de lote falsa para ajustarse a las dimensiones de entrada de la red
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = image_loader("https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg")
style_img = image_loader("https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


# Ahora, vamos a crear una función que muestre una imagen reconvirtiendo un
# copia en formato PIL y mostrar la copia usando  ``plt.imshow``. Intentaremos mostrar el contenido y las imágenes de
# estilo para asegurarse de que se importaron correctamente.


unloader = transforms.ToPILImage()  # reconvertir en imagen PIL

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clonamos el tensor para no hacerle cambios
    image = image.squeeze(0)      # Eliminar la dimensión del lote falso
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pausa un poco para que se actualicen las tramas


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


### **3. Funciones de pérdida**
#
# ### **Pérdida de contenido**
#
# La pérdida de contenido es una función que representa una versión ponderada de la
# distancia de contenido para una capa individual. La función toma la característica
# mapea $F_{XL}$ de una capa $L$ en una entrada de procesamiento de red $X$ y devuelve la
# distancia de contenido ponderado $w_{CL}.D_C^L(X,C)$ entre la imagen $X$ y la
# imagen de contenido $C$. Los mapas de características de la imagen de contenido ($F_{CL}$) deben ser
# conocido por la función para calcular la distancia del contenido. Nosotros
# implementar esta función como un módulo toch con un constructor que toma
# $F_{CL}$ como entrada. La distancia $\|F_{XL} - F_{CL}\|^2$ es el error cuadrático medio
# entre los dos conjuntos de mapas de características, y se puede calcular usando ``nn.MSELoss``.
#
# Agregaremos este módulo de pérdida de contenido directamente después de la convolución
# capa(s) que se utilizan para calcular la distancia del contenido. De esta manera
# cada vez que la red recibe una imagen de entrada, las pérdidas de contenido serán
# calculado en las capas deseadas y debido a la graduación automática, todos los
# Se calcularán gradientes.
# Ahora, para hacer que la capa de pérdida de contenido
# transparente debemos definir un método ``forward`` que calcule el contenido
# pérdida y luego devuelve la entrada de la capa. La pérdida calculada se guarda como
# parámetro del módulo.


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 'separamos' el contenido de destino del árbol utilizado
        # para calcular dinámicamente el gradiente: este es un valor establecido,
        # no es una variable. De lo contrario, el método directo del criterio
        # arrojará un error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


### **4. Matriz de pérdida de estilo y gramo**
#
# El módulo de pérdida de estilo se implementa de manera similar a la pérdida de contenido
# módulo. Actuará como una capa transparente en un
# red que calcula la pérdida de estilo de esa capa. Con el fin de
# calcular la pérdida de estilo, necesitamos calcular la **matriz de gramo** $G_{XL}$. un gramo
# matriz es el resultado de multiplicar una matriz dada por su traspuesta
# matriz. En esta aplicación, la matriz dada es una versión remodelada de
# la función mapea $F_{XL}$ de una capa $L$. $F_{XL}$ se reforma para formar $\hat{F}_{XL}$, un $K$\ x\ $N$
# matriz, donde $K$ es el número de mapas de características en la capa $L$ y $N$ es la
# longitud de cualquier mapa de características vectorizado $F_{XL}^k$. Por ejemplo, la primera línea
# de $\hat{F}_{XL}$ corresponde al primer mapa de características vectorizado $F_{XL}^1$.
#
# Finalmente, la matriz de Gram debe normalizarse dividiendo cada elemento por
# el número total de elementos en la matriz. Esta normalización es para
# contrarrestar el hecho de que las matrices $\hat{F}_{XL}$ con una gran dimensión $N$ producen
# valores más grandes en la matriz de Gram. Estos valores mayores harán que la
# primeras capas (antes de agrupar capas) para tener un mayor impacto durante el
# descenso de gradiente. Las características del estilo tienden a estar en las capas más profundas del
# red, por lo que este paso de normalización es crucial.
#

def gram_matrix(input):
    a, b, c, d = input.size()  # a=tamaño del lote(=1)
    # b=número de mapas de características
    # (c,d)=dimensiones de una f. mapa (N=c*d)

    features = input.view(a * b, c * d)  # cambia el tamaño de F_XL a \hat F_XL

    G = torch.mm(features, features.t())  # calcular el producto gramo

    # 'normalizamos' los valores de la matriz de gramos
    # dividiendo por el número de elementos en cada mapa de características.
    return G.div(a * b * c * d)


# Ahora el módulo de pérdida de estilo se ve casi exactamente igual que el módulo de pérdida de contenido
# módulo. La distancia de estilo también se calcula usando el cuadrado medio
# error entre $G_{XL}$ y $G_{SL}$.

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


### **5. Importación del modelo**
#
#
# Ahora necesitamos importar una red neuronal preentrenada. Usaremos un 19
# Red VGG de # capa como la utilizada en el artículo.
#
# La implementación de PyTorch de VGG es un módulo dividido en dos
# Módulos ``secuenciales``: ``features`` (que contienen capas de convolución y agrupación),
# y ``classifier`` (que contiene capas completamente conectadas). Usaremos el
# módulo ``features`` porque necesitamos la salida del individuo
# capas de convolución para medir la pérdida de contenido y estilo. Algunas capas tienen
# Comportamiento diferente durante el entrenamiento que la evaluación, por lo que debemos establecer el
# red al modo de evaluación usando ``.eval()``.


cnn = models.vgg19(pretrained=True).features.to(device).eval()


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# crear un módulo para normalizar la imagen de entrada para que podamos ponerlo fácilmente en un
# nn.Secuencial
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .ver la media y estándar para hacerlos [C x 1 x 1] para que puedan
        # trabajar directamente con imagen Tensor de forma [B x C x H x W].
        # B es el tamaño del lote. C es el número de canales. H es alto y W es ancho.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalizar imagen
        return (img - self.mean) / self.std


# Un módulo ``Sequential`` contiene una lista ordenada de módulos secundarios. Para
# instancia, ``vgg19.features`` contiene una secuencia (Conv2d, ReLU, MaxPool2d,
# Conv2d, ReLU…) alineados en el orden correcto de profundidad. Necesitamos agregar nuestro
# capas de pérdida de contenido y pérdida de estilo inmediatamente después de la convolución
# capa que están detectando. Para ello debemos crear un nuevo ``Sequential``
# módulo que tiene módulos de pérdida de contenido y pérdida de estilo insertados correctamente.


# capas de profundidad deseadas para calcular las pérdidas de estilo/contenido:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # módulo de normalización
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # solo para tener un acceso iterable o una lista de contenido/estilo
    # pérdidas
    content_losses = []
    style_losses = []

    # suponiendo que cnn es un nn.Sequential, creamos un nuevo nn.Sequential
    # para poner módulos que se supone que deben activarse secuencialmente
    model = nn.Sequential(normalization)

    i = 0  # incremento cada vez que vemos una conversión
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # La versión in situ no funciona muy bien con ContentLoss
            # y StyleLoss insertamos a continuación. Así que reemplazamos con fuera de lugar
            # unos aquí.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # agregar pérdida de contenido:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # agregar pérdida de estilo:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # ahora recortamos las capas después de las últimas pérdidas de contenido y estilo
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# A continuación, seleccionamos la imagen de entrada. Puedes usar una copia de la imagen del contenido.
# o ruido blanco.
#
#

input_img = content_img.clone()
# si desea utilizar ruido blanco en su lugar, descomente la siguiente línea:
# input_img = torch.randn(content_img.data.size(), dispositivo=dispositivo)

# agregue la imagen de entrada original a la figura:
plt.figure()
imshow(input_img, title='Input Image')


### **6. Descenso de gradiente**
#
# Como sugirió Leon Gatys, el autor del algoritmo `aquí <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis -jacq>`__, usaremos
# Algoritmo L-BFGS para ejecutar nuestro descenso de gradiente. A diferencia de entrenar una red,
# queremos entrenar la imagen de entrada para minimizar el contenido/estilo
# pérdidas. Crearemos un optimizador PyTorch L-BFGS ``optim.LBFGS`` y pasaremos
# nuestra imagen como el tensor a optimizar.
#
#

def get_input_optimizer(input_img):
    # esta línea para mostrar que la entrada es un parámetro que requiere un gradiente
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# Finalmente, debemos definir una función que realice la transferencia neuronal. Para
# cada iteración de las redes, se alimenta con una entrada actualizada y calcula
# nuevas pérdidas. Ejecutaremos los métodos ``hacia atrás`` de cada módulo de pérdida para
# calcular dinámicamente sus gradientes. El optimizador requiere un “cierre”
# función, que reevalúa el módulo y devuelve la pérdida.
#
# Todavía tenemos una restricción final que abordar. La red puede intentar
# optimizar la entrada con valores que excedan el rango de tensor de 0 a 1 para
# la imagen. Podemos solucionar esto corrigiendo los valores de entrada para que sean
# entre 0 y 1 cada vez que se ejecuta la red.


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """ Ejecutar la transferencia de estilo."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # corregir los valores de la imagen de entrada actualizada
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # una última corrección...
    input_img.data.clamp_(0, 1)

    return input_img


### **7. Ejecutar algoritmo**


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()





