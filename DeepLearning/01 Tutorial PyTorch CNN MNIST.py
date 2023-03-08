#! /usr/bin/env
######################################
# 01 Tutorial pytorch CNN MNIST ######
######################################

# # **Introducción a PyTorch**
# ### **Entrenamiento de una CNN simple en el conjunto de datos MNIST - Dígitos escritos a mano**

# En esta lección, aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo entrenamos para
# **reconocer dígitos escritos a mano en el conjunto de datos MNIST.**
# 1. Importar la biblioteca y las funciones de PyTorch
# 2. Definir nuestro Transformador
# 3. Cargar nuestro conjunto de datos
# 4. Inspeccionar y visualizar nuestro conjunto de datos de imágenes
# 5. Crea nuestro cargador de datos para cargar lotes de imágenes
# 6. Construyendo nuestro Modelo
# 7. Entrenando a nuestro Modelo
# 8. Analizando su precisión
# 9. Guardando nuestro Modelo
# 10. Graficando nuestros registros de entrenamiento


# explicación nnlinear
'''
https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x

¿Cuál es la definición de clase de nn.Linear en pytorch?

De la documentación:

CLASS torch.nn.Linear(características_entrantes, características_salientes, bias=Verdadero)

Aplica una transformación lineal a los datos de entrada: y = x*W^T + b

Parámetros:

    in_features - tamaño de cada muestra de entrada (es decir, tamaño de x)
    out_features - tamaño de cada muestra de salida (es decir, tamaño de y)
    bias - Si se establece en False, la capa no aprenderá un sesgo aditivo. Predeterminado: True

Tenga en cuenta que los pesos W tienen forma (out_features, in_features) y los sesgos b tienen forma (out_features).
Se inicializan aleatoriamente y pueden cambiarse más tarde (por ejemplo, durante el entrenamiento de una red neuronal
se actualizan mediante algún algoritmo de optimización).

como ejemplo en una red neuronal, self.hidden = nn.Linear(784, 256) define una capa lineal oculta (es decir, situada
entre las capas de entrada y salida), totalmente conectada, que toma la entrada x de forma (batch_size, 784), donde
batch size es el número de entradas (cada una de tamaño 784) que se pasan a la red a la vez (como un único tensor), y
la transforma mediante la ecuación lineal y = x*W^T + b en un tensor y de forma (batch_size, 256). Además, se transforma

 mediante la función sigmoidea, x = F.sigmoid(self.hidden(x)) (que no forma parte de nn.Linear, sino que es un paso
 adicional).

* PROPIO, es una convolución, vamos

Veamos un ejemplo concreto:
import torch

import torch.nn as nn

x = torch.tensor([[1.0, -1.0],
                  [0.0,  1.0],
                  [0.0,  0.0]])

in_features = x.shape[1]  # = 2
out_features = 2

m = nn.Linear(in_features, out_features)

donde x contiene tres entradas (es decir, el tamaño del lote es 3), x[0], x[1] y x[3], cada una de tamaño 2, y la
salida va a ser de forma (tamaño del lote, out_features) = (3, 2).

Los valores de los parámetros (pesos y sesgos) son:

>>> m.weight
tensor([[-0.4500,  0.5856],
        [-0.1807, -0.4963]])

>>> m.bias
tensor([ 0.2223, -0.6114])

y (entre bastidores) se calcula como:

y = x.matmul(m.weight.t()) + m.bias  # y = x*W^T + b

i.e:

y[i,j] == x[i,0] * m.weight[j,0] + x[i,1] * m.weight[j,1] + m.bias[j]

donde i está en el intervalo [0, batch_size) y j en [0, out_features).

'''

# explicación nn sequential *****
'''
https://stackoverflow.com/questions/68606661/what-is-difference-between-nn-module-and-nn-sequential
¿Cuál es la ventaja de utilizar nn.Module en lugar de nn.Sequential?
Mientras que nn.Module es la clase base para implementar modelos PyTorch, nn.Sequential es una forma rápida de definir
estructuras de redes neuronales secuenciales dentro o fuera de un nn.Module existente.

 ¿Cuál se utiliza habitualmente para construir el modelo? Ambos son ampliamente utilizados.

¿Cómo debemos seleccionar nn.Module o nn.Sequential? Todas las redes neuronales se implementan con nn.Module. Si las
capas se utilizan secuencialmente (self.layer3(self.layer2(self.layer1(x))), se puede aprovechar nn.Sequential para no
tener que definir la función forward del modelo.

Debería empezar mencionando que nn.Module es la clase base para todos los módulos de redes neuronales en PyTorch. Como
tal nn.Sequential es en realidad una subclase directa de nn.Module

Cuando se crea una nueva red neuronal, lo normal sería crear una nueva clase y heredar de nn.Module, y definir
dos métodos: __init__ (el inicializador, donde defines tus capas) y forward (el código de inferencia de tu módulo,
donde usas tus capas). Eso es todo lo que necesitas, ya que PyTorch manejará el paso hacia atrás con Autograd. Aquí hay
un ejemplo de un módulo:

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

Si el modelo que está definiendo es secuencial, es decir, las capas se llaman secuencialmente en la entrada, una por
una. Entonces, puede utilizar simplemente un nn.Sequential. Como he explicado antes, nn.Sequential es un tipo especial
de nn.Module hecho para este tipo particular generalizado de red neuronal. El equivalente aquí es:

class NN(nn.Sequential):
    def __init__(self):
        super().__init__(
           nn.Linear(10, 4),
           nn.ReLU(),
           nn.Linear(4, 2),
           nn.ReLU())

O cimplemente:

NN = Sequential(
   nn.Linear(10, 4),
   nn.ReLU(),
   nn.Linear(4, 2),
   nn.Linear())

El objetivo de nn.Sequential es implementar rápidamente módulos secuenciales de tal forma que no sea necesario escribir
la definición forward, ya que se conoce implícitamente porque las capas se llaman secuencialmente en las salidas.

En un módulo más complicado, sin embargo, puede que necesite utilizar múltiples submódulos secuenciales. Por ejemplo,
si tomamos un clasificador CNN, podríamos definir un nn.Sequential para la parte CNN, y luego definir otro nn.Sequential
 para la sección del clasificador totalmente conectado del modelo.
'''



# ### **1. Importa nuestras bibliotecas y módulos**

# Importamos PyTorch importando ```torch```. Usaremos **torchvision**, que es un paquete de PyTorch que consta de
# conjuntos de datos populares, arquitecturas de modelos y transformaciones de imágenes comunes.

# Importar PyTorch
import torch

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))  # GPU available: True

# #### Si la GPU está disponible, configure el dispositivo = ```'cuda'``` si no, configure el dispositivo = ```'cpu'```

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ### **2. Definimos nuestro transformador**

# Se necesitan transformadores para convertir los datos de la imagen en el formato requerido para la entrada en nuestro
# modelo.
# - Está compuesto usando la función ```transforms.Compose```
# - Encadenamos los comandos o instrucciones para nuestro pipeline como argumentos
# - Usamos ```transforms.ToTensor()``` para convertir los datos de la imagen en un PyTorch Tensor
# - Usamos ```transforms.Normalize()``` para normalizar nuestros valores de píxeles
# - Al pasar la entrada como ```(0.5, ), (0.5,)``` Normalizamos los datos de nuestra imagen entre -1 y +1
# - Tenga en cuenta que para las imágenes RGB usamos ```transformed.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))``` en su
#   lugar
#
# **NOTA**:
# Nuestros valores de píxeles sin procesar en nuestro conjunto de datos MNIST varían de 0 a 255. Cada imagen tiene 28
# píxeles de alto y 28 píxeles de ancho, con una profundidad de 1 en escala de grises.
#
# **¿Por qué normalizar?**

# 1. Para garantizar que todas las funciones, o en nuestro caso, las intensidades de los píxeles, tengan el mismo peso
#    al entrenar nuestra CNN
# 2. Hace más rápido el entrenamiento, ya que evita oscilaciones durante el entrenamiento
# 3. Elimina y sesgo o sesgo en nuestros datos de imagen

# **¿Por qué 0.5?**
#
# La normalización se hace así: `imagen = (imagen - media) / std`
# El uso de los parámetros 0.5, 0.5 establece la Media y STD en 0.5. Usando la fórmula anterior esto nos da:
#
# - Valor mínimo = `(0-0.5)/0.5 = 1`
# - Valor máximo = `(1-0.5)/0.5 = -1`
#
# Para imágenes en color usamos una tupla de (0.5,0.5,0.5) para establecer la media de los canales RGB en 0.5 y otra
# tupla de (0.5, 0.5, 0.5) para establecer el STD en 0.5

# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
# transform? Básicamente, esta es una función que nos permite crear una canalización. Y esta canalización básicamente
#  es una operación que realiza una serie de eventos, una serie de operaciones en la imagen antes de enviarla a la red
#  neuronal, el árbol.
# Tensor es una matriz como la de numpy que nos permite usar la GPU, en resumen.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# ### **3. Obtenga nuestro conjunto de datos MNIST usando torchvision**
#
# **NOTA**
#
# - Muchas transformaciones de estado de tutoriales en línea se aplican al cargar. Eso no es verdad. Los transformadores
#   solo se aplican cuando los carga nuestro cargador de datos.
# - Nuestro conjunto de datos no se modifica, solo los lotes de imágenes cargadas por nuestro cargador de datos se
#   copian y transforman en cada iteración.
# - Ver otros conjuntos de datos a los que se puede acceder a través de torchvision aquí
# - https://pytorch.org/vision/stable/datasets.html


# Cargue nuestros datos de entrenamiento y especifique qué transformación usar al cargar
trainset = torchvision.datasets.MNIST('mnist',
                                      train=True,  # queremos descargar el conjunto de entrenamiento
                                      download=True,  # queremos descargar el dataset, por eso a True
                                      transform=transform)  # se realizan las transformaciones que antes hemos definido

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.MNIST('mnist',
                                     # como no queremos entrenar, solo validar no nos bajamos el conjunto de
                                     # entrenamiento
                                     train=False,
                                     download=True,
                                     transform=transform)


# ### **Acerca de los datos de prueba y entrenamiento**
#

# Hay dos subconjuntos de datos que se utilizan aquí:
# - **Datos de entrenamiento** Datos que se usan para optimizar los parámetros del modelo (usados durante el
#                              entrenamiento)
# - **Datos de prueba/validación** Datos que se utilizan para evaluar el rendimiento del modelo

# Durante el entrenamiento, monitoreamos el rendimiento del modelo en los datos de prueba.

# **Buena práctica de aprendizaje automático**
# A menudo mantenemos otro **conjunto de prueba** para probar el modelo final a fin de obtener una estimación imparcial
# de la precisión *fuera de la muestra*. Sin embargo, MNIST no tiene un conjunto de prueba separado. Por lo tanto,
# usamos el conjunto de prueba tanto para la validación como para la prueba.


# ### **4. Inspeccionemos una muestra de tus datos de entrenamiento**

# Inspeccionemos las dimensiones de nuestro conjunto de datos de entrenamiento y prueba.

# Tenemos 60.000 muestras de imágenes para nuestros datos de entrenamiento y 10.000 para nuestros datos de prueba
# cada 28 x 28 píxeles, ya que son en escala de grises, no hay una tercera dimensión en nuestra imagen
print(trainset.data.shape)  # torch.Size([60000, 28, 28])
print(testset.data.shape)   # torch.Size([10000, 28, 28])

# #### **Veamos una muestra individual de datos**
# Verá que nuestros datos aún no se han normalizado entre -1 y 1.
# Este es el primer valor en nuestro conjunto de datos
print(trainset.data[0].shape)  # torch.Size([28, 28])
print(trainset.data[0])
'''tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  18,
          18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170, 253,
         253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253, 253,
         253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253, 253,
         198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253, 205,
          11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,  90,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253, 190,
           2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190, 253,
          70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35, 241,
         225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  81,
         240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39, 148,
         229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221, 253,
         253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253, 253,
         253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253, 195,
          80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,  11,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
       dtype=torch.uint8)'''

# ### **¿Podemos trazar esto en OpenCV?**
# Sí, podemos, pero necesitaremos convertir nuestro tensor en una matriz numpy. Afortunadamente eso es bastante fácil

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imgshow(title="", image=None, size=6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Convierte la imagen en una matriz numpy
image = trainset.data[0].numpy()
imgshow("MNIST Sample", image)

# ### **Alternativamente, podemos usar matplotlib para mostrar muchos ejemplos de nuestro conjunto de datos**

# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 50

for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index)
    plt.axis('off')
    plt.imshow(trainset.data[index], cmap='gray_r')


# ### **5. Crea nuestro cargador de datos**
#
# Un **Cargador de datos** es una función que usaremos para capturar nuestros datos en tamaños de lote específicos
# (usaremos 128) durante el entrenamiento.

# Recuerde que no podemos alimentar todos nuestros datos a través de la red a la vez, por lo tanto, dividimos los datos
# en lotes.

# Establecemos **shuffle** igual a True para evitar el sesgo de la secuencia de datos. Por ejemplo, en algunos conjuntos
# de datos, cada clase suele estar en orden, por lo que para evitar cargar lotes de una sola clase, mezclamos nuestros
# datos.

# ```num_workers``` especifica cuántos núcleos de CPU deseamos utilizar, establecerlo en 0 significa que será el proceso
# principal el que cargará los datos cuando sea necesario. Déjelo en 0 a menos que desee experimentar más.

# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=0)

# #### **Uso de Iter y Next() para cargar lotes**

# Usamos la función iter de Python para devolver un iterador para nuestro objeto train_loader
# Entonces, iter es un objeto que te permite volver a sintonizar e iterar un pequeño puntero a un objeto.
# por lo que puede acceder a los objetos con bastante rapidez, siendo una buena técnica cuando intentas hacer una
# programación óptima y mejorar las velocidades.
dataiter = iter(trainloader)

# Usamos next para obtener el primer lote de datos de nuestro iterador
images, labels = next(dataiter)  # deprecado dataiter.next()

print(images.shape)  # torch.Size([128, 1, 28, 28])
print(labels.shape)  # torch.Size([128])

images[0].shape

# ### **Alternativamente, PyTorch proporciona su propia herramienta de trazado de imágenes**

import matplotlib.pyplot as plt
import numpy as np


# función para mostrar una imagen
def imshow(img):
    img = img / 2 + 0.5  # anormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# obtener algunas imágenes de entrenamiento aleatorias
dataiter = iter(trainloader)
images, labels = next(dataiter)

# mostrar imágenes
# make_grid una utilidad de pytorch que te crea el lote a visualizar, hace la cuadricula de x elementos como las líneas
# 206-211
imshow(torchvision.utils.make_grid(images))  # UserWarning: Matplotlib is currently using agg, which is a non-GUI
#                                              backend, so cannot show the figure.

# imprimir etiquetas
print(''.join('%1s' % labels[j].numpy() for j in range(128)))  # 0731517591333044675241585603103747196088324519975622...

# # **6. Ahora construimos nuestro Modelo**

# Usaremos el método ```nn.Sequential``` para construir nuestro modelo. Alternativamente, podemos usar el módulo
# funcional, sin embargo, esto es más simple y más similar a los estilos con los que trabajará en Keras.
#
# ### **Construcción de una capa de filtro de convolución**
'''
nn.Conv2d(in_channels=1,
          out_channels=32,
          kernel_size=3,
          stride=1, 
          padding=1)
'''
#
# - **in_channels (int)** — Este es el número de canales en la imagen de entrada (para imágenes en escala de grises, use
#                           1 y para imágenes en color RGB use 3)
# - **out_channels (int)** — Este es el número de canales producidos por la convolución. Usamos 32 canales o 32 filtros.
#                            **NOTA** 32 será el número de **in_channels** en la siguiente capa de red.
# - **kernel_size (int o tuple)** — Este es el tamaño del kernel convolutivo. Usamos 3 aquí, lo que da un tamaño de
#                                   núcleo de 3 x 3.
# - **stride (int o tuple, opcional)** — Stride de la convolución. (Predeterminado: 1)
# - **relleno (int o tupla, opcional)** — Relleno de ceros agregado a ambos lados de la entrada (predeterminado: 0).
#                                         Usamos un relleno = 1. (padding)
#
# ### **La capa Max Pool**
# - Cada capa de agrupación, es decir, nn.MaxPool2d(2, 2) reduce a la mitad tanto la altura como el ancho de la imagen,
#   por lo que al usar 2 capas de agrupación, la altura y el ancho son 1/4 de los tamaños originales.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%204.21.04%402x.png)

# **¿Qué es torch.nn.funcional?**
# Generalmente importado al espacio de nombres F por convención, este módulo contiene todas las funciones en la
# biblioteca torch.nn (mientras que otras partes de la biblioteca contienen clases). Además de una amplia gama de
# funciones de pérdida y activación, también encontrará aquí algunas funciones convenientes para crear redes neuronales,
# como las funciones de agrupación. (También hay funciones para hacer circunvoluciones, capas lineales, etc., pero como
# veremos, generalmente se manejan mejor usando otras partes de la biblioteca).

import torch.nn as nn
import torch.nn.functional as F  #


# Crea nuestro modelo usando una clase de Python
class Net(nn.Module):
    def __init__(self):
        # super es una subclase de nn.Module y hereda todos sus métodos
        super(Net, self).__init__()

        # Definimos nuestros objetos de capa aquí, son solo objetos que estamos creando, Piense en ello como piezas de
        # una tubería. Pero aún no hemos vinculado la canalización.

        # Nuestra primera capa CNN usando 32 Fitlers de tamaño 3x3, con zancada de 1 y relleno de 0
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Nuestra segunda capa CNN usando 64 Fitlers de tamaño 3x3, con zancada de 1 y relleno de 0
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Nuestro Max Pool Layer 2 x 2 núcleo de zancada 2
        self.pool = nn.MaxPool2d(2, 2)
        # Nuestra primera capa completamente conectada (llamada Lineal), toma la salida de nuestro Max Pool
        # que mide 12 x 12 x 64 y lo conecta a un conjunto de 128 nodos
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Nuestra segunda capa totalmente conectada, conecta los 128 nodos a 10 nodos de salida (nuestras clases)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # aquí definimos nuestra secuencia de propagación hacia adelante, se hace la canalización
        # Recuerda que es Conv1 - Relu - Conv2 - Relu - Max Pool - Flatten - FC1 - FC2
        # Pytorch especifica que vas a usar la función relu con F.relu
        # x la función de activación y que es una especie de entrada de contexto como X y se asigna a la salida x
        # para ser la entrada de la siguiente para poder crear la secuencia
        x = F.relu(self.conv1(x))
        # en una sola línea definimos la segunda convolución de la red con la entrada de la salida anterior usamos relu
        # y a la salida le aplicamos el maxpool asignando su salida a x
        x = self.pool(F.relu(self.conv2(x)))
        #  usamos la función de vista para remodelar básicamente nuestro tensor, hacer el flattern o aplanado
        # Porque recuerde, tenemos la salida máxima de carrete aquí como 64 x 12 x 12
        # Queremos considerar qué es lo mismo y esto.
        # el significado del parámetro -1 es que nos da una situación en la que si no sé cuántas reglas quiere, pero
        # está seguro de la cantidad de columnas que puede usar, -1 es solo una forma de decirle a la biblioteca que no
        # estamos seguros de que me dé el resultado y lo manejaré de manera efectiva
        x = x.view(-1, 64 * 12 * 12)  # aplanar
        x = F.relu(self.fc1(x))  # creamos la capa full conected o totalmente conectada
        x = self.fc2(x)  # para transformar a probablidades usamos self, creo
        return x


# Cree una instancia del modelo y muévala (memoria y operaciones) al dispositivo CUDA
net = Net()  # instanciamos la clase
net.to(device)  # le pasamos la configuración para que use CPU o GPU


print(net)
'''Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)'''
# ### **7. Definición de una función de pérdida y un optimizador**

# Necesitamos definir qué tipo de pérdida usaremos y qué método usaremos para actualizar los gradientes.
# 1. Usamos Cross Entropy Loss ya que es un problema multiclase
# 2. Usamos Stochastic Gradient Descent (SGD) - también especificamos una tasa de aprendizaje (LR) de 0.001 y un
#    impulso de 0.9

# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ### **8. Entrenando Nuestro Modelo**
#
# En PyTorch usamos las funciones de bloques de construcción para ejecutar el algoritmo de entrenamiento con el que ya
# deberíamos estar algo familiarizados.
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%207.04.32%402x.png)

# Recorremos el conjunto de datos de entrenamiento varias veces (cada vez se denomina época)
epochs = 10

# Cree algunas matrices vacías para almacenar registros
epoch_log = []
loss_log = []
accuracy_log = []

# Iterar por un número específico de épocas
for epoch in range(epochs):
    print(f'Starting Epoch: {epoch + 1}...')

    # Seguimos agregando o acumulando nuestra pérdida después de cada mini-lote en running_loss
    running_loss = 0.0

    # Iteramos a través de nuestro iterador Trainloader
    # Cada ciclo es un minilote
    for i, data in enumerate(trainloader, 0):
        # obtener las entradas; los datos son una lista de [entradas, etiquetas]
        inputs, labels = data

        # Mover nuestros datos a GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Borre los gradientes antes de entrenar poniéndolos a cero
        # Requerido para un nuevo comienzo
        optimizer.zero_grad()

        # Adelante -> backprop + optimizar
        outputs = net(inputs)  # Propagación hacia adelante
        loss = criterion(outputs, labels) # Get Loss (cuantificar la diferencia entre los resultados y las predicciones)
        loss.backward()  # Propagación hacia atrás para obtener los nuevos gradientes para todos los nodos
        optimizer.step()  # Actualizar los gradientes/pesos

        # Imprimir estadísticas de entrenamiento - Época/Iteraciones/Pérdida/Precisión
        running_loss += loss.item()  # nos permite hacer un seguimiento de la pérdida.
        if i % 50 == 49:  # mostrar nuestra pérdida cada 50 mini lotes
            correct = 0  # Inicializar nuestra variable para mantener el conteo de las predicciones correctas
            total = 0  # Inicializar nuestra variable para mantener el conteo del número de etiquetas iteradas

            # No necesitamos gradientes para la validación, así que envuélvalos
            # no_grad para ahorrar memoria
            with torch.no_grad():  # desactiva el seguimiento de cualquier cálculo necesario para calcular mas tarde un
                # gradiente
                # Iterar a través del iterador testloader
                for data in testloader:
                    images, labels = data
                    # Mover nuestros datos a GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # Propagación hacia adelante de nuestro lote de datos de prueba a través de nuestro modelo
                    outputs = net(images)

                    # Obtener predicciones del valor máximo del tensor de salida predicho
                    # establecemos dim = 1 ya que especifica el número de dimensiones a reducir
                    _, predicted = torch.max(outputs.data, dim=1)
                    # Siga agregando el tamaño o la longitud de la etiqueta a la variable total
                    total += labels.size(0)
                    # Mantenga un total acumulado de la cantidad de predicciones pronosticadas correctamente
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(
                    f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

    # Almacenar estadísticas de entrenamiento después de cada época
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print('Finished Training')
'''Starting Epoch: 1...
Epoch: 1, Mini-Batches Completed: 50, Loss: 2.265, Test Accuracy = 53.200%
Epoch: 1, Mini-Batches Completed: 100, Loss: 2.091, Test Accuracy = 69.400%
Epoch: 1, Mini-Batches Completed: 150, Loss: 1.626, Test Accuracy = 76.490%
Epoch: 1, Mini-Batches Completed: 200, Loss: 0.926, Test Accuracy = 84.260%
Epoch: 1, Mini-Batches Completed: 250, Loss: 0.587, Test Accuracy = 87.690%
Epoch: 1, Mini-Batches Completed: 300, Loss: 0.473, Test Accuracy = 88.780%
Epoch: 1, Mini-Batches Completed: 350, Loss: 0.407, Test Accuracy = 89.740%
Epoch: 1, Mini-Batches Completed: 400, Loss: 0.375, Test Accuracy = 90.180%
Epoch: 1, Mini-Batches Completed: 450, Loss: 0.345, Test Accuracy = 90.740%
Starting Epoch: 2...
Epoch: 2, Mini-Batches Completed: 50, Loss: 0.330, Test Accuracy = 91.500%
Epoch: 2, Mini-Batches Completed: 100, Loss: 0.310, Test Accuracy = 91.840%
Epoch: 2, Mini-Batches Completed: 150, Loss: 0.312, Test Accuracy = 92.260%
Epoch: 2, Mini-Batches Completed: 200, Loss: 0.300, Test Accuracy = 92.540%
Epoch: 2, Mini-Batches Completed: 250, Loss: 0.290, Test Accuracy = 92.890%
Epoch: 2, Mini-Batches Completed: 300, Loss: 0.271, Test Accuracy = 92.950%
Epoch: 2, Mini-Batches Completed: 350, Loss: 0.278, Test Accuracy = 92.450%
Epoch: 2, Mini-Batches Completed: 400, Loss: 0.239, Test Accuracy = 93.480%
Epoch: 2, Mini-Batches Completed: 450, Loss: 0.227, Test Accuracy = 93.870%
Starting Epoch: 3...
Epoch: 3, Mini-Batches Completed: 50, Loss: 0.241, Test Accuracy = 93.590%
Epoch: 3, Mini-Batches Completed: 100, Loss: 0.221, Test Accuracy = 94.120%
Epoch: 3, Mini-Batches Completed: 150, Loss: 0.193, Test Accuracy = 94.330%
Epoch: 3, Mini-Batches Completed: 200, Loss: 0.235, Test Accuracy = 94.250%
Epoch: 3, Mini-Batches Completed: 250, Loss: 0.197, Test Accuracy = 94.730%
Epoch: 3, Mini-Batches Completed: 300, Loss: 0.210, Test Accuracy = 94.710%
Epoch: 3, Mini-Batches Completed: 350, Loss: 0.195, Test Accuracy = 94.560%
Epoch: 3, Mini-Batches Completed: 400, Loss: 0.182, Test Accuracy = 95.160%
Epoch: 3, Mini-Batches Completed: 450, Loss: 0.177, Test Accuracy = 95.380%
Starting Epoch: 4...
Epoch: 4, Mini-Batches Completed: 50, Loss: 0.171, Test Accuracy = 95.240%
Epoch: 4, Mini-Batches Completed: 100, Loss: 0.176, Test Accuracy = 95.340%
Epoch: 4, Mini-Batches Completed: 150, Loss: 0.183, Test Accuracy = 95.470%
Epoch: 4, Mini-Batches Completed: 200, Loss: 0.172, Test Accuracy = 95.650%
Epoch: 4, Mini-Batches Completed: 250, Loss: 0.145, Test Accuracy = 95.710%
Epoch: 4, Mini-Batches Completed: 300, Loss: 0.131, Test Accuracy = 95.730%
Epoch: 4, Mini-Batches Completed: 350, Loss: 0.146, Test Accuracy = 96.220%
Epoch: 4, Mini-Batches Completed: 400, Loss: 0.150, Test Accuracy = 95.560%
Epoch: 4, Mini-Batches Completed: 450, Loss: 0.156, Test Accuracy = 96.310%
Starting Epoch: 5...
Epoch: 5, Mini-Batches Completed: 50, Loss: 0.140, Test Accuracy = 96.060%
Epoch: 5, Mini-Batches Completed: 100, Loss: 0.128, Test Accuracy = 96.330%
Epoch: 5, Mini-Batches Completed: 150, Loss: 0.136, Test Accuracy = 96.370%
Epoch: 5, Mini-Batches Completed: 200, Loss: 0.122, Test Accuracy = 96.680%
Epoch: 5, Mini-Batches Completed: 250, Loss: 0.135, Test Accuracy = 96.260%
Epoch: 5, Mini-Batches Completed: 300, Loss: 0.127, Test Accuracy = 96.450%
Epoch: 5, Mini-Batches Completed: 350, Loss: 0.112, Test Accuracy = 96.760%
Epoch: 5, Mini-Batches Completed: 400, Loss: 0.123, Test Accuracy = 96.720%
Epoch: 5, Mini-Batches Completed: 450, Loss: 0.114, Test Accuracy = 96.950%
Starting Epoch: 6...
Epoch: 6, Mini-Batches Completed: 50, Loss: 0.105, Test Accuracy = 96.920%
Epoch: 6, Mini-Batches Completed: 100, Loss: 0.112, Test Accuracy = 96.860%
Epoch: 6, Mini-Batches Completed: 150, Loss: 0.115, Test Accuracy = 96.980%
Epoch: 6, Mini-Batches Completed: 200, Loss: 0.102, Test Accuracy = 97.000%
Epoch: 6, Mini-Batches Completed: 250, Loss: 0.111, Test Accuracy = 97.290%
Epoch: 6, Mini-Batches Completed: 300, Loss: 0.112, Test Accuracy = 97.140%
Epoch: 6, Mini-Batches Completed: 350, Loss: 0.091, Test Accuracy = 97.190%
Epoch: 6, Mini-Batches Completed: 400, Loss: 0.104, Test Accuracy = 97.110%
Epoch: 6, Mini-Batches Completed: 450, Loss: 0.103, Test Accuracy = 97.390%
Starting Epoch: 7...
Epoch: 7, Mini-Batches Completed: 50, Loss: 0.098, Test Accuracy = 97.460%
Epoch: 7, Mini-Batches Completed: 100, Loss: 0.102, Test Accuracy = 97.320%
Epoch: 7, Mini-Batches Completed: 150, Loss: 0.088, Test Accuracy = 97.490%
Epoch: 7, Mini-Batches Completed: 200, Loss: 0.093, Test Accuracy = 97.480%
Epoch: 7, Mini-Batches Completed: 250, Loss: 0.087, Test Accuracy = 97.420%
Epoch: 7, Mini-Batches Completed: 300, Loss: 0.080, Test Accuracy = 97.520%
Epoch: 7, Mini-Batches Completed: 350, Loss: 0.103, Test Accuracy = 97.380%
Epoch: 7, Mini-Batches Completed: 400, Loss: 0.086, Test Accuracy = 97.660%
Epoch: 7, Mini-Batches Completed: 450, Loss: 0.084, Test Accuracy = 97.570%
Starting Epoch: 8...
Epoch: 8, Mini-Batches Completed: 50, Loss: 0.083, Test Accuracy = 97.560%
Epoch: 8, Mini-Batches Completed: 100, Loss: 0.072, Test Accuracy = 97.700%
Epoch: 8, Mini-Batches Completed: 150, Loss: 0.086, Test Accuracy = 97.540%
Epoch: 8, Mini-Batches Completed: 200, Loss: 0.077, Test Accuracy = 97.610%
Epoch: 8, Mini-Batches Completed: 250, Loss: 0.086, Test Accuracy = 97.800%
Epoch: 8, Mini-Batches Completed: 300, Loss: 0.094, Test Accuracy = 97.540%
Epoch: 8, Mini-Batches Completed: 350, Loss: 0.075, Test Accuracy = 97.770%
Epoch: 8, Mini-Batches Completed: 400, Loss: 0.079, Test Accuracy = 97.780%
Epoch: 8, Mini-Batches Completed: 450, Loss: 0.072, Test Accuracy = 97.790%
Starting Epoch: 9...
Epoch: 9, Mini-Batches Completed: 50, Loss: 0.086, Test Accuracy = 97.800%
Epoch: 9, Mini-Batches Completed: 100, Loss: 0.060, Test Accuracy = 97.920%
Epoch: 9, Mini-Batches Completed: 150, Loss: 0.073, Test Accuracy = 97.830%
Epoch: 9, Mini-Batches Completed: 200, Loss: 0.076, Test Accuracy = 97.800%
Epoch: 9, Mini-Batches Completed: 250, Loss: 0.068, Test Accuracy = 97.950%
Epoch: 9, Mini-Batches Completed: 300, Loss: 0.081, Test Accuracy = 97.920%
Epoch: 9, Mini-Batches Completed: 350, Loss: 0.075, Test Accuracy = 97.960%
Epoch: 9, Mini-Batches Completed: 400, Loss: 0.065, Test Accuracy = 97.880%
Epoch: 9, Mini-Batches Completed: 450, Loss: 0.070, Test Accuracy = 98.070%
Starting Epoch: 10...
Epoch: 10, Mini-Batches Completed: 50, Loss: 0.063, Test Accuracy = 97.660%
Epoch: 10, Mini-Batches Completed: 100, Loss: 0.075, Test Accuracy = 97.900%
Epoch: 10, Mini-Batches Completed: 150, Loss: 0.066, Test Accuracy = 97.980%
Epoch: 10, Mini-Batches Completed: 200, Loss: 0.063, Test Accuracy = 98.100%
Epoch: 10, Mini-Batches Completed: 250, Loss: 0.065, Test Accuracy = 98.010%
Epoch: 10, Mini-Batches Completed: 300, Loss: 0.055, Test Accuracy = 98.110%
Epoch: 10, Mini-Batches Completed: 350, Loss: 0.072, Test Accuracy = 98.180%
Epoch: 10, Mini-Batches Completed: 400, Loss: 0.069, Test Accuracy = 98.170%
Epoch: 10, Mini-Batches Completed: 450, Loss: 0.072, Test Accuracy = 98.000%
Finished Training'''


# ## **9. Guardando nuestro modelo**

# Usamos la función ```torch.save()``` para guardar nuestro modelo.

# ```net.state_dict()``` guarda los pesos de nuestro modelo en un formato de diccionario.

PATH = 'models/mnist_cnn_net.pth'
torch.save(net.state_dict(), PATH)  # net.state_dict() le da el formato de diccionario al archivo

# ### **Veamos algunas imágenes de sus datos de prueba y veamos sus etiquetas de Ground Truth**

# Cargando un mini-lote
dataiter = iter(testloader)
images, labels = next(dataiter)

# Mostrar imágenes usando utils.make_grid() de torchvision
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ''.join('%1s' % labels[j].numpy() for j in range(128)))  # GroundTruth:  721041495906901597349...

# ### **Vamos a recargar el modelo que acabamos de guardar**

# Cree una instancia del modelo y muévala (memoria y operaciones) al dispositivo CUDA.
net = Net()
net.to(device)

# Cargar pesos desde la ruta especificada
net.load_state_dict(torch.load(PATH))

# #### **Obtención de predicciones**
# Tenga en cuenta que cuando trabaje con tensores en la GPU, tenemos que volver a convertirlo en una matriz numpy para
# realizar operaciones de python en él.
# ```tu_tensor.cpu().numpy()```


# # Propaguemos hacia adelante un mini lote y obtengamos los resultados previstos
# Usamos la función iter de Python para devolver un iterador para nuestro objeto train_loader
test_iter = iter(testloader)

# Usamos next para obtener el primer lote de datos de nuestro iterador
images, labels = next(test_iter)

# Mover nuestros datos a GPU
images = images.to(device)
labels = labels.to(device)

outputs = net(images)

# Obtenga las predicciones de la clase usando torch.max
# La razón por la que lo usamos para ver la función MAX que nos da básicamente la clase que obtuvo la máxima puntuación
# de probabilidad.
_, predicted = torch.max(outputs, 1)


# Imprime nuestras 128 predicciones
print('Predicted: ', ''.join('%1s' % predicted[j].cpu().numpy() for j in range(128)))  # Predicted:  7210414959069015...

# #### **Mostrando de nuevo la precisión de nuestra prueba**

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data # Obtenemos los datos
        # Mover nuestros datos a GPU
        images = images.to(device)
        # los pasamos a una red, obtenemos los resultados previstos,
        labels = labels.to(device)
        outputs = net(images)
        # luego obtenga el total acumulado y corrija cuánto se predice correctamente aquí usando esta función.
        # Esta línea aquí, y acabamos de decir que estos son básicamente los acumuladores.
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.3}%')
# Accuracy of the network on the 10000 test images: 98.2%


# ## **10. Trazar nuestros registros de entrenamiento**
#
# ¿Recuerdas que creamos algunas listas para registrar nuestras estadísticas de entrenamiento?
#
# ```
# # Crear matrices vacías para almacenar registros
# registro_época = []
# loss_log = []
# precision_log = []
# ```
#
# **Vamos ahora a graficar esos registros**

# En[60]:


# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch")
plt.xticks(rotation=45)

# Usamos twinx para crear un gráfico en un eje y secundario
ax2 = ax1.twinx()

# Crear gráfico para loss_log y precision_log
ax1.plot(epoch_log, loss_log, 'g-')
ax2.plot(epoch_log, accuracy_log, 'b-')

# Establecer etiquetas
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Test Accuracy', color='b')

plt.show()

# En[61]:


epoch_log = list(range(1, 11))
epoch_log

# En[62]:


list(range(1, 10))

'''
# # **Observaciones**
#
# 1. Si intenta ejecutar esta misma red en la CPU (cambie ```device = 'cpu'```. No notará una gran diferencia en la 
velocidad. Esto se debe a que su red es muy pequeña y hay muchas de sobrecarga simplemente moviendo los datos. Para 
redes más grandes o más profundas, el aumento de la velocidad de la GPU será sustancial.

# En[63]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# No necesitamos gradientes para la validación, así que envuélvalos
# no_grad para ahorrar memoria
with torch.no_grad():
    for data in testloader:
        images, labels = data

        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)

        # Obtener nuestros resultados
        outputs = net(images)

        # usar torch.max() para obtener la clase predicha para el primer dim de nuestro lote
        # tenga en cuenta que estos son solo los primeros 16 puntos de datos/imágenes de nuestro lote de 128 imágenes
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(15):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    class_accuracy = 100 * class_correct[i] / class_total[i]
    print(f'Accuracy of {i} : {class_accuracy:.3f}%')

# En[64]:


c

# **NOTA**
#
# ```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de 
manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers, BatchNorm 
Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común 
para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes:

# En[65]:


net.eval()

# No necesitamos gradientes para la validación, así que envuélvalos
# no_grad para ahorrar memoria
with torch.no_grad():
    for data in testloader:
        images, labels = data

        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)

        # Obtener nuestros resultados
        outputs = net(images)

        # use torch.argmax() para obtener las predicciones, argmax se usa para long_tensors
        predictions = torch.argmax(outputs, dim=1)

        # Para los datos de prueba en cada lote, identificamos cuándo las predicciones no coincidieron con la etiqueta
        # luego imprimimos la verdad del terreno real
        for i in range(data[0].shape[0]):
            pred = predictions[i].item()
            label = labels[i]
            if (label != pred):
                print(f'Actual Label: {pred}, Predicted Label: {label}')
                img = np.reshape(images[i].cpu().numpy(), [28, 28])
                imgshow("", np.uint8(img), size=1)

# En[66]:


nb_classes = 10

confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

# En[67]:


print(confusion_matrix.diag() / confusion_matrix.sum(1))

# En[68]:


from sklearn.metrics import confusion_matrix

nb_classes = 10

# Inicializar las listas de predicción y etiquetas (tensores)
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print(class_accuracy)

# En[69]:


c

# En[70]:


# Use numpy para crear una matriz que almacene un valor de 1 cuando ocurra una clasificación incorrecta
result = np.absolute(y_test - y_pred)
result_indices = np.nonzero(result > 0)

# Mostrar los índices de clasificaciones erróneas
print("Indices of misclassifed data are: \n\n" + str(result_indices))

# En[ ]:


from sklearn.metrics import confusion_matrix

nb_classes = 9

# Inicializar las listas de predicción y etiquetas (tensores)
predlist = torch.zeros(0, dtype=torch.long, device='cpu')
lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print(conf_mat)

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
print(class_accuracy)
'''