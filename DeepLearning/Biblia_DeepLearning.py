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

# !/usr/bin/env python
# codificación: utf-8
##############################################
# 02 Keras TensorFlow CNN MNIST Tutorial ######
##############################################
# # **Introducción a Keras usando un backend de TensorFlow 2.0**
# ### **Entrenamiento de una CNN simple en el conjunto de datos MNIST - Dígitos escritos a mano**

# En esta lección, usamos **Keras con TensorFlow 2.0** Backend para crear un **modelo de red neuronal convolucional
# simple** en PyTorch y entrenarlo para **reconocer dígitos escritos a mano en el conjunto de datos MNIST.**
# 1. Cargando nuestro conjunto de datos MNIST
# 2. Inspeccionar nuestro conjunto de datos
# 3. Visualización de nuestro conjunto de datos de imágenes
# 5. Preprocesamiento de nuestro conjunto de datos
# 6. Construyendo nuestro Modelo
# 7. Entrenando a nuestro Modelo
# 8. Graficando nuestros registros de entrenamiento
# 9. Guardando y Cargando nuestro Modelo
# 10. Probando nuestro modelo con datos de prueba

### **1. Cargando nuestros datos**
#
# Hay conjuntos de datos incorporados de ```tensorflow.keras.datasets``` para cargar nuestros datos. Usamos la función
# ```mnist.load_data()```.
#
# Devuelve: **2 tuplas**
# - x_train, x_test: matriz uint8 de datos de imagen RGB con forma (num_samples, 3, 32, 32) o (num_samples, 32, 32, 3)
# según la configuración del backend image_data_format de channel_first o channel_last respectivamente.
# - y_train, y_test: matriz uint8 de etiquetas de categoría (enteros en el rango 0-9) con forma (num_samples, 1).

# - Más información sobre los datos disponibles en https://keras.io/datasets/

# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# #### **Una revisión rápida para ver si estamos usando la GPU**

# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
'''
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 12361390168452218670
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 14190510080
locality {
  bus_id: 1
  links {
  }
}
incarnation: 12939192816315270802
physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3080 Ti Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6"
xla_global_id: 416903419
]'''

# ## **2. Inspeccionando nuestro conjunto de datos**

# Mostrar el número de muestras en x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))
# Initial shape or dimensions of x_train (60000, 28, 28)


# Imprimir el número de muestras en nuestros datos
print("Number of samples in our training data: " + str(len(x_train)))  # Number of samples in our training data: 60000
print("Number of labels in our training data: " + str(len(y_train)))  # Number of labels in our training data: 60000
print("Number of samples in our test data: " + str(len(x_test)))  # Number of samples in our test data: 10000
print("Number of labels in our test data: " + str(len(y_test)))  # Number of labels in our test data: 10000

# Imprimir las dimensiones de la imagen y nº de etiquetas en nuestros datos de entrenamiento y prueba
print("\n")
print("Dimensions of x_train:" + str(x_train[0].shape))
print("Labels in x_train:" + str(y_train.shape))
print("\n")
print("Dimensions of x_test:" + str(x_test[0].shape))
print("Labels in y_test:" + str(y_test.shape))
'''Dimensions of x_train:(28, 28)
Labels in x_train:(60000,)

Dimensions of x_test:(28, 28)
Labels in y_test:(10000,)'''

# ## **3. Visualizando nuestro conjunto de datos de imágenes**
# Echemos un vistazo a algunas de las imágenes en este conjunto de datos
# - Usando OpenCV
# - Usando Matplotlib


# Usando OpenCV
# importar opencv y numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt


def imshow(title, image=None, size=6):
    if image.any():
        w, h = image.shape[0], image.shape[1]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        print("Image not found")


# Use OpenCV para mostrar 6 imágenes aleatorias de nuestro conjunto de datos
for i in range(0, 6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    imshow("Sample", img, size=2)

# ### **Hagamos lo mismo, pero usando matplotlib para trazar 6 imágenes**
# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

# Crear figura y cambiar tamaño
figure = plt.figure()
plt.figure(figsize=(16, 10))

# Establecer cuantas imágenes deseamos ver
num_of_images = 50

# iterar índice de 1 a 51
for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index).set_title(f'{y_train[index]}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')

# ## **4. Preprocesando nuestro conjunto de datos**

# Antes de pasar nuestros datos a nuestra CNN para entrenamiento, primero debemos prepararlos. Este entials:
# 1. Remodelar nuestros datos agregando una 4ta Dimensión
# 2. Cambiar nuestro tipo de datos de uint8 a float32
# 3. Normalizando nuestros datos a valores entre 0 y 1
# 4. Una codificación en caliente

# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# Obtener nuestros datos en la 'forma' correcta necesaria para Keras
# Necesitamos agregar una cuarta dimensión a nuestros datos, cambiando así nuestra
# Nuestra forma de imagen original de (60000,28,28) a (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# cambiar nuestro tipo de imagen a tipo de datos float32
x_train = x_train.astype('float32')  # uint8 originalmente
x_test = x_test.astype('float32')

# Normalizar nuestros datos cambiando el rango de (0 a 255) a (0 a 1)
x_train /= 255.0
x_test /= 255.0

print('x_train shape:', x_train.shape)  # x_train shape: (60000, 28, 28, 1)

print(x_train.shape[0], 'train samples')  # 60000 train samples
print(x_test.shape[0], 'test samples')  # 10000 test samples

print(img_rows, img_cols)  # 28 28

# #### **Una codificación en caliente de nuestras etiquetas (Y)**
#
# Podemos implementar fácilmente este transformm usando ```to_categorical``` de ``` tensorflow.keras.utils```

# En[ ]:


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print("Number of Classes: " + str(y_test.shape[1]))  # Number of Classes: 10

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# #### **Ejemplo de una codificación activa**
# ![Imagen de una codificación activa]
# (https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/hotoneencode.JPG)

# Mira nuestros datos sin procesar
y_train[0]

# ## **5. Construyendo nuestro modelo**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-29%20at%204.21.04%402x.png)
# - Estamos construyendo una CNN simple pero efectiva que utiliza 32 filtros de tamaño 3x3
# - Hemos agregado una segunda capa CONV de 64 filtros del mismo tamaño 3x3
# - Luego reducimos la muestra de nuestros datos a 2x2
# - Luego aplanamos nuestra salida Max Pool que está conectada a una capa Dense/FC que tiene un tamaño de salida de 128
# - Luego conectamos nuestras 128 salidas a otra capa FC/Densa que da salida a las 10 unidades categóricas


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

# crear modelo
model = Sequential()

# Nuestra primera capa de convolución, tamaño de filtro 32 que reduce el tamaño de nuestra capa a 26 x 26 x 32
# Usamos la activación de ReLU y especificamos nuestro input_shape que es 28 x 28 x 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Nuestra segunda capa de convolución, tamaño de filtro 64 que reduce el tamaño de nuestra capa a 24 x 24 x 64
model.add(Conv2D(64, (3, 3), activation='relu'))

# Usamos MaxPooling con un tamaño de kernel de 2 x 2, esto reduce nuestro tamaño a 12 x 12 x 64
model.add(MaxPooling2D(pool_size=(2, 2)))

# Luego aplanamos nuestro objeto tensor antes de ingresarlo en nuestra capa densa
# Una operación de aplanamiento en un tensor remodela el tensor para que tenga la forma que es
# igual al número de elementos contenidos en el tensor
# En nuestra CNN va de 12*12*64 a 9216*1
model.add(Flatten())

# Conectamos esta capa a una capa Totalmente Conectada/Densa de tamaño 1 * 128
model.add(Dense(128, activation='relu'))

# Creamos nuestra capa final totalmente conectada/densa con una salida para cada clase (10)
model.add(Dense(num_classes, activation='softmax'))

# Compilamos nuestro modelo, esto crea un objeto que almacena el modelo que acabamos de crear
# Configuramos nuestro Optimizer para usar Stochastic Gradient Descent (tasa de aprendizaje de 0.001)
# Configuramos nuestra función de pérdida para que sea categorical_crossentropy ya que es adecuada para problemas
# multiclase
# Finalmente, las métricas (en qué juzgamos nuestro desempeño) para ser precisión
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.001),
              metrics=['accuracy'])

# Podemos usar la función de resumen para mostrar las capas y los parámetros de nuestro modelo
print(model.summary())
'''Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       

 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     

 max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0         
 )                                                               

 flatten (Flatten)           (None, 9216)              0         

 dense (Dense)               (None, 128)               1179776   

 dense_1 (Dense)             (None, 10)                1290      

=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
'''

### **6. Entrenando a nuestro Modelo**
# - Nuestros datos preprocesados se utilizan como entrada
# - Establecemos el tamaño del lote en 128 (o cualquier número entre 8 y 256 es bueno)
# - Establecemos el número de épocas en 2, esto es solo para el propósito de este tutorial, pero se debe usar un valor
# de al menos 10
# - Almacenamos los resultados de entrenamiento de nuestro modelo para trazar en el futuro
# - Luego usamos la función de evaluación de Molel de Kera para generar el rendimiento final del modelo. Aquí estamos
# examinando la pérdida de prueba y la precisión de la prueba

# En[ ]:


batch_size = 128
epochs = 25

# Almacene nuestros resultados aquí para que podamos graficar más tarde
# En nuestra función de ajuste especificamos nuestros conjuntos de datos (x_train y y_train),
# el tamaño del lote (típicamente de 16 a 128 dependiendo de su RAM), el número de
# épocas (generalmente de 10 a 100) y nuestros conjuntos de datos de validación (x_test & y_test)
# verbose = 1, configura nuestro entrenamiento para generar métricas de rendimiento cada época
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,  # la información que te da actualmente 0,1 y 2 ma numeración + info
                    validation_data=(x_test, y_test))

'''469/469 [==============================] - 41s 54ms/step - loss: 2.2307 - accuracy: 0.3249 - val_loss: 2.1226 - val_accuracy: 0.5544
Epoch 2/25
469/469 [==============================] - 24s 52ms/step - loss: 1.8059 - accuracy: 0.6838 - val_loss: 1.2555 - val_accuracy: 0.7972
Epoch 3/25
469/469 [==============================] - 23s 49ms/step - loss: 0.8263 - accuracy: 0.8237 - val_loss: 0.5420 - val_accuracy: 0.8657
Epoch 4/25
469/469 [==============================] - 25s 53ms/step - loss: 0.4859 - accuracy: 0.8678 - val_loss: 0.4071 - val_accuracy: 0.8884
Epoch 5/25
469/469 [==============================] - 24s 50ms/step - loss: 0.4012 - accuracy: 0.8860 - val_loss: 0.3566 - val_accuracy: 0.8988
Epoch 6/25
469/469 [==============================] - 34s 72ms/step - loss: 0.3617 - accuracy: 0.8951 - val_loss: 0.3283 - val_accuracy: 0.9047
Epoch 7/25
469/469 [==============================] - 97s 206ms/step - loss: 0.3367 - accuracy: 0.9017 - val_loss: 0.3072 - val_accuracy: 0.9120
Epoch 8/25
469/469 [==============================] - 21s 44ms/step - loss: 0.3180 - accuracy: 0.9079 - val_loss: 0.2925 - val_accuracy: 0.9154
Epoch 9/25
469/469 [==============================] - 4s 8ms/step - loss: 0.3027 - accuracy: 0.9117 - val_loss: 0.2806 - val_accuracy: 0.9198
Epoch 10/25
469/469 [==============================] - 3s 7ms/step - loss: 0.2894 - accuracy: 0.9155 - val_loss: 0.2674 - val_accuracy: 0.9249
Epoch 11/25
469/469 [==============================] - 4s 7ms/step - loss: 0.2776 - accuracy: 0.9183 - val_loss: 0.2580 - val_accuracy: 0.9261
Epoch 12/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2670 - accuracy: 0.9214 - val_loss: 0.2496 - val_accuracy: 0.9290
Epoch 13/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2572 - accuracy: 0.9244 - val_loss: 0.2384 - val_accuracy: 0.9325
Epoch 14/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2478 - accuracy: 0.9272 - val_loss: 0.2316 - val_accuracy: 0.9346
Epoch 15/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2393 - accuracy: 0.9291 - val_loss: 0.2232 - val_accuracy: 0.9362
Epoch 16/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2311 - accuracy: 0.9320 - val_loss: 0.2144 - val_accuracy: 0.9379
Epoch 17/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2234 - accuracy: 0.9342 - val_loss: 0.2088 - val_accuracy: 0.9403
Epoch 18/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2159 - accuracy: 0.9366 - val_loss: 0.2022 - val_accuracy: 0.9419
Epoch 19/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2092 - accuracy: 0.9387 - val_loss: 0.1964 - val_accuracy: 0.9432
Epoch 20/25
469/469 [==============================] - 4s 8ms/step - loss: 0.2028 - accuracy: 0.9407 - val_loss: 0.1909 - val_accuracy: 0.9456
Epoch 21/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1970 - accuracy: 0.9425 - val_loss: 0.1856 - val_accuracy: 0.9465
Epoch 22/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1914 - accuracy: 0.9437 - val_loss: 0.1812 - val_accuracy: 0.9478
Epoch 23/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1862 - accuracy: 0.9453 - val_loss: 0.1795 - val_accuracy: 0.9466
Epoch 24/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1812 - accuracy: 0.9472 - val_loss: 0.1724 - val_accuracy: 0.9489
Epoch 25/25
469/469 [==============================] - 4s 8ms/step - loss: 0.1770 - accuracy: 0.9479 - val_loss: 0.1683 - val_accuracy: 0.9522'''

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])  # Test loss: 0.16829311847686768
print('Test accuracy:', score[1])  # Test accuracy: 0.9521999955177307

### **7. Trazado de nuestras tablas de pérdida y precisión**


history_dict = history.history
history_dict

# Trazando nuestras tablas de pérdidas
import matplotlib.pyplot as plt

# Use el objeto Historial que creamos para obtener nuestros resultados de rendimiento guardados
history_dict = history.history

# Extraer la pérdida y las pérdidas de validación
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# Obtenga el número de épocas y cree una matriz hasta ese número usando range()
epochs = range(1, len(loss_values) + 1)

# Trazar gráficos de líneas para validación y pérdida de entrenamiento
line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# #### **Nuestras tablas de precisión**


# Trazando nuestros gráficos de precisión
import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# ## **8. Guardando y Cargando nuestro Modelo**
#
# **Guardar nuestro modelo es simple, solo use:**
#
# ```modelo.guardar("modelo_nombre_archivo.h5")```

# En[ ]:


model.save("mnist_simple_cnn_10_Epochs.h5")
print("Model Saved")  # Model Saved

# **Cargar nuestro modelo guardado también es simple, solo use:**

# ```load_model(modelo_nombre_archivo.h5)```

# Necesitamos importar nuestra función load_model
from tensorflow.keras.models import load_model

classifier = load_model('mnist_simple_cnn_10_Epochs.h5')

# ## **9. Obtener predicciones de nuestros datos de prueba de muestra**
#
# **Predicción de todos los datos de prueba**


# x_prueba = x_prueba.reforma(10000,28,28,1)
print(x_test.shape)  # (10000, 28, 28, 1)

print("Predicting classes for all 10,000 test images...")  # Predicting classes for all 10,000 test images...

pred = np.argmax(classifier.predict(x_test), axis=-1)
print("Completed.\n")  # Completed.

print(pred)  # [7 2 1 ... 4 5 6]
print(type(pred))  # <class 'numpy.ndarray'>
print(len(pred))  # 10000

# **Predecir una imagen de prueba individual**

# Obtenga la primera imagen por índice 0 de x_test y muestre su forma
input_im = x_test[0]
print(input_im.shape)  # (28, 28, 1)

# Necesitamos agregar una cuarta dimensión al primer eje
input_im = input_im.reshape(1, 28, 28, 1)
print(input_im.shape)  # (1, 28, 28, 1)

# Ahora obtenemos las predicciones para esa sola imagen
pred = np.argmax(classifier.predict(input_im), axis=-1)  # 1/1 [==============================] - 0s 34ms/step
print(pred)  # [7]
print(type(pred))  # <class 'numpy.ndarray'>
print(len(pred))  # 1

# ### **Ahora hagamos algo elegante, pongamos la etiqueta predicha en una imagen con la imagen de datos de prueba**

import cv2
import numpy as np

# Recargar nuestros datos ya que lo reescalamos
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def draw_test(name, pred, input_im):
    '''Function that places the predicted class next to the original image'''
    # Crea nuestro fondo negro
    BLACK = [0, 0, 0]
    # Ampliamos nuestra imagen original a la derecha para crear espacio para colocar nuestro texto de clase predicho
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    # convertir nuestra imagen en escala de grises a color
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    # Ponga nuestro texto de clase predicho en nuestra imagen expandida
    cv2.putText(expanded_image, str(pred), (150, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    imshow(name, expanded_image)


for i in range(0, 10):
    # Obtenga una imagen de datos aleatorios de nuestro conjunto de datos de prueba
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]

    # Cree una imagen redimensionada más grande para contener nuestro texto y permitir una visualización más grande
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # Reformar nuestros datos para que podamos ingresarlos (hacia adelante propagarlos) a nuestra red
    input_im = input_im.reshape(1, 28, 28, 1)

    # Obtener predicción, use [0] para acceder al valor en la matriz numpy ya que está almacenada como una matriz
    res = str(np.argmax(classifier.predict(input_im), axis=-1)[0])

    # Coloque la etiqueta en la imagen de nuestra muestra de datos de prueba
    draw_test("Prediction", res, np.uint8(imageL))

# !/usr/bin/env python
# codificación: utf-8
##################################################################################
# 03 Clasificaciones erróneas de Keras y análisis de rendimiento del modelo######
##################################################################################
# # **Análisis de rendimiento del modelo Keras**
# En esta lección, aprendemos a usar el modelo MNIST que entrenamos en la lección anterior y analizamos su desempeño,
# hacemos:
# 1. Cargue nuestro modelo y datos de Keras
# 2. Ver las imágenes que clasificamos mal
# 3. Crea una matriz de confusión
# 4. Crear informe de clasificación

# ## **1. Cargue nuestro modelo Keras y el conjunto de datos MNIST**
# **Descargue nuestro modelo anterior (02) y cárguelo con load_model**

# Necesitamos importar nuestra función load_model
from tensorflow.keras.models import load_model

model = load_model('models/mnist_simple_cnn_10_Epochs.h5')

# **Cargar nuestro conjunto de datos MNIST**
# Técnicamente, solo necesitamos cargar el conjunto de datos de prueba, ya que estamos analizando el rendimiento en ese
# segmento de datos.


# Podemos cargar los conjuntos de datos incorporados desde esta función
from tensorflow.keras.datasets import mnist

# carga el conjunto de datos de entrenamiento y prueba del MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # **2. Ver nuestras clasificaciones erróneas**
# #### **Primero obtengamos nuestras predicciones de prueba**

import numpy as np

# Reformamos nuestros datos de prueba
print(x_test.shape)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_test.shape)

# Obtenga las predicciones para todas las muestras de 10K en nuestros datos de prueba
print("Predicting classes for all 10,000 test images...")
pred = np.argmax(model.predict(x_test), axis=-1)  # 313/313 [==============================] - 3s 1ms/step
print("Completed.\n")  # Completed.

import cv2
import numpy as np

# Use numpy para crear una matriz que almacene un valor de 1 cuando ocurra una clasificación incorrecta
result = np.absolute(y_test - pred)
misclassified_indices = np.nonzero(result > 0)

# Mostrar los índices de clasificaciones erróneas
print(f"Indices of misclassifed data are: \n{misclassified_indices}")
'''Indices of misclassifed data are: 
(array([   8,   62,  195,  241,  247,  259,  290,  300,  318,  320,  321,
        340,  341,  352,  362,  412,  445,  448,  449,  478,  502,  507,
        515,  531,  551,  565,  578,  591,  613,  619,  628,  659,  684,
        689,  691,  707,  717,  720,  740,  791,  795,  839,  898,  938,
        939,  947,  950,  951,  965,  975,  990, 1003, 1014, 1032, 1039,
       1044, 1062, 1068, 1073, 1082, 1107, 1112, 1114, 1191, 1198, 1204,
       1206, 1226, 1232, 1234, 1242, 1243, 1247, 1260, 1270, 1283, 1289,
       1299, 1319, 1326, 1328, 1337, 1364, 1378, 1393, 1410, 1433, 1440,
       1444, 1453, 1494, 1500, 1522, 1527, 1530, 1549, 1553, 1581, 1609,
       1634, 1640, 1671, 1681, 1709, 1717, 1718, 1737, 1754, 1774, 1790,
       1800, 1850, 1865, 1878, 1883, 1901, 1911, 1938, 1940, 1952, 1970,
       1981, 1984, 2016, 2024, 2035, 2037, 2040, 2043, 2044, 2053, 2070,
       2098, 2109, 2118, 2125, 2129, 2130, 2135, 2182, 2185, 2186, 2189,
       2215, 2224, 2266, 2272, 2293, 2299, 2325, 2369, 2371, 2381, 2387,
       2394, 2395, 2406, 2408, 2414, 2422, 2425, 2433, 2488, 2545, 2573,
       2574, 2607, 2610, 2648, 2654, 2751, 2760, 2771, 2780, 2810, 2832,
       2836, 2863, 2896, 2914, 2925, 2927, 2930, 2945, 2953, 2986, 2990,
       2995, 3005, 3060, 3073, 3106, 3110, 3117, 3130, 3136, 3145, 3157,
       3167, 3189, 3206, 3240, 3269, 3284, 3330, 3333, 3336, 3376, 3384,
       3410, 3422, 3503, 3520, 3547, 3549, 3550, 3558, 3565, 3567, 3573,
       3597, 3604, 3629, 3664, 3681, 3702, 3716, 3718, 3751, 3757, 3763,
       3767, 3776, 3780, 3796, 3806, 3808, 3811, 3838, 3846, 3848, 3853,
       3855, 3862, 3869, 3876, 3893, 3902, 3906, 3926, 3941, 3946, 3968,
       3976, 3985, 4000, 4007, 4017, 4063, 4065, 4068, 4072, 4075, 4076,
       4078, 4093, 4131, 4145, 4152, 4154, 4163, 4176, 4180, 4199, 4205,
       4211, 4212, 4224, 4238, 4248, 4256, 4271, 4289, 4300, 4306, 4313,
       4315, 4341, 4355, 4356, 4369, 4374, 4425, 4433, 4435, 4449, 4451,
       4477, 4497, 4498, 4500, 4523, 4536, 4540, 4571, 4575, 4578, 4601,
       4615, 4633, 4639, 4671, 4740, 4751, 4761, 4785, 4807, 4808, 4814,
       4823, 4837, 4863, 4876, 4879, 4880, 4886, 4910, 4939, 4950, 4956,
       4966, 4981, 4990, 4997, 5009, 5068, 5135, 5140, 5210, 5331, 5457,
       5600, 5611, 5642, 5654, 5714, 5734, 5749, 5757, 5835, 5842, 5887,
       5888, 5891, 5912, 5913, 5936, 5937, 5955, 5972, 5973, 5985, 6035,
       6042, 6043, 6045, 6059, 6065, 6071, 6081, 6091, 6093, 6112, 6157,
       6166, 6168, 6172, 6173, 6400, 6421, 6505, 6555, 6560, 6568, 6569,
       6571, 6572, 6574, 6597, 6598, 6603, 6641, 6642, 6651, 6706, 6740,
       6746, 6765, 6783, 6817, 6847, 6906, 6926, 7043, 7094, 7121, 7130,
       7338, 7432, 7434, 7451, 7459, 7492, 7498, 7539, 7580, 7812, 7847,
       7849, 7886, 7899, 7921, 7928, 7945, 7990, 8020, 8062, 8072, 8091,
       8094, 8095, 8183, 8246, 8272, 8277, 8279, 8311, 8332, 8339, 8406,
       8408, 8444, 8520, 8522, 9009, 9013, 9015, 9016, 9019, 9024, 9026,
       9036, 9045, 9245, 9280, 9427, 9465, 9482, 9544, 9587, 9624, 9634,
       9642, 9643, 9662, 9679, 9698, 9719, 9729, 9741, 9744, 9745, 9749,
       9752, 9768, 9770, 9779, 9808, 9811, 9832, 9839, 9856, 9858, 9867,
       9879, 9883, 9888, 9893, 9905, 9941, 9944, 9970, 9980, 9982, 9986]),)'''
print(len(misclassified_indices[0]))  # 495

# ### **Visualización de las imágenes clasificadas incorrectamente por nuestro modelo**


import cv2
import numpy as np
from matplotlib import pyplot as plt


# Definir nuestra función imshow
def imshow(title="", image=None, size=6):
    if image.any():
        w, h = image.shape[0], image.shape[1]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        print("Image not found")


# Recargar nuestros datos ya que lo reescalamos
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def draw_test(name, pred, input_im):
    '''Function that places the predicted class next to the original image'''
    # Crea nuestro fondo negro
    BLACK = [0, 0, 0]
    # Ampliamos nuestra imagen original a la derecha para crear espacio para colocar nuestro texto de clase predicho
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    # convertir nuestra imagen en escala de grises a color
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    # Ponga nuestro texto de clase predicho en nuestra imagen expandida
    cv2.putText(expanded_image, str(pred), (150, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    imshow(name, expanded_image)


for i in range(0, 10):
    # Obtenga una imagen de datos aleatorios de nuestro conjunto de datos de prueba
    input_im = x_test[misclassified_indices[0][i]]
    # Cree una imagen redimensionada más grande para contener nuestro texto y permitir una visualización más grande
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # Reformar nuestros datos para que podamos ingresarlos (hacia adelante propagarlos) a nuestra red
    input_im = input_im.reshape(1, 28, 28, 1)

    # Obtener predicción, use [0] para acceder al valor en la matriz numpy ya que está almacenada como una matriz
    res = str(np.argmax(model.predict(input_im), axis=-1)[0])  # 1/1 [==============================] - 0s 28ms/step

    # Coloque la etiqueta en la imagen de nuestra muestra de datos de prueba
    draw_test("Misclassified Prediction", res, np.uint8(imageL))  # imprime la imagen mal predicha y la etiqueta real

# ### **Una forma más elegante de trazar esto**
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    input_im = x_test[misclassified_indices[0][i]]
    ind = misclassified_indices[0][i]
    predicted_class = str(np.argmax(model.predict(input_im.reshape(1, 28, 28, 1)), axis=-1)[0])
    axes[i].imshow(input_im.reshape(28, 28), cmap='gray_r')
    axes[i].set_title(f"Prediction Class = {predicted_class}\n Original Class = {y_test[ind]}")
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)  # lo imprime todo en una sola imagen con subplots

## **3. Creando nuestra Matriz de Confusión**
#
# Usamos la herramienta Confusion Matrix de Sklean para crearlo. Todo lo que necesitamos es:
# 1. Las verdaderas etiquetas
# 2. Las etiquetas predichas


from sklearn.metrics import confusion_matrix
import numpy as np

x_test = x_test.reshape(10000, 28, 28, 1)
y_pred = np.argmax(model.predict(x_test), axis=-1)

print(confusion_matrix(y_test, y_pred))
'''
[[ 968    0    0    2    0    3    3    2    2    0]
 [   0 1116    3    2    0    0    4    1    9    0]
 [   7    1  969   19    5    0    7    7   16    1]
 [   2    2    6  961    0    7    2    8   14    8]
 [   1    0    4    0  943    0    7    3    2   22]
 [   9    2    0   27    6  805   15    5   14    9]
 [   9    2    2    3    7    9  922    2    2    0]
 [   1    6   16    9    2    0    0  969    4   21]
 [   5    2    1   13    8    7    6    6  919    7]
 [   8    6    1   12   28    1    1   13    6  933]]'''

# #### **Interpretación de la matriz de confusión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2010.46.45.png)
#
# ### **Creando una trama más presentable**
#
# Reutilizaremos esta función muy bien hecha de la documentación de sklearn sobre el trazado de una matriz de confusión
# usando gradientes de color y etiquetas.


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """dada una matriz de confusión de sklearn (cm), haga una buena trama

Argumentos
---------
cm: matriz de confusión de sklearn.metrics.confusion_matrix

target_names: clases de clasificación dadas como [0, 1, 2]
los nombres de las clases, por ejemplo: ['high', 'medium', 'low']

título: el texto que se mostrará en la parte superior de la matriz

cmap: el gradiente de los valores mostrados desde matplotlib.pyplot.cm
consulte http://matplotlib.org/examples/color/colormaps_reference.html
plt.get_cmap('jet') o plt.cm.Blues

normalizar: si es falso, graficar los números sin procesar
Si es Verdadero, traza las proporciones

Uso
-----
plot_confusion_matrix(cm = cm,  # matriz de confusión creada por
# sklearn.metrics.confusion_matrix
normalize = True,                # mostrar proporciones
target_names = y_labels_vals,    # lista de nombres de las clases
title = best_estimator_name) # título del gráfico

Citación
---------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


target_names = list(range(0, 10))
conf_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat, target_names)

# ## **Veamos nuestra precisión por clase**

# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

for (i, ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')

'''Accuracy for 0 : 98.776%
Accuracy for 1 : 98.326%
Accuracy for 2 : 93.895%
Accuracy for 3 : 95.149%
Accuracy for 4 : 96.029%
Accuracy for 5 : 90.247%
Accuracy for 6 : 96.242%
Accuracy for 7 : 94.261%
Accuracy for 8 : 94.353%
Accuracy for 9 : 92.468%'''

# # **4. Ahora veamos el Informe de Clasificación**

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
'''              precision    recall  f1-score   support

           0       0.96      0.99      0.97       980
           1       0.98      0.98      0.98      1135
           2       0.97      0.94      0.95      1032
           3       0.92      0.95      0.93      1010
           4       0.94      0.96      0.95       982
           5       0.97      0.90      0.93       892
           6       0.95      0.96      0.96       958
           7       0.95      0.94      0.95      1028
           8       0.93      0.94      0.94       974
           9       0.93      0.92      0.93      1009

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

Process finished with exit code 0
'''

# ### **4.1 El soporte es la suma total de esa clase en el conjunto de datos**
# ### **4.2 Revisión del retiro del mercado**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.12.png)
#
# ### **4.3 Revisión de la precisión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.22.png)
#
# ### **4.4 Alta recuperación (o sensibilidad) con baja precisión.**
# Esto nos dice que la mayoría de los ejemplos positivos se reconocen correctamente (falsos negativos bajos), pero hay
# falsos positivos, es decir, otras clases se predicen como nuestra clase en cuestión.
#
# ### **4.5 Baja recuperación (o sensibilidad) con alta precisión.**
#
# A nuestro clasificador le faltan muchos ejemplos positivos (FN alto), pero aquellos que predecimos como positivos son
# realmente positivos (Falsos positivos bajos)
#

#!/usr/bin/env python
# codificación: utf-8
##################################################################################
# 04 Clasificaciones erróneas de PyTorch y análisis de rendimiento del modelo#####
##################################################################################
# # **Análisis de rendimiento del modelo PyTorch**

# En esta lección, aprendemos a usar el modelo MNIST que entrenamos en la lección anterior y analizamos su desempeño,
# hacemos:

# 1. Configure nuestro modelo y datos de PyTorch
# 2. Cargar el modelo previamente entrenado
# 3. Ver las imágenes que clasificamos mal
# 4. Crea una Matriz de Confusión
# 5. Crear informe de clasificación
#

# # **1. Configure nuestras importaciones de PyTorch, modele y cargue el conjunto de datos MNIST**
#
# Solo necesitamos cargar el conjunto de datos de prueba, ya que estamos analizando el rendimiento en ese segmento de
# datos.


# Importar PyTorch
import torch

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))  # GPU available: True

# Establecer dispositivo en cuda
device = 'cuda'


# #### **Nuestra función de trazado de imágenes**

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Definir nuestra función imshow
def imgshow(title, image = None, size = 6):
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.show()


# ### **Cargando nuestro conjunto de datos de prueba MNIST**

# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, )) ])

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.MNIST('mnist',
                                     train = False,
                                     download = True,
                                     transform = transform)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size = 128,
                                          shuffle = False,
                                          num_workers = 0)


# ### **Creando nuestra clase de definición de modelo**

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # **2. Cargando modelo**
# Usamos gdown en nuestra terminal para descargar el archivo modelo que entrenamos en la última lección.
# #### **NOTA**
#
# Al cargar nuestro modelo, necesitamos crear la instancia del modelo, es decir, ```net = Net()``` y luego, dado que lo
# entrenamos usando nuestra GPU, lo movemos a la GPU usando ```net.to(dispositivo``` donde dispositivo = 'cuda'.

# Entonces podemos cargar los pesos de nuestro modelo descargado.


# Crear una instancia del modelo
net = Net()
net.to(device)

# Cargar pesos desde la ruta especificada
net.load_state_dict(torch.load('models/mnist_cnn_net.pth'))


# Modelo cargado con éxito si se muestra ```Todas las claves coincidieron correctamente```.
#
# ### **Ahora calculemos su precisión (hecho en la lección anterior, así que esto es solo un resumen) en los datos de
# prueba**


correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.3}%')
# Accuracy of the network on the 10000 test images: 98.2%



# # **3. Mostrando nuestras imágenes mal clasificadas** ##
#
# De 10.000 imágenes, nuestro modelo predijo que el 98,7 % era correcto. Esto es bueno para un primer intento con un
# modelo tan simple. (hay modelos mucho mejores).
#
# **¡Una buena práctica!**
#
# Es un buen hábito al crear clasificadores de imágenes inspeccionar visualmente las imágenes que están mal
# clasificadas.
# 1. Podemos detectar qué tipos de imágenes son un desafío para nuestro modelo
# 2. Podemos detectar cualquier imagen etiquetada incorrectamente
# 3. Si a veces no podemos identificar correctamente la clase, ver tu lucha en CNN duele menos :)
#
# **Recordatorio** de por qué usamos ```net.eval()``` y ```torch.no_grad()```
#
# [Tomado de Stackoverflow:](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
#
# **model.eval()** es una especie de interruptor para algunas capas/partes específicas del modelo que se comportan de
# manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, capas de **abandonos**,
# capas BatchNorm, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además,
# la práctica común para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de
# gradientes.
#
# Entonces, aunque no usamos Dropouts o BatchNorm en nuestro modelo, es una buena práctica usarlo al hacer inferencias.


# Establecer el modelo en modo de evaluación o inferencia
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

        # Para los datos de prueba en cada lote, identificamos qué predicciones no coincidieron con la etiqueta
        # luego imprimimos la verdad del terreno real
        for i in range(data[0].shape[0]):
            pred = predictions[i].item()
            label = labels[i]
            if(label != pred):
                print(f'Actual Label: {label}, Predicted Label: {pred}')
                img = np.reshape(images[i].cpu().numpy(),[28,28])
                imgshow("", np.uint8(img), size = 1)

'''
Actual Label: 4, Predicted Label: 9
Actual Label: 6, Predicted Label: 0
Actual Label: 2, Predicted Label: 3
... para todos las predicciones erróneas'''

# # **4. Creando nuestra Matriz de Confusión**
#
# Usamos la herramienta Confusion Matrix de Sklean para crearlo. Todo lo que necesitamos es:
# 1. Las verdaderas etiquetas
# 2. Las etiquetas predichas

from sklearn.metrics import confusion_matrix


# Inicializar tensores en blanco para almacenar nuestras predicciones y listas de etiquetas (tensores)
pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
label_list = torch.zeros(0, dtype=torch.long, device='cpu')

with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        # Agregar resultados de predicción por lotes,
        # cat es para concatenar, entiendo que preds.view(-1).cpu() es la ultima predicción despues de ese recorrido
        # net(inputs) y de sacar el máximo torch.max(outputs, 1) y lo va añadiendo a lo que ya se ha realizado en las
        # iteraciónes anteriores
        pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
        label_list = torch.cat([label_list, classes.view(-1).cpu()])

# Matriz de confusión
conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
print(conf_mat)
'''[[ 970    0    1    1    0    0    1    1    3    3]
 [   1 1129    1    1    0    0    3    0    0    0]
 [   4    6 1005    6    3    0    1    3    3    1]
 [   0    0    1 1000    0    3    0    2    3    1]
 [   1    0    1    0  971    0    0    0    2    7]
 [   2    0    0    5    0  879    3    1    2    0]
 [   7    3    0    0    3    5  937    0    3    0]
 [   1    4   12    4    0    0    0  994    5    8]
 [   6    0    1    4    3    2    2    2  950    4]
 [   3    4    0    4    6    3    0    5    1  983]]'''

# #### **Interpretación de la matriz de confusión**
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2010.46.45.png)

# ### **Creando una trama más presentable**
#
# Reutilizaremos esta función muy bien hecha de la documentación de sklearn sobre el trazado de una matriz de confusión
# usando gradientes de color y etiquetas.


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """dada una matriz de confusión de sklearn (cm), haga una buena trama

Argumentos
---------
cm: matriz de confusión de sklearn.metrics.confusion_matrix

target_names: clases de clasificación dadas como [0, 1, 2]
los nombres de las clases, por ejemplo: ['high', 'medium', 'low']

título: el texto que se mostrará en la parte superior de la matriz

cmap: el gradiente de los valores mostrados desde matplotlib.pyplot.cm
consulte http://matplotlib.org/examples/color/colormaps_reference.html
plt.get_cmap('jet') o plt.cm.Blues

normalizar: si es falso, graficar los números sin procesar
Si es Verdadero, traza las proporciones

Uso
-----
plot_confusion_matrix(cm           = cm,                  # matriz de confusión creada por
# sklearn.metrics.confusion_matrix
normalize    = True,                # mostrar proporciones
target_names = y_labels_vals,       # lista de nombres de las clases
title        = best_estimator_name) # título del gráfico

Citación
---------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()




target_names = list(range(0,10))
plot_confusion_matrix(conf_mat, target_names)


# ## **Veamos nuestra precisión por clase**


# Precisión por clase
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
for (i, ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')
'''Accuracy for 0 : 98.980%
Accuracy for 1 : 99.471%
Accuracy for 2 : 97.384%
Accuracy for 3 : 99.010%
Accuracy for 4 : 98.880%
Accuracy for 5 : 98.543%
Accuracy for 6 : 97.808%
Accuracy for 7 : 96.693%
Accuracy for 8 : 97.536%
Accuracy for 9 : 97.423%'''

# # **5. Ahora veamos el Informe de Clasificación**

from sklearn.metrics import classification_report
print(classification_report(label_list.numpy(), pred_list.numpy()))
'''precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.98      1032
           3       0.98      0.99      0.98      1010
           4       0.98      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       0.99      0.98      0.98       958
           7       0.99      0.97      0.98      1028
           8       0.98      0.98      0.98       974
           9       0.98      0.97      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
'''

# ### **5.1 El soporte es la suma total de esa clase en el conjunto de datos**
# ### **5.2 Revisión del retiro del mercado**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.12.png)
#
# ### **5.3 Revisión de la precisión**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-11-30%20at%2011.11.22.png)
#
# ### **5.4 Alta recuperación (o sensibilidad) con baja precisión.**
# Esto nos dice que la mayoría de los ejemplos positivos se reconocen correctamente (falsos negativos bajos), pero hay
# muchos falsos positivos, es decir, otras clases se predicen como nuestra clase en cuestión.
#
# ### **5.5 Baja recuperación (o sensibilidad) con alta precisión.**
#
# A nuestro clasificador le faltan muchos ejemplos positivos (FN alto), pero aquellos que predecimos como positivos son
# realmente positivos (Falsos positivos bajos)
#

# !/usr/bin/env python
# codificación: utf-8
########################################################
# 05 Keras -Fashion-MNIST Part 1 - No Regularisation####
########################################################

# # **Regularización en Keras - Parte 1 - Sin regularización**
# ### **Primero entrenamos una CNN en el conjunto de datos Fashion-MNIST sin usar métodos de regularización**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)

# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** usando Keras con
# TensorFlow 2.0 y lo entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, sin el uso de
# ningún método de regularización.

# 1. Cargando, Inspeccionando y Visualizando nuestros datos
# 2. Preprocesamiento de nuestros datos
# 3. Construya una CNN simple sin regularización
# 4. Capacitar a nuestra CNN
# 5. Eche un vistazo al aumento de datos

# # **Cargando, Inspeccionando y Visualizando nuestros datos**


# Cargamos nuestros datos directamente desde los conjuntos de datos incluidos en tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist

# carga el conjunto de datos de entrenamiento y prueba de Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Nuestros nombres de clase, al cargar datos de .datasets() nuestras clases son números enteros
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #### **Comprueba si estamos usando la GPU**
# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
'''[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 11394071547476944173
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 14363459584
locality {
  bus_id: 1
  links {
  }
}
incarnation: 14253582277000023109
physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3080 Ti Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6"
xla_global_id: 416903419
]
'''

# ### **Inspeccionar nuestros datos**
# Mostrar el número de muestras en x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))

# Imprimir el número de muestras en nuestros datos
print("Number of samples in our training data: " + str(len(x_train)))
print("Number of labels in our training data: " + str(len(y_train)))
print("Number of samples in our test data: " + str(len(x_test)))
print("Number of labels in our test data: " + str(len(y_test)))

'''Initial shape or dimensions of x_train (60000, 28, 28)
Number of samples in our training data: 60000
Number of labels in our training data: 60000
Number of samples in our test data: 10000
Number of labels in our test data: 10000'''

# Imprimir las dimensiones de la imagen y nº de etiquetas en nuestros datos de entrenamiento y prueba
print("\n")
print("Dimensions of x_train:" + str(x_train[0].shape))
print("Labels in x_train:" + str(y_train.shape))
print("\n")
print("Dimensions of x_test:" + str(x_test[0].shape))
print("Labels in y_test:" + str(y_test.shape))

'''Dimensions of x_train:(28, 28)
Labels in x_train:(60000,)
Dimensions of x_test:(28, 28)
Labels in y_test:(10000,)'''

# ### **Visualización de algunos de nuestros datos de muestra**
# Tracemos 50 imágenes de muestra.


# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

# Crear figura y cambiar tamaño
figure = plt.figure()
plt.figure(figsize=(16, 10))

# Establecer cuantas imágenes deseamos ver
num_of_images = 50

# iterar índice de 1 a 51
for index in range(1, num_of_images + 1):
    class_names = classes[y_train[index]]
    plt.subplot(5, 10, index).set_title(f'{class_names}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')

# # **2. Preprocesamiento de datos**
# Primero, hagamos un seguimiento de algunas dimensiones de datos:
# - ```img_rows``` que debería ser 28
# - ```img_cols``` que debería ser 28
# - ```input_shape```, que es 28 x 28 x 1


# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# ### **Una codificación en caliente de nuestras etiquetas**
# **Ejemplo de una codificación activa**
# ![Imagen de una codificación activa](https://raw.githubusercontent.com/rajeevratan84/DeepLearningCV/master/hotoneencode.JPG)
# Además, mantenga las clases de números almacenadas como una variable, ```num_classess```


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print("Number of Classes: " + str(y_test.shape[1]))  # Number of Classes: 10

num_classes = y_test.shape[1]

# # **3. Construyendo nuestro modelo**
#
# Esta es la misma CNN que usamos anteriormente para el proyecto de clasificación MNIST.
#
# **Agregar capas de conversión**
#
# ```Conv2D(32, kernel_size=(3, 3),
# activación='releer',
# forma_entrada=forma_entrada)```
#
# Nuestro **Conv2D()** crea el filtro con los siguientes argumentos:
# - Número de filtros, usamos 32
# - kernel_size, usamos un filtro 3x3 por lo que se define como una tupla ```(3,3)```
# - activación, donde especificamos ```'relu'```
# - input_shape, que obtuvimos y almacenamos en una variable anterior, en nuestro ejemplo es una imagen en escala de
# grises de 28 x 28. Por lo tanto, nuestra forma es ```(28,28,1)```

# **Agregar capas de MaxPool**
#
# De nuevo, usamos ```model.add()``` y especificamos ```MaxPooling2D(pool_size=(2,2))```.
#
# Usamos el argumento de entrada pool_size para definir el tamaño de nuestra ventana. Podemos especificar la zancada y
# el relleno de esta manera:
#
# ```pool_size=(2, 2), strides=Ninguno, relleno='válido'```
#
# Sin embargo, tenga en cuenta que la zancada predeterminada se usa como el tamaño de la ventana de agrupación (2 en
# nuestro caso).
#
# Usar ```padding ='valid'``` significa que no usamos relleno.
#
# **Añadiendo Flatten**
#
# Usando model.add(Flatten()) simplemente estamos aplanando la salida de nuestro último nodo. Lo que equivale a
# 12 x 12 * 64 * 1 = 9216.
#
# **Agregar capas densas o completamente conectadas**
#
# ```modelo.add(Dense(128, activación='relu'))```
#
# Usamos ```model.add()``` una vez más y especificamos el número de nodos que nuestra capa anterior también conectará.
# También especificamos la función de activación de ReLU aquí.


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

# crear modelo
model = Sequential()

# Agrega nuestras capas usando model.add()

# Creamos una capa Conv2D con nuestras especificaciones
# Aquí estamos usando 32 filtros, de tamaño 3x3 con activación ReLU
# Nuestra forma de entrada es 28 x 28 x 1
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# Agregamos una segunda capa Conv con 64 filtros, 3x3 y activación ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Usamos un MaxPool estándar de 2x2 y un paso de 2 (por defecto, Kera usa el mismo paso que el tamaño de la ventana)
model.add(MaxPooling2D(pool_size=(2, 2)))
# Ahora aplanamos la salida de nuestras capas anteriores, que es 12 x 12 * 64 * 1 = 9216
model.add(Flatten())
# Ahora conectamos esto aplanado más tarde a 128 Nodos de nuestra Capa Completamente Conectada o Densa, nuevamente
# usando ReLU
model.add(Dense(128, activation='relu'))
# Ahora creamos nuestra última capa totalmente conectada/densa que consta de 10 nodos que corresponden a nuestras clases
# de salida
# Esto luego se usa con una activación 'softmax' para darnos nuestras probabilidades finales de clase
model.add(Dense(num_classes, activation='softmax'))

# ### **Compilando nuestro modelo**
#
# Aquí usamos ```model.compile()``` para compilar nuestro modelo. Especificamos lo siguiente:
# - Función de pérdida - categorical_crossentropy
# - Optimizer - SGD o Stochastic Gradient Descent (tasa de aprendizaje de 0.001 y momento 0.9)
# - métricas - con qué criterios evaluaremos el rendimiento. Usamos precisión aquí.

# Compilamos nuestro modelo, esto crea un objeto que almacena el modelo que acabamos de crear
# Configuramos nuestro Optimizer para usar Stochastic Gradient Descent (tasa de aprendizaje de 0.001)
# Configuramos nuestra función de pérdida para que sea categorical_crossentropy ya que es adecuada para problemas
# multiclase
# Finalmente, las métricas (en qué juzgamos nuestro desempeño) para ser precisión
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.001, momentum=0.9),
              metrics=['accuracy'])

# Podemos usar la función de resumen para mostrar las capas y los parámetros de nuestro modelo
print(model.summary())
'''Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       

 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     

 max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0         
 )                                                               

 flatten (Flatten)           (None, 9216)              0         

 dense (Dense)               (None, 128)               1179776   

 dense_1 (Dense)             (None, 10)                1290      

=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
'''

# # **4. Entrenando Nuestro Modelo**

# Establecer nuestro tamaño de lote y épocas
batch_size = 32
epochs = 15

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
''''
....
Epoch 15/15
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1147 - accuracy: 0.9577 - val_loss: 0.5426 - val_accuracy: 0.8744
Test loss: 0.5426446795463562
Test accuracy: 0.8744000196456909'''

# # **5. Ejemplo de aumento de datos**
#
# Usamos generadores porque no podemos cargar todo el conjunto de datos en la memoria de nuestros sistemas. Por lo
#  tanto, utilícelo para crear un iterador para que podamos acceder a lotes de nuestros datos para el aumento o
#  preprocesamiento de datos y propagarlos a través de nuestra red.


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Recargar nuestros datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reformar nuestros datos para que tengan el formato [número de muestras, ancho, alto, color_profundidad]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cambiar el tipo de datos a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Definir generador de datos para aumento
data_aug_datagen = ImageDataGenerator(rotation_range=30,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.2,
                                      zoom_range=0.1,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

# Crea nuestro iterador
aug_iter = data_aug_datagen.flow(x_train[0].reshape(1, 28, 28, 1), batch_size=1)

# #### **Mostrar los resultados de nuestro aumento de datos**


import cv2


def showAugmentations(augmentations=6):
    fig = plt.figure()
    for i in range(augmentations):
        a = fig.add_subplot(1, augmentations, i + 1)
        img = next(aug_iter)[0].astype('uint8')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()


showAugmentations(6)

# !/usr/bin/env python
# codificación: utf-8
########################################################
# 06 Keras -Fashion-MNIST Part 2 - con Regularizacion####
########################################################
#
# # **Regularización en Keras - Parte 2 - Con Regularización**
# ### **Primero entrenamos una CNN en el conjunto de datos Fashion-MNIST sin usar métodos de regularización**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)

# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** usando Keras con
# TensorFlow 2.0 y lo entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, con el uso de
# ningún método de regularización.

# 1. Cargando, Inspeccionando y Visualizando nuestros datos
# 2. Preprocesar nuestros datos y definir nuestro **Aumento de datos**
# 3. Construya una CNN simple con regularización
# - Regularización L2
# - Aumento de datos
#   - Abandonar
# - Norma de lote
# 4. Capacitar a nuestra CNN con Regularización


# # **Cargando, Inspeccionando y Visualizando nuestros datos**


# Cargamos nuestros datos directamente desde los conjuntos de datos incluidos en tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist

# carga el conjunto de datos de entrenamiento y prueba de Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Nuestros nombres de clase, al cargar datos de .datasets() nuestras clases son números enteros
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #### **Comprueba si estamos usando la GPU**


# Comprobar para ver si estamos usando la GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# ### **Inspeccionar nuestros datos**


# Mostrar el número de muestras en x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))

# Imprimir el número de muestras en nuestros datos
print("Number of samples in our training data: " + str(len(x_train)))
print("Number of labels in our training data: " + str(len(y_train)))
print("Number of samples in our test data: " + str(len(x_test)))
print("Number of labels in our test data: " + str(len(y_test)))

# Imprimir las dimensiones de la imagen y Nº. de etiquetas en nuestros datos de entrenamiento y prueba
print("\n")
print("Dimensions of x_train:" + str(x_train[0].shape))
print("Labels in x_train:" + str(y_train.shape))
print("\n")
print("Dimensions of x_test:" + str(x_test[0].shape))
print("Labels in y_test:" + str(y_test.shape))

# ### **Visualización de algunos de nuestros datos de muestra**
# Tracemos 50 imágenes de muestra.


# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

# Crear figura y cambiar tamaño
figure = plt.figure()
plt.figure(figsize=(16, 10))

# Establecer cuantas imágenes deseamos ver
num_of_images = 50

# iterar índice de 1 a 51
for index in range(1, num_of_images + 1):
    class_names = classes[y_train[index]]
    plt.subplot(5, 10, index).set_title(f'{class_names}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')

plt.show()

# # **2. Preprocesamiento de datos con ImageDataGenerator**
#
# Primero remodelamos y cambiamos nuestros tipos de datos como lo habíamos hecho anteriormente.


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras import backend as K

# Reformar nuestros datos para que tengan el formato [número de muestras, ancho, alto, color_profundidad]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cambiar el tipo de datos a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# **Recopilamos el tamaño y la forma de nuestra imagen y normalizamos nuestros datos de prueba**
# Usaremos ImageDataGenerator para normalizar y proporcionar aumentos de datos para nuestros **datos de entrenamiento**.

# Permite almacenar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# almacenar la forma de una sola imagen
input_shape = (img_rows, img_cols, 1)

# Normalizar nuestros datos entre 0 y 1
x_test /= 255.0

# ### **Una codificación en caliente de nuestras etiquetas**


from tensorflow.keras.utils import to_categorical

# Ahora codificamos las salidas en caliente
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Contemos las columnas de números en nuestra matriz codificada en caliente
print("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

## **3. Construyendo nuestro modelo**
#
# Esta es la misma CNN que usamos anteriormente para el proyecto de clasificación MNIST.


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

L2 = 0.001

# crear modelo, añadiendo la regularización
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_regularizer=regularizers.l2(L2),  # añadimos L2
                 input_shape=input_shape))
# model.add(BatchNormalization())  # añadimos normalización
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(L2)))  # añadimos L2
model.add(BatchNormalization())  # añadimos normalización
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # añadimos Dropout
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2)))  # añadimos L2
model.add(Dropout(0.2))  # añadimos Dropout
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9),
              metrics=['accuracy'])

print(model.summary())

# # **Entrenando Nuestro Modelo**

# Definir generador de datos para aumento
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# Aquí ajustamos el generador de datos a algunos datos de muestra.
# train_datagen.fit(x_train)

batch_size = 32
epochs = 30  # AUMENTADO PARA VER LA MEJORA tarda lo suyo

# Ajustar el modelo
# Tenga en cuenta que usamos train_datagen.flow, esto toma datos y etiqueta matrices, genera lotes de datos aumentados.
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    steps_per_epoch=x_train.shape[0] // batch_size)

# Obtenemos nuestra puntuación de precisión usando la función de evaluación
# La puntuación tiene dos valores, nuestra pérdida de prueba y precisión
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
Epoch 29/30
1875/1875 [==============================] - 7s 4ms/step - loss: 0.4028 - accuracy: 0.8880 - val_loss: 0.3576 - val_accuracy: 0.9069
Epoch 30/30
1875/1875 [==============================] - 12s 6ms/step - loss: 0.3941 - accuracy: 0.8900 - val_loss: 0.3723 - val_accuracy: 0.8964
Test loss: 0.3722538650035858
Test accuracy: 0.896399974822998'''

# !/usr/bin/env python
# codificación: utf-8
########################################################
# 07 PyTorch - Moda-MNSIT Parte 1 - Sin regularización##
########################################################

# # **Regularización en PyTorch - Parte 1**
# ### **Primero entrenamos una CNN en el conjunto de datos Fashion-MNIST sin usar métodos de regularización**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)

# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo
# entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, sin el uso de ningún método de
# regularización.

# 1. Importe bibliotecas de PyTorch, defina nuestros transformadores, cargue nuestro conjunto de datos y visualice
# nuestras imágenes.
# 2. Construya una CNN simple sin regularización
# 3. Capacitar a nuestra CNN
# 3. Eche un vistazo al aumento de datos
#
#

# # **Importar bibliotecas de PyTorch, definir transformadores y cargar y visualizar conjuntos de datos**
# Importar PyTorch
import torch
import PIL

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))  # GPU available: True
device = 'cuda'  # 'cpu' si no hay GPU disponible

# ### **Nuestra transformación de datos**

# Transforme a un tensor PyTorch y normalice nuestro valor entre -1 y +1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Cargue nuestros datos de entrenamiento y especifique qué transformación usar al cargar
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)

# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Crear una lista con los nombres de nuestras clases
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# En 20]:


# Tenemos 60 000 muestras de imágenes para nuestros datos de entrenamiento y 10 000 para nuestros datos de prueba
# cada 28 x 28 píxeles, ya que son en escala de grises, no hay una tercera dimensión en nuestra imagen
print(trainset.data.shape)
print(testset.data.shape)

# ### **Visualización de nuestros datos**

# Veamos las 50 primeras imágenes del conjunto de datos de entrenamiento del MNIST
import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 50

for index in range(1, num_of_images + 1):
    plt.subplot(5, 10, index)
    plt.axis('off')
    plt.imshow(trainset.data[index], cmap='gray_r')

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

# mostrar imagenes
imshow(torchvision.utils.make_grid(images))

# imprimir etiquetas
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

# # **2. Construyendo y entrenando nuestra CNN simple sin regularización**
# #### **Definiendo nuestro modelo**


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to(device)

# #### **Definición de nuestras funciones de pérdida y optimización**

# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# # **3. Entrenando Nuestro Modelo**

# Recorremos el conjunto de datos de entrenamiento varias veces (cada vez se denomina época)
epochs = 15

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
        loss = criterion(outputs,
                         labels)  # Get Loss (cuantificar la diferencia entre los resultados y las predicciones)
        loss.backward()  # Propagación hacia atrás para obtener los nuevos gradientes para todos los nodos
        optimizer.step()  # Actualizar los gradientes/pesos

        # Imprimir estadísticas de entrenamiento - Época/Iteraciones/Pérdida/Precisión
        running_loss += loss.item()
        if i % 100 == 99:  # mostrar nuestra pérdida cada 50 mini lotes
            correct = 0  # Inicializar nuestra variable para mantener el conteo de las predicciones correctas
            total = 0  # Inicializar nuestra variable para mantener el conteo del número de etiquetas iteradas

            # No necesitamos gradientes para la validación, así que envuélvalos
            # no_grad para ahorrar memoria
            with torch.no_grad():
                # Iterar a través del iterador testloader
                for data in testloader:
                    images, labels = data
                    # Mover nuestros datos a GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # Propagación hacia adelante de nuestro lote de datos de prueba a través de nuestro modelo
                    outputs = net(images)

                    # Obtenga predicciones del valor máximo
                    _, predicted = torch.max(outputs.data, 1)
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

# #### **Precisión de nuestro modelo**

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.4}%')

# #### **Nuestras Parcelas de Entrenamiento**

# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch Mini-Batches")
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

# #### **Guardando los pesos de nuestro modelo**

PATH = 'models/fashion_mnist_cnn_net.pth'
torch.save(net.state_dict(), PATH)

# # **4. Aumento de datos**
# Para introducir el aumento de datos en nuestros datos de entrenamiento, simplemente creamos nuevas funciones de
# transformación.
#
# **Recuerda nuestra función de transformación anterior**
#
# ```transform = transforma.Compose([transforma.ToTensor(),
# transforma.Normalizar((0.5, ), (0.5, )) ])```
#
# ### **Primero vamos a demostrar cómo el aumento de datos afecta nuestras imágenes**


# Importamos PIL, una biblioteca de procesamiento de imágenes para implementar rotaciones aleatorias
import PIL

data_aug_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15, resample = PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=1)
])

# #### **Realice el aumento de datos en una sola imagen usando la función a continuación para visualizar los efectos**

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread


def showAugmentations(image, augmentations=6):
    fig = plt.figure()
    for i in range(augmentations):
        a = fig.add_subplot(1, augmentations, i + 1)
        img = data_aug_transform(image)
        imshow(img, cmap='Greys_r')
        plt.axis('off')


# Cargue la primera imagen de nuestros datos de entrenamiento como una matriz numpy
image = trainset.data[0].numpy()

# Convertirlo a formato de imagen PIL
img_pil = PIL.Image.fromarray(image)
showAugmentations(img_pil, 8)

# !/usr/bin/env python
# codificación: utf-8
################################################################
# 08 PyTorch - Moda-MNSIT Parte 2 - Con Regularización.ipynb####
################################################################
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Regularización en PyTorch - Parte 2**
# ### **Ahora usamos algunos métodos de regularización en nuestro Fashion-MNIST CNN**
# En esta lección, primero aprendemos a crear un **modelo de red neuronal convolucional simple** en PyTorch y lo
# entrenamos para **clasificar imágenes en el conjunto de datos Fashion-MNIST**, ahora **CON** el uso de cualquier
# regularización métodos.

# 1. Importe bibliotecas de PyTorch, defina nuestros transformadores, cargue nuestro conjunto de datos y visualice
#    nuestras imágenes.
# 2. Cree una CNN simple con los siguientes métodos de **regularización**:
# - Regularización L2
# - Aumento de datos
#   - Abandonar
# - Norma de lote

# 3. Capacitar a nuestra CNN con Regularización
#

# # **1. Importe bibliotecas de PyTorch, defina transformadores y cargue y visualice conjuntos de datos**

# Importar PyTorch
import torch
import PIL
import numpy as np

# Usamos torchvision para obtener nuestro conjunto de datos y transformaciones de imágenes útiles
import torchvision
import torchvision.transforms as transforms

# Importar la biblioteca de optimización de PyTorch y nn
# nn se utiliza como bloque de construcción básico para nuestros gráficos de red
import torch.optim as optim
import torch.nn as nn

# ¿Estamos usando nuestra GPU?
print("GPU available: {}".format(torch.cuda.is_available()))

device = 'cuda'  # 'cpu' si no hay GPU disponible

# # **2. Construyendo una CNN con Regulación**
# ## **Implementación del aumento de datos**
#
# #### **Comprender lo que hacen nuestras transformaciones**
#
# 1. Nuestras transformaciones se aplican a una imagen o lote de imágenes cada vez que se carga.
# 2. Estas nuevas imágenes no se guardan, se generan o se 'alteran' cada vez que se carga un lote
#
# #### **NOTA**
#
# No aplicamos los mismos aumentos a nuestros conjuntos de datos de prueba o validación. Por lo tanto, mantenemos
# funciones de transformación separadas (ver más abajo) para nuestros datos de Entrenamiento y Validación/Prueba.


data_transforms = {
    'train': transforms.Compose([
        # Tenga en cuenta que estos se ejecutan en el orden en que se llaman aquí
        # Algunas de estas transformaciones devuelven una imagen en color, por lo que necesitamos convertir
        # la imagen vuelve a escala de grises
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),

        # Estas Siempre
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

# ### **Obtener y crear nuestros cargadores de datos**

# Cargue nuestros datos de tren y especifique qué transformación usar al cargar
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=data_transforms['train'])

# Cargue nuestros datos de prueba y especifique qué transformación usar al cargar
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=data_transforms['val'])

# Preparar el tren y probar el cargador
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# ### **Agregando Abandono**
# En las redes neuronales convolucionales, el abandono se agrega comúnmente después de las capas CONV-RELU.
#
# P.ej. CONV->RELU->**CAÍDA**
#
# #### **Recomendaciones de abandono**
#
# En los valores de CNN de 0.1 a 0.3 se ha encontrado que funcionan bien.
#
# ### **Añadiendo BatchNorm**
#
# En las CNN, **BatchNorm** se usa mejor entre la capa de conversión y la capa de función de activación (ReLU)
# Cuando se usa con Dropout, el orden recomendado es:
#
# CONV_1 -> **BatchNorm** -> ReLU -> Abandono - CONV_2
#
# **NOTA** El argumento de entrada de BatchNorm es el tamaño de **salida** de la capa anterior.

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Agregando BatchNorm, usando 32 como entrada ya que 32 fue la salida de nuestra primera capa Conv
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Agregando BatchNorm, usando 64 como entrada ya que 64 fue la salida de nuestra primera capa Conv
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        # Definiendo nuestra función de abandono con una tasa de 0.2
        # Podemos aplicar esto después de cualquier capa, pero es más adecuado después de ReLU
        self.dropOut = nn.Dropout(0.2)

    def forward(self, x):
        # a la salida de la convolución 1 le aplicamos la normalización y a eso la activación relu
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropOut(x)
        # a la salida de la convolución 2 le aplicamos la normalización y a eso la activación relu
        x = self.dropOut(F.relu(self.conv2_bn(self.conv2(x))))

        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to(device)

# ### **Añadir regularización L2**
# La regularización L2 sobre los parámetros/pesos del modelo se incluye directamente en la mayoría de los optimizadores,
# incluido optim.SGD.
#
# Se puede controlar con el parámetro **weight_decay** como se puede ver en la [documentación SGD]
# (http://pytorch.org/docs/optim.html#torch.optim.SGD).
#
# ```weight_decay``` (**flotante**, opcional) – disminución del peso *(penalización L2) (predeterminado: 0)*
#
# **Buenos valores de L2 oscilan entre 0,1 y 0,0001**
#
# **NOTA:**
#
# La regularización L1 no está incluida por defecto en los optimizadores, pero podría añadirse incluyendo un extra loss
# nn.L1Loss en los pesos del modelo.
#
#

# Importamos nuestra función de optimizador
import torch.optim as optim

# Usamos Cross Entropy Loss como nuestra función de pérdida
criterion = nn.CrossEntropyLoss()

# Para nuestro algoritmo de descenso de gradiente u Optimizer
# Usamos Stochastic Gradient Descent (SGD) con una tasa de aprendizaje de 0.001
# Establecemos el impulso en 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

# # **3. Entrenamiento de nuestro modelo utilizando métodos de regulación: aumento de datos, abandono, BatchNorm y
# regularización L2**


# Recorremos el conjunto de datos de entrenamiento varias veces (cada vez se denomina época)
epochs = 15

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
        loss = criterion(outputs,
                         labels)  # Get Loss (cuantificar la diferencia entre los resultados y las predicciones)
        loss.backward()  # Propagación hacia atrás para obtener los nuevos gradientes para todos los nodos
        optimizer.step()  # Actualizar los gradientes/pesos

        # Imprimir estadísticas de entrenamiento - Época/Iteraciones/Pérdida/Precisión
        running_loss += loss.item()
        if i % 100 == 99:  # mostrar nuestra pérdida cada 50 mini lotes
            correct = 0  # Inicializar nuestra variable para mantener el conteo de las predicciones correctas
            total = 0  # Inicializar nuestra variable para mantener el conteo del número de etiquetas iteradas

            # No necesitamos gradientes para la validación, así que envuélvalos
            # no_grad para ahorrar memoria
            with torch.no_grad():
                # Iterar a través del iterador testloader
                for data in testloader:
                    images, labels = data
                    # Mover nuestros datos a GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # Propagación hacia adelante de nuestro lote de datos de prueba a través de nuestro modelo
                    outputs = net(images)

                    # Obtenga predicciones del valor máximo
                    _, predicted = torch.max(outputs.data, 1)
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

# ### **Precisión de nuestros modelos**

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Mover nuestros datos a GPU
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.4}%')

import matplotlib.pyplot as plt

# Para crear una trama con un eje y secundario, necesitamos crear una subtrama
fig, ax1 = plt.subplots()

# Establecer el título y la rotación de la etiqueta del eje x
plt.title("Accuracy & Loss vs Epoch Mini-Batches")
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

'''Epoch: 15, Mini-Batches Completed: 1600, Loss: 0.540, Test Accuracy = 90.810%
Epoch: 15, Mini-Batches Completed: 1700, Loss: 0.555, Test Accuracy = 90.660%
Epoch: 15, Mini-Batches Completed: 1800, Loss: 0.526, Test Accuracy = 90.870%
Finished Training
Accuracy of the network on the 10000 test images: 90.94%'''

# #### **Detención anticipada en PyTorch**
#
# https://github.com/Bjarten/early-stopping-pytorch



