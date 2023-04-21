#!/usr/bin/env python
# codificación: utf-8

# # **PyTorch Lightning - Transferencia de aprendizaje**
# ---
# https://pytorch-lightning.readthedocs.io/en/1.2.0/advanced/transfer_learning.html
# ---
#
# En esta lección, aprendemos a usar la increíble biblioteca **PyTorch Lightning**. Es una excelente manera de
# organizarse. su código PyTorch y obtenga muchas funciones excelentes y beneficios adicionales. Haremos lo siguiente
# en esta guía:
# 1. Configurar e instalar Lightning
# 2. Cree nuestra clase de modelo Lightning y use un modelo pre-entrenado
# 3. Entrena a nuestro modelo

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

# >>> m.weight
tensor([[-0.4500,  0.5856],
        [-0.1807, -0.4963]])

# >>> m.bias
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

# Explicación Torchmetrics
'''     Una parte crítica de una buena capacitación es administrar adecuadamente las métricas de validación. Esto es 
        necesario para monitorear qué estado del modelo está funcionando mejor. Diferentes tareas requieren diferentes 
        métricas para evaluar la precisión del modelo y su implementación generalmente requiere escribir un código 
        repetitivo. TorchMetrics aborda este problema proporcionando un enfoque modular para definir y rastrear todas 
        las métricas de evaluación.
        
        Antes de usarse, la tenemos que instanciar ( loq ue estamos haciendo)***
         Después de eso, por cada lote leído del cargador de datos, el objetivo y la predicción se pasan al objeto de 
         métrica. Luego, calcula el resultado de la métrica para el lote actual y lo guarda en su estado interno, 
         que realiza un seguimiento de los datos vistos hasta el momento. Cuando los lotes están terminados, es posible 
         retirar del objeto métrico el resultado final.

        Cada métrica hereda los siguientes métodos de la Metricclase:

        - metric.forward(preds, target)— calcula la métrica utilizando predsy target. son la predicción y el 
                                        objetivo del lote actual. Luego, actualiza el estado de la métrica y 
                                        devuelve el resultado de la métrica. Como atajo, es posible usar 
                                        metric(preds, target), no hay diferencia entre las dos sintaxis.
        - metric.update(preds, target)— lo mismo que adelante pero sin devolver el resultado métrico de eficiencia.
                                        Si no es necesario registrar o imprimir el resultado de la métrica para cada 
                                        lote, este es el método que debe usarse porque es más rápido.            
        - metric.compute()— devuelve el resultado calculado sobre todos los datos vistos hasta el momento. Debe llamarse 
                            después del final de todos los lotes.
        - metric.reset() borra el estado de la métrica. Tiene que ser llamado al final de cada fase de validación.
        
        Resumiendo:

        Para cada lote, llame forwardo update.
        Fuera del ciclo del cargador de validación, llame computepara obtener el resultado final.
        Al final, llame resetpara borrar el estado de la métrica.
        
        ejemplo Uso de TorchMetrics con PyTorch Lightning:
        
        TorchMetrics es una buena combinación con PyTorch Lightning para reducir aún más el código repetitivo. Si nunca 
        ha oído hablar de PyTorch Lightning, es un marco para simplificar la codificación de modelos. Para obtener más 
        información, consulte su sitio web . Si no usa PyTorch Lightning, omita esta sección.

        A través del método PL self.log_dict(collection, on_step, on_epoch)es posible registrar el objeto de colección 
        de métricas. Al registrar el objeto de métrica, PyTorch Lightning se encarga de cuándo calcular o restablecer 
        la métrica.
        
        Configúrelo on_step=True para registrar las métricas de cada lote. Configúrelo on_epoch=Truepara registrar las 
        métricas de época (por lo tanto, los resultados calculados sobre todos los lotes). Si ambos están configurados 
        en True step metrics y epoch ambos se registran. Para obtener más información sobre PyTorch Lightning Logging, 
        consulte su documentación .

        El siguiente código muestra un módulo PyTorch Lightning que usa TorchMetrics para manejar las métricas:

        import torch
        import torchmetrics
        
        class MyAccuracy(Metric):
            def __init__(self):
                super().__init__()
                # to count the correct predictions
                self.add_state('corrects', default=torch.tensor(0))
                # to count the total predictions
                self.add_state('total', default=torch.tensor(0))
        
            def update(self, preds, target):
                # update correct predictions count
                self.correct += torch.sum(preds == target)
                # update total count. numel() returns the total number of elements 
                self.total += target.numel()
        
            def compute(self):
                # final computation
                return self.correct / self.total * 100
                
        De esta forma, durante la etapa de entrenamiento, las métricas se registran para cada lote. Mientras que durante
         la etapa de validación, las métricas de paso se acumulan para registrar solo las métricas finales al final de 
         la época. En este caso, los métodos computey resetno se llaman porque PyTorch Lightning los maneja solo.

        Para imprimir las métricas al final de cada validación, se debe agregar este código:

        def validation_epoch_end(self, outputs): 
            resultados = metric_collection.compute() 
            print(resultados) 
            self.metric_collection.reset()

'''
# Explicación Eval()
'''```net.eval()``` es un tipo de interruptor para algunas capas/partes específicas del modelo que se comportan de 
manera diferente durante el tiempo de entrenamiento e inferencia (evaluación). Por ejemplo, Dropouts Layers, BatchNorm 
Layers, etc. Debe desactivarlas durante la evaluación del modelo y .eval() lo hará por usted. Además, la práctica común 
para evaluar/validar es usar torch.no_grad() junto con model.eval() para desactivar el cálculo de gradientes:
'''


### **1. Configurar e instalar Lightning**



# Primero instalamos PyTorch Lightning y TorchMetricsß
'''pip install pytorch-lightning --quiet'''
'''pip install torchmetrics'''


# Importar todos los paquetes que usaremos
import os
import torch
import torchmetrics
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image

#torch.set_float32_matmul_precision('high')
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# #### **Descargue nuestros conjuntos de datos (en 17)


# ## **Configurar nuestros cargadores de datos**


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


# Establecer rutas de directorio para nuestros archivos
train_dir = 'images/gatos_perros/train'
test_dir = 'images/gatos_perros/train'

# ### **Acerca de los datos de prueba y entrenamiento**
#

# Hay dos subconjuntos de datos que se utilizan aquí:
# - **Datos de entrenamiento** Datos que se usan para optimizar los parámetros del modelo (usados durante el
#                              entrenamiento)
# - **Datos de prueba/validación** Datos que se utilizan para evaluar el rendimiento del modelo

# Durante el entrenamiento, monitoreamos el rendimiento del modelo en los datos de prueba.

# **Buena práctica de aprendizaje automático**
# A menudo mantenemos otro **conjunto de prueba** para probar el modelo final a fin de obtener una estimación imparcial
# de la precisión *fuera de la muestra*.


# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

#
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
# EJEMPLO :transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])  # aqui no normalizamos


# **#  Cargue nuestros datos de entrenamiento y prueba  y especifique qué transformación usar al cargar
# 1. Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

train, val = torch.utils.data.random_split(train,[20000,5000])

# 2. Cargue nuestros datos de entrenamiento y especifique qué transformación usar al cargar
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)







# ## **2.Creación de nuestro módulo Lightning usando un modelo preentrenado (ImageNet)**
#
# ¡Usar modelos previamente entrenados para Transfer Learning es simple!
#
# Todo lo que hacemos es cargar los pesos del modelo en la función init. Aquí usamos **resNet50**.

import torchvision.models as models

class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        # super es una subclase de pl.LightningModule y hereda todos sus métodos
        super().__init__()

        # instnaicamos la métrica de accuracy ( explicación torchmetrics arriba)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        # iniciar un resnet preentrenado
        ''' estamos cargando una red de red de imagen pre-entrenada. En este caso, estamos utilizando un resnet.
        Mantenemos todas las capas convolucionales congeladas  en la red y reemplazamos el encabezado con una capa 
        diferente, sinedo lo que recuperamos. sí Verizon aquí definitivamente parte de la clase.'''
        backbone = models.resnet50(weights=True)
        # Accedemos a las capas totalmente conectadas y obtenemos el número de filtros de salida.
        num_filters = backbone.fc.in_features
        # obtener todas las capas aquí menos la última porque tiene los detalles y las salidas, y creamos este
        # extractor de características aquí usándolo y usamos nn.secuencial.
        # Mientras que nn.Module es la clase base para implementar modelos PyTorch, nn.Sequential ( explicación arriba)
        layers = list(backbone.children())[:-1]
        '''no tenemos una red con las capas individualmente como en la defincdón de una nueva red, sino que Hemos
        extraído una característica, que es básicamente un nodo del que definimod mediante 
        self.feature_extractor = nn.Sequential(*layers) .
        Este tipo de comando nos permite usar esto en diferentes partes de la clase para que podamos usarlo en el bucle 
        de avance.(foward)'''
        self.feature_extractor = nn.Sequential(*layers)  # básicamente reconstruye el modelo de manera efectiva
        # usar el modelo preentrenado
        num_target_classes = 2  # como solo tenemos 2 clases lo establecemos
        # estamos estableciendo una red neuronal para realizar transfer learning con extracción de características
        # con la red preentrenada pero en la ultima capa totalmente conectada que tiene los detalles y salidas
        # cambiandolo por nuestras 2 clases objetivo
        # nn.linear Aplica una transformación lineal a los datos entrantes (explicación arriba)
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # en lightning, forward define las acciones de predicción/inferencia
        # aquí definimos nuestra secuencia de propagación hacia adelante, se hace la canalización, sin embargo para
        # transfer learning:
        # eval() explicado arriba
        ''' configuramos esto en modo de evaluación. Siendo como congelamos la forma en que está, usando torch.nograd
        que básicamente, nos permite usar este modelo congelado, donde simplemente aplanamos a la salida'''
        # Cambiar nuestra función de reenvío para incluir las 4 líneas a continuación
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)  # después, alimentamos el modelo congelado a la capa final,
        # tras lo que tenemos las actualizaciones de la capa soft max.'''
        return F.softmax(x,dim = 1) 

    def train_dataloader(self):
        # definimos una función que usando torch y el objeto dataset que hemos definido cargará y transoformará los
        # datos usando el tamaño de batch indicado
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        # del mismo modo definimos una función para la validación para la validación,
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
        # definimos una función para devolvernos una función de pérdida
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        # definimos una función de paso de entrenamiento
        data, label = batch  # desempaquetamos desde el batch los datos y etiquetas
        output = self.forward(data)  # realizamos la inferencia realizando una pasada de la red
        loss = nn.CrossEntropyLoss()(output, label)  # calculamos mediante su función la pérdida
        self.log('train_loss', loss)  # añadimos la perdida a las métricas
        self.log('train_acc_step', self.accuracy(output, label))  # calculamos y añadimos la medida de accuracy
        return {'loss': loss, 'log': self.log}  # devolvemos las métricas

    def training_epoch_end(self, outs):
        # métrica de época de registro
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        # función de paso de la red pero de validación
        val_data, val_label = batch  # desempaquetamos desde el batch los datos y etiquetas
        val_output = self.forward(val_data)  # # realizamos la inferencia realizando una pasada de la red
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)  # # calculamos mediante su función la pérdida
        self.log('val_acc_step', self.accuracy(val_output, val_label))  # calculamos y añadimos la medida de accuracy
        self.log('val_loss', val_loss)  # añadimos la perdida a las métricas

    def validation_epoch_end(self, outs):
        # métrica de época de registro
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        # optimizador = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# ## **Empezar a entrenar - Exactamente igual que antes**


model = ImagenetTransferLearning()


# Añadimos funcionalidades,

# Configuración de parada anticipada
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    strict = False,
    verbose = False,
    mode = 'min'
)



# Punto de control del modelo de configuración
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='catsvsdogs-transfer',  # ponerle la epoca y perdida al nombre del ckpt {epoch:02d}-{val_loss:.2f}',
    save_top_k=3,  # Guardamos los 3 mejores modelos
    mode='min',
)



trainer = pl.Trainer(
    accelerator='gpu', devices=1,
    # max_epochs=10,
    callbacks=[EarlyStopping('val_loss'), checkpoint_callback], # Añadido por mi
    default_root_dir="checkpoints/"
)
trainer.fit(model)


# ## **8. Restaurar desde puntos de control**

# Obtener la ruta del mejor modelo
checkpoint_callback.best_model_path


# ### **Cargar y ejecutar la inferencia usando el mejor modelo de punto de control**


# cargando los mejores puntos de control para modelar
pretrained_model = ImagenetTransferLearning.load_from_checkpoint(batch_size = 32, learning_rate=0.001, checkpoint_path = checkpoint_callback.best_model_path)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()
pretrained_model.freeze()



# ## **10. Ejecute la inferencia en 32 imágenes de nuestro registrador de datos de prueba**

import torchvision
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


samples, _ = next(iter(val_loader))
samples = samples.to('cuda')

fig = plt.figure(figsize=(12, 8))
fig.tight_layout()

output = pretrained_model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}

for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
plt.show()

# CARGAR CHECKPOINT
model = ImagenetTransferLearning.load_from_checkpoint("checkpoints/catsvsdogs-transfer.ckpt").cuda()

# disable randomness, dropout, etc...
model.eval()

# predict with the model
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}

for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
plt.show()

# ## **9 PRUEBAS GUARDAR
# SEGUARDA CADA CHECKPOINT
try:
    # guardar para usar en el entorno de producción PARA INFERENCIA
    torch.save(model.state_dict(), "models/model_Transfer.pt")
except Exception as e:
    print(e)


'''
*** principal
# https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html
https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html


https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
'''


