#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # ** PyTorch Lightning**
# ### **PyTorch Lightning es una biblioteca Python de código abierto que proporciona una interfaz de alto nivel para
# PyTorch, un popular marco de aprendizaje profundo.**
# ---
# https://pytorch-lightning.readthedocs.io/en/latest/
# ---
#
# En esta lección, aprendemos a usar la increíble biblioteca **PyTorch Lightning**. Es una excelente manera de
# organizarse. su código PyTorch y obtenga muchas funciones excelentes y beneficios adicionales. Haremos lo siguiente
# en esta guía:

# 1. Configurar e instalar Lightning
# 2. Organizar su código en la estructura/filosofía de diseño Lightning
# 3. Selección automática de lotes
# 4. Selección automática de tasa de aprendizaje
# 5. Entrenamiento usando Lightning
# 6. Registros de tensorboard
# 7. Devoluciones de llamadas: detención anticipada, puntos de control y uso de métricas Lightning Bolts
# 8. Guardar y cargar modelos desde puntos de control
# 9. Guardar como Torchscript para implementación en producción
# 10. Inferencias
# 12. Profiler - Performance and Bottleneck Profiler**

## 11. Entrenamiento Multi-GPU** y 13. Training on TPUs, no se pueden probar por no disponer ni de multiples gpu ni TPU


### **1. Configurar e instalar Lightning**


# Primero instalamos PyTorch Lightning y TorchMetrics
'''pip install pytorch-lightning --quiet'''
'''pip install torchmetrics'''


# Importar todos los paquetes que usaremos

import os
import torch
import torchmetrics
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from PIL import Image

#torch.set_float32_matmul_precision('high')
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# #### **Descargue nuestros conjuntos de datos** ( en 17)

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
test_dir = 'images/gatos_perros/test1'

# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# Crea nuestras transformaciones
transformations = transforms.Compose([transforms.Resize((60, 60)), transforms.ToTensor()])

# Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

# Dividirse en nuestro tren y validación
train, val = torch.utils.data.random_split(train,[20000,5000]) 

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


# Set directory paths for our files
train_dir = 'images/gatos_perros/train'
test_dir = 'images/gatos_perros/test1'

# Get files in our directories
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# Create our transforms
transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])

# Create our train and test dataset objects
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

# Split into our train and validation
train, val = torch.utils.data.random_split(train,[20000,5000])

# train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)
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


# Set directory paths for our files
train_dir = 'images/gatos_perros/train'
test_dir = 'images/gatos_perros/test1'

# Get files in our directories
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# Create our transforms
transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])

# Create our train and test dataset objects
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

# Split into our train and validation
train, val = torch.utils.data.random_split(train,[20000,5000])

train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)


# ## **2. Organizar su código en la estructura Lightning/filosofía de diseño**

class LitModel(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(64*5*5,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,2),)

    def train_dataloader(self):
        # transforma
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
      return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output,label)
        self.log('train_loss', loss)
        return {'loss': loss, 'log': self.log}

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self, x):
        # en lightning, forward define las acciones de predicción/inferencia
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,dim = 1) 



# ## **3. Selección automática de lotes**

model = LitModel(batch_size = 32)

trainer = pl.Trainer(auto_scale_batch_size=True)
# entrenador = pl.Entrenador(auto_scale_batch_size='binsearch')

trainer.tune(model)


# ## **4. Selección automática de tasa de aprendizaje**
#
# Edite el módulo Lightning como se muestra a continuación. Tenga en cuenta que hemos agregado nuevas líneas en las
# líneas 5 a 8.

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate #
        # error en .Acurracy() deprecado,
        # cambio por https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#module-interface
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2) #
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2) #
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2) #
        # Convoluciones
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(64*5*5,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,2),)

    def train_dataloader(self):
        # definimos una función que usando torch y el objeto dataset que hemos definido cargará y transoformará los
        # datos usando el tamaño de batch indicado
        # transforma
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        # del mismo modo definimos una función para la validación para la validación,
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
        # definimos una función para devolvernos una función de pérdida
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        # definimos una función de paso de entrenamiento
        data, label = batch # desempaquetamos desde el batch los datos y etiquetas
        output = self.forward(data)  # realizamos la inferencia realizando una pasada de la red
        loss = nn.CrossEntropyLoss()(output,label)  # calculamos mediante su función la pérdida
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
        # definimos la función de optimización, usanod Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        return optimizer

    def forward(self, x):
        # en lightning, forward define las acciones de predicción/inferencia
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,dim = 1) 


# ### **Implemente nuestro sintonizador de tasa de aprendizaje automático**

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

model = LitModel(batch_size = 32, learning_rate=0.001)

# Registrador de tasa de aprendizaje
# trainer = pl.Trainer(gpus=1, auto_lr_find=True) # por UserWarning Deprecado

trainer = pl.Trainer(accelerator='gpu', devices=1, auto_lr_find=True)

# Los resultados se pueden encontrar en
trainer.tune(model)


# ### **Visualizar el gráfico LR vs Loss**
# La figura producida por lr_finder.plot() debería parecerse a la figura de abajo. Se recomienda no elegir la tasa de
# aprendizaje que logra la pérdida más baja, sino algo en el medio de la pendiente descendente más pronunciada
# (punto rojo). Este es el punto devuelto por py lr_finder.suggestion().


lr_finder = trainer.tuner.lr_find(model)

# Parcela con
fig = lr_finder.plot(suggest=True)
fig.show()


### **5. Entrene el modelo con el tamaño de lote aprendido y la tasa de aprendizaje**

# La tasa de aprendizaje y el tamaño del lote almacenados en `/content/lr_find_temp_model.ckpt`
# y `/content/scale_batch_size_temp_model.ckpt` respectivamente, se usarán sobre el conjunto de la tasa de aprendizaje
# y los tamaños de lote que establecemos.

# modelo inicial
model = LitModel(batch_size = 32, learning_rate=0.001)

# Inicializar un entrenador
# trainer = pl.Trainer(gpus=1, max_epochs=10) # por UserWarning Deprecado

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)

# Entrena al modelo ⚡
trainer.fit(model)


# ## **Registros de Tensorboard**


'''
# Iniciar tensorboard.
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')
'''
'''from tensorboard import program

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'lightning_logs'])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")'''

# ## **6. Uso de devoluciones de llamada - Detención anticipada y puntos de control**
#
# **Detención anticipada**: la detención anticipada es una forma de regularización utilizada para evitar el sobreajuste
# cuando se entrena a un alumno con un método iterativo, como el descenso de gradiente.
#
# ![](https://cdn-images-1.medium.com/max/920/1*iAK5uMoOlX1gZu-cSh1nZw.png)
#
# **Punto de control del modelo**: la devolución de llamada de ModelCheckpoint se usa para guardar un modelo o pesos
# (en un archivo de punto de control) en algún intervalo, de modo que el modelo o los pesos se puedan cargar más tarde
# para continuar el entrenamiento desde el estado guardado.


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
    dirpath='models/',
    filename='sample-catsvsdogs-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,  # Guardamos los 3 mejores modelos
    mode='min',
)


# Incluso podemos usar algunas devoluciones de llamadas personalizadas
# class MyPrintingCallback(pl.callbacks.Callback):  # class MyPrintingCallback(pl.callbacks.base.Callback) Deprecado
#     # RuntimeError: The `on_init_start` callback hook was deprecated in v1.6 and is no longer supported as of v1.8.
#     # def on_init_start(self, trainer):
#     #     print('Starting to init trainer!')
#
#     def on_init_end(self, trainer):
#         print('trainer is init now')
#
#     def on_train_end(self, trainer, pl_module):
#         print('do something when training ends')


# ### **Entrena con nuestras devoluciones de llamadas**


# modelo inicial
model = LitModel(batch_size = 32, learning_rate=0.001)

# Inicializar un entrenador
trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    # callbacks=[EarlyStopping('val_loss'), checkpoint_callback, MyPrintingCallback()]
    callbacks=[EarlyStopping('val_loss'), checkpoint_callback]
)

# por UserWarning Deprecado

trainer.fit(model)

'''# Iniciar tensorboard.
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')'''


# ## **8. Restaurar desde puntos de control**

# Obtener la ruta del mejor modelo
checkpoint_callback.best_model_path


# ### **Cargar y ejecutar la inferencia usando el mejor modelo de punto de control**


# cargando los mejores puntos de control para modelar
pretrained_model = LitModel.load_from_checkpoint(batch_size = 32, learning_rate=0.001, checkpoint_path = checkpoint_callback.best_model_path)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()
pretrained_model.freeze()


# ## **9. Guarde nuestro modelo para implementaciones de producción**
#
# **Exportando a TorchScript**
#
# TorchScript le permite serializar sus modelos de manera que pueda cargarse en entornos que no sean de Python.
# LightningModule tiene un método útil to_torchscript() que devuelve un módulo con script que puede guardar o usar
# directamente.


model = LitModel.load_from_checkpoint(batch_size = 32, learning_rate=0.001, checkpoint_path = checkpoint_callback.best_model_path)

script = model.to_torchscript()

# guardar para usar en el entorno de producción
torch.jit.save(script, "model.pt")


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

### **12. Profiler - Perfilador de rendimiento y cuellos de botella**


# modelo inicial
# model = LitModel_mGPU()

# Inicializar un entrenador
'''trainer = pl.Trainer( # por UserWarning Deprecado 
    gpus=1,
    max_epochs=1,
    progress_bar_refresh_rate=10,
    profiler="simple"
)'''
trainer = pl.Trainer(
    accelerator='gpu', devices=1,
    max_epochs=1,
    profiler="simple"
)
# trainer = pl.Trainer(gpus=1, max_epochs=10)


## 11. Entrenamiento Multi-GPU** y 13. Training on TPUs, no se pueden probar por no disponer ni de multiples gpu ni TPU
'''
### **11. Entrenamiento Multi-GPU**
#
# Para entrenar en CPU/GPU/TPU sin cambiar su código, necesitamos desarrollar algunos buenos hábitos :)
#
# Eliminar todas las llamadas `.cuda()` o `.to()`.
#
# **Sincronizar validación y registro de prueba**
#
# Cuando se ejecuta en modo distribuido, debemos asegurarnos de que las llamadas de registro de pasos de validación y prueba estén sincronizadas entre procesos. Esto se hace agregando sync_dist=True a todas las llamadas self.log en el paso de validación y prueba. Esto garantiza que cada trabajador de la GPU tenga el mismo comportamiento al realizar un seguimiento de los puntos de control del modelo, lo cual es importante para tareas posteriores posteriores, como probar el mejor punto de control entre todos los trabajadores.
#
# Tenga en cuenta que si utiliza métricas integradas o métricas personalizadas que usan la API de métricas, no es necesario actualizarlas y se gestionan automáticamente.
#
# def validación_paso(self, lote, lote_idx):
#         x, y = batch
# logits = self(x)
# pérdida = self.loss(logits, y)
# # Agregue sync_dist=True para sincronizar el registro en todos los trabajadores de GPU
# self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
#
# def test_step(self, lote, lote_idx):
#         x, y = batch
# logits = self(x)
# pérdida = self.loss(logits, y)
# # Agregue sync_dist=True para sincronizar el registro en todos los trabajadores de GPU
# self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
#
# Hay otras buenas prácticas que no usamos aquí, pero se pueden encontrar aquí: https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html

# En[ ]:


class LitModel_mGPU(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2,2)) 
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(64*5*5,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,2),)

    def train_dataloader(self):
        # transforma
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
      return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output,label)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(output, label))
        return {'loss': loss, 'log': self.log}

    def training_epoch_end(self, outs):
        # métrica de época de registro
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outs):
        # métrica de época de registro
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        return optimizer

    def forward(self, x):
        # en lightning, forward define las acciones de predicción/inferencia
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,dim = 1) 


# ###**Dispositivos GPU seleccionados**
# Puede seleccionar los dispositivos GPU usando rangos, una lista de índices o una cadena que contiene una lista separada por comas de ID de GPU:

# En[ ]:


# DEFAULT (int) especifica cuántas GPU usar por nodo
#pl.Entrenador(gpus=k)

# Arriba es equivalente a
#pl.Trainer(gpus=lista(rango(k)))

# Especifique qué GPU usar (no usar cuando se ejecuta en un clúster)
#pl.Entrenador(gpus=[0, 1])

# Equivalente usando una cadena
#pl.Entrenador(gpus='0, 1')

# Para usar todas las GPU disponibles, ponga -1 o '-1'
# equivalente a lista(rango(torch.cuda.device_count()))
#pl.Entrenador(gpus=-1)


# #### **Nota: En Colab solo tenemos una GPU, así que esto no acelerará las cosas aquí**

# En[ ]:


# modelo inicial
model = LitModel_mGPU()

# Inicializar un entrenador
trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=10,
    callbacks=[EarlyStopping('val_loss'), checkpoint_callback, MyPrintingCallback()]
)

trainer.fit(model)

### **12. Profiler - Perfilador de rendimiento y cuellos de botella**


# modelo inicial
model = LitModel_mGPU()

# Inicializar un entrenador
trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    progress_bar_refresh_rate=10,
    profiler="simple"
)


trainer.fit(model)

'''
'''### **13. Capacitación en TPU**
#
# **Unidad de procesamiento de tensor** es un circuito integrado específico de la aplicación del acelerador de IA desarrollado por Google específicamente para el aprendizaje automático de redes neuronales.
#
# **Terminología de la TPU**
#
# Una TPU es una unidad de procesamiento Tensor. Cada TPU tiene 8 núcleos donde cada núcleo está optimizado para multiplicaciones de matriz de 128x128. En general, ¡una sola TPU es tan rápida como 5 GPU V100!
#
# Un pod de TPU aloja muchos TPU en él. ¡Actualmente, TPU pod v2 tiene 2048 núcleos! Puede solicitar un pod completo de la nube de Google o una "porción" que le brinda un subconjunto de esos 2048 núcleos.

# ### **Pasos para entrenar en TPU**
#
# 1. Cambiar el tiempo de ejecución a TPU
# 2. Instalar PyTorch TPU
# 3. Vuelva a instalar PyTorch Lightning y TorchMetrics (si es necesario) ya que se ha restablecido el tiempo de ejecución.

# En[ ]:

# pip install -U torch_xla
# Importar todos los paquetes que usaremos

import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torchmetrics
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.metrics import funcional como FM
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image

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
test_dir = 'images/gatos_perros/test1'

# Obtener archivos en nuestros directorios
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

transformations = transforms.Compose([transforms.Resize((60,60)),transforms.ToTensor()])

# Crear nuestros objetos de conjunto de datos de tren y prueba
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

train, val = torch.utils.data.random_split(train,[20000,5000])

train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)


# En[ ]:


class LitModel_mGPU(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)  #
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)  #
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)  #
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2,2))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(64*5*5,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,2),)

    def train_dataloader(self):
        # transforma
        return torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=False)

    def cross_entropy_loss(self, logits, labels):
      return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output,label)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(output, label))
        return {'loss': loss, 'log': self.log}

    def training_epoch_end(self, outs):
        # métrica de época de registro
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outs):
        # métrica de época de registro
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        return optimizer

    def forward(self, x):
        # en lightning, forward define las acciones de predicción/inferencia
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,dim = 1)




# modelo inicial
model = LitModel_mGPU()

# Inicializar un entrenador
trainer = pl.Trainer(
    accelerator="tpu", devices=1,
    tpu_cores=8,
    max_epochs=1
)

trainer.fit(model)
'''