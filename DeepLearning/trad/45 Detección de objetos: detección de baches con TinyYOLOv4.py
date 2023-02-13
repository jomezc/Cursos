#!/usr/bin/env python
# codificación: utf-8

#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# ## **Entrenando a Tiny YOLOv4 usando un conjunto de datos de baches**
#
# ## Introducción
#
#
# En este cuaderno, implementamos la versión diminuta de [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) para entrenar a nuestro público [Pot Hole Dataset] que consta de 665 imágenes (https://public. roboflow.com/object-detection/pothole), [YOLOv4 tiny](https://github.com/AlexeyAB/darknet/issues/6067).
#
# También recomendamos leer nuestra publicación de blog sobre [Entrenamiento de YOLOv4 en datos personalizados](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) en paralelo.
#
# Seguiremos los siguientes pasos para implementar YOLOv4 en nuestros datos personalizados:
# * Configurar nuestro entorno de GPU en Google Colab
# * Instalar el entorno de entrenamiento Darknet YOLOv4
# * Descargue nuestro conjunto de datos personalizado para YOLOv4 y configure directorios
# * Configure un archivo de configuración de entrenamiento YOLOv4 personalizado para Darknet
# * Entrena nuestro detector de objetos personalizado YOLOv4
# * Vuelva a cargar los pesos entrenados de YOLOv4 y haga inferencias en las imágenes de prueba
#
# Cuando haya terminado, tendrá un detector personalizado que puede usar. Hará una inferencia como esta:
#
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/pothole.png)
#
#
# ### **Pida ayuda**
#
# Si se encuentra con algún obstáculo en su propio conjunto de datos o simplemente quiere compartir algunos resultados interesantes en su propio dominio, [¡comuníquese!](https://roboflow.ai)
#
#
#
# #### ![Marca de trabajo de Roboflow](https://i.imgur.com/WHFqYSJ.png)

# # Configuración de cuDNN en Colab para YOLOv4
#
#

# En[ ]:


# CUDA: Verifiquemos que los controladores Nvidia CUDA ya estén preinstalados y qué versión es.
get_ipython().system('/usr/local/cuda/bin/nvcc --version')
# Necesitamos instalar el cuDNN correcto de acuerdo con esta salida


# En[ ]:


#echa un vistazo al tipo de GPU que tenemos
get_ipython().system('nvidia-smi')


# En[ ]:


# Esta celda garantiza que tenga la arquitectura correcta para su GPU respectiva
# Si no se encuentra su comando, mire a través de estas GPU, busque el respectivo
# GPU y agréguelos al diccionario archTypes

#Tesla V100
# ARCO= -gencode arco=computar_70,código=[sm_70,computar_70]

#Tesla K80
# ARCO= -gencode arco=compute_37,código=sm_37

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCO= -gencode arco=computar_75,código=[sm_75,computar_75]

# Supersónico XAVIER
# ARCO= -gencode arco=computar_72,código=[sm_72,computar_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titán Xp, Tesla P40, Tesla P4
# ARCO= -gencode arco=compute_61,código=sm_61

# GP100/Tesla P100 - DGX-1
# ARCO= -gencode arco=compute_60,código=sm_60

# Para Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - descomentar:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# Para Jetson Tx2 o Drive-PX2 descomentar:
# ARCO= -gencode arco=computar_62,código=[sm_62,computar_62]
import os
os.environ['GPU_TYPE'] = str(os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read())

def getGPUArch(argument):
  try:
    argument = argument.strip()
    # Todas las GPU de Colab
    archTypes = {
        "Tesla V100-SXM2-16GB": "-gencode arch=compute_70,code=[sm_70,compute_70]",
        "Tesla K80": "-gencode arch=compute_37,code=sm_37",
        "Tesla T4": "-gencode arch=compute_75,code=[sm_75,compute_75]",
        "Tesla P40": "-gencode arch=compute_61,code=sm_61",
        "Tesla P4": "-gencode arch=compute_61,code=sm_61",
        "Tesla P100-PCIE-16GB": "-gencode arch=compute_60,code=sm_60"

      }
    return archTypes[argument]
  except KeyError:
    return "GPU must be added to GPU Commands"
os.environ['ARCH_VALUE'] = getGPUArch(os.environ['GPU_TYPE'])

print("GPU Type: " + os.environ['GPU_TYPE'])
print("ARCH Value: " + os.environ['ARCH_VALUE'])


# # Instalación de Darknet para YOLOv4 en Colab
#
#
#

# En[ ]:


get_ipython().run_line_magic('cd', '/content/')
get_ipython().run_line_magic('rm', '-rf darknet')


# En[ ]:


#clonamos la bifurcación de darknet mantenida por roboflow
Se han realizado #pequeños cambios para configurar darknet para entrenamiento
get_ipython().system('git clone https://github.com/roboflow-ai/darknet.git')


# En[ ]:


#instalar entorno desde el Makefile
get_ipython().run_line_magic('cd', '/content/darknet/')
# computar_37, sm_37 para Tesla K80
# computar_75, sm_75 para Tesla T4
# !sed -i 's/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= -gencode arch=compute_75,code=sm_75/g' Makefile

#instalar entorno desde el Makefile
#note si está en Colab Pro, esto funciona en una GPU P100
#si está en Colab gratis, es posible que deba cambiar el Makefile para la GPU K80
#esto se aplica a cualquier GPU, debe cambiar el Makefile para informar a darknet en qué GPU se está ejecutando.
get_ipython().system("sed -i 's/OPENCV=0/OPENCV=1/g' Makefile")
get_ipython().system("sed -i 's/GPU=0/GPU=1/g' Makefile")
get_ipython().system("sed -i 's/CUDNN=0/CUDNN=1/g' Makefile")
get_ipython().system('sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= ${ARCH_VALUE}/g" Makefile')
get_ipython().system('make')


# En[ ]:


#descargue las pesas pequeñas yolov4 recién lanzadas
get_ipython().run_line_magic('cd', '/content/darknet')
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights')
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29')


# # Configure un conjunto de datos personalizado para YOLOv4

# Usaremos Roboflow para convertir nuestro conjunto de datos de cualquier formato al formato YOLO Darknet.
#
# 1. Para hacerlo, cree una [cuenta Roboflow] gratuita (https://app.roboflow.ai).
# 2. Sube tus imágenes y sus anotaciones (en cualquier formato: VOC XML, COCO JSON, TensorFlow CSV, etc).
# 3. Aplique los pasos de preprocesamiento y aumento que le gusten. Recomendamos al menos una "orientación automática" y un "cambio de tamaño" a 416x416. Genera tu conjunto de datos.
# 4. Exporte su conjunto de datos en el **formato YOLO Darknet**.
# 5. Copie su enlace de descarga y péguelo a continuación.
#
# Consulte nuestra [publicación de blog](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) para obtener más detalles.
#
# En este ejemplo, utilicé el [conjunto de datos BCCD] de código abierto (https://public.roboflow.ai/object-detection/bccd). (Puede "bifurcarlo" en su cuenta de Roboflow para seguirlo).

# En[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Pothole.v1-raw.darknet.zip')
get_ipython().system('unzip -q Pothole.v1-raw.darknet.zip')


# En[ ]:


#si ya tiene el formato darknet de YOLO, puede omitir este paso
#de lo contrario recomendamos formatear en Roboflow
get_ipython().run_line_magic('cd', '/content/darknet')
get_ipython().system('curl -L "https://public.roboflow.com/ds/I2ZXTaUHUY?key=BBFNVcFack" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip')


# En[ ]:


#Configurar directorios de archivos de entrenamiento para conjuntos de datos personalizados
get_ipython().run_line_magic('cd', '/content/darknet/')
get_ipython().run_line_magic('cp', 'train/_darknet.labels data/obj.names')
get_ipython().run_line_magic('mkdir', 'data/obj')
#copiar imagen y etiquetas
get_ipython().run_line_magic('cp', 'train/*.jpg data/obj/')
get_ipython().run_line_magic('cp', 'valid/*.jpg data/obj/')

get_ipython().run_line_magic('cp', 'train/*.txt data/obj/')
get_ipython().run_line_magic('cp', 'valid/*.txt data/obj/')

with open('data/obj.data', 'w') as out:
  out.write('classes = 3\n')
  out.write('train = data/train.txt\n')
  out.write('valid = data/valid.txt\n')
  out.write('names = data/obj.names\n')
  out.write('backup = backup/')

#escribir archivo de tren (solo la lista de imágenes)
import os

with open('data/train.txt', 'w') as out:
  for img in [f for f in os.listdir('train') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

#escribe el archivo válido (solo la lista de imágenes)
import os

with open('data/valid.txt', 'w') as out:
  for img in [f for f in os.listdir('valid') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')


# # Escribir una configuración de entrenamiento personalizada para YOLOv4

# En[ ]:


#construimos la configuración dinámicamente en función del número de clases
#construimos iterativamente a partir de archivos de configuración base. Esta es la misma forma de archivo que cfg/yolo-obj.cfg
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

num_classes = file_len('train/_darknet.labels')
max_batches = num_classes*2000
steps1 = .8 * max_batches
steps2 = .9 * max_batches
steps_str = str(steps1)+','+str(steps2)
num_filters = (num_classes + 5) * 3


print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

#Instrucciones del repositorio de darknet
#cambie la línea max_batches a (clases*2000 pero no menos del número de imágenes de entrenamiento, y no menos de 6000), ej. max_batches=6000 si entrenas para 3 clases
#cambiar pasos de línea al 80% y 90% de max_batches, p.e. pasos=4800,5400
if os.path.exists('./cfg/custom-yolov4-tiny-detector.cfg'): os.remove('./cfg/custom-yolov4-tiny-detector.cfg')


#personalizar el archivo de escritura de iPython para que podamos escribir variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# En[ ]:


get_ipython().run_cell_magic('writetemplate', './cfg/custom-yolov4-tiny-detector.cfg', '[net]\n# Prueba\n#lote=1\n#subdivisiones=1\n# Entrenamiento\nlote=64\nsubdivisiones=24\nancho=416\nalto=416\ncanales=3\nmomentum=0.9\ndecay=0.0005\nangle=0 \nsaturación = 1,5\exposición = 1,5\ntono=0,1\n\níndice de aprendizaje=0,00261\nquemar_en=1000\nmax_batches = {max_batches}\npolicy=steps\nsteps={steps_str}\nscales=.1,.1\n\ n[convolucional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolucional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride= 2\npad=1\nactivación=con fugas\n\n[convolucional]\nbatch_normalize=1\nfiltros=64\ntamaño=3\nzancada=1\npad=1\nactivación=con fugas\n\n[ruta]\ncapas= -1\ngroups=2\ngroup_id=1\n\n[convolucional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolucional]\nbatch_normalize =1\nfiltros=32\ntamaño=3\nzancada=1\npad=1\nactivación=fugas\n\n[ruta]\ncapas = -1,-2\n\n[convolucional]\nbatch_normalize=1\nfiltros =64\ntamaño=1\nzancada=1\npad=1\nactivación=fugas\n\n[ruta]\ncapas = -6,-1\n\n[maxpool]\ntamaño=2\nzancada=2\n \n[con volucional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n [convolucional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolucional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1 \npad=1\nactivación=fugas\n\n[ruta]\ncapas = -1,-2\n\n[convolucional]\nbatch_normalize=1\nfiltros=128\ntamaño=1\nzancada=1\npad=1 \nactivación=fugas\n\n[ruta]\ncapas = -6,-1\n\n[maxpool]\nsize=2\nstride=2\n\n[convolucional]\nbatch_normalize=1\nfilters=256\ nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n[convolucional]\nbatch_normalize=1\nfilters=128 \ntamaño=3\nzancada=1\npad=1\nactivación=fugas\n\n[convolucional]\nbatch_normalize=1\nfiltros=128\ntamaño=3\nzancadas=1\npad=1\nactivación=fugas\n\ n[ruta]\ncapas = -1,-2\n\n[convolucional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[route] \ncapas = -6,-1\n\n[maxpool]\ntamaño=2\nzancada=2 \n\n[convolucional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n############### ###################\n\n[convolucional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky \n\n[convolucional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolucional]\nsize=1\nstride=1\npad=1 \nfilters={num_filters}\nactivation=linear\n\n\n\n[yolo]\nmask = 3,4,5\nanchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319\nclasses={num_classes}\nnum=6\njitter=.3\nscale_x_y = 1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\nignore_thresh = .7\ntruth_thresh = 1\nrandom=0\nnms_kind=greedynms\ nbeta_nms=0.6\n\n[ruta]\ncapas = -4\n\n[convolucional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n [upsample]\nzancada=2\n\n[ruta]\ncapas = -1, 23\n\n[convolucional]\nbatch_normalize=1\nfilters=256\nsize=3\nzancada=1\npad=1\nactivación =con fugas\n\n[convolucional]\ntamaño=1\nzancada=1\npad=1 \nfilters={num_filters}\nactivation=linear\n\n[yolo]\nmask = 1,2,3\nanchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319\nclasses= {num_classes}\nnum=6\njitter=.3\nscale_x_y = 1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\nignore_thresh = .7\ntruth_thresh = 1\nrandom=0\nnms_kind=greedynms\nbeta_nms=0.6\ norte')


# En[ ]:


#aquí está el archivo que se acaba de escribir.
#usted puede considerar ajustar ciertas cosas

#como la cantidad de subdivisiones 64 se ejecuta más rápido, pero la GPU de Colab puede no ser lo suficientemente grande
#si la memoria GPU de Colab es demasiado pequeña, deberá ajustar las subdivisiones a 16
get_ipython().run_line_magic('cat', 'cfg/custom-yolov4-tiny-detector.cfg')


# # Entrenar detector YOLOv4 personalizado

# En[ ]:


get_ipython().system('./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map')
#Si obtiene CUDA fuera de la memoria, ajuste las subdivisiones anteriores.
#ajuste los lotes máximos hacia abajo para un entrenamiento más corto arriba


# # Inferir objetos personalizados con pesos YOLOv4 guardados

# En[ ]:


#define la función de utilidad
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  get_ipython().run_line_magic('matplotlib', 'inline')

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  # plt.rcParams['figura.tamaño de figura'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()


# En[ ]:


#compruebe si los pesos ya se han guardado
#backup alberga los últimos pesos para nuestro detector
#(el archivo yolo-obj_last.weights se guardará en build\darknet\x64\backup\ por cada 100 iteraciones)
#(el archivo yolo-obj_xxxx.weights se guardará en build\darknet\x64\backup\ por cada 1000 iteraciones)
#Después de completar el entrenamiento, obtenga el resultado yolo-obj_final.weights de la ruta build\darknet\x64\bac
get_ipython().system('ls backup')
#si está vacío, aún no ha entrenado durante el tiempo suficiente, debe entrenar durante al menos 100 iteraciones


# En[ ]:


#coco.names está codificado en algún lugar del detector
get_ipython().run_line_magic('cp', 'data/obj.names data/coco.names')


# En[ ]:


#/test tiene imágenes en las que podemos probar nuestro detector
test_images = [f for f in os.listdir('test') if f.endswith('.jpg')]
import random
img_path = "test/" + random.choice(test_images);

¡#prueba nuestro detector!
get_ipython().system('./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights {img_path} -dont-show')
imShow('/content/darknet/predictions.jpg')


# En[ ]:


while True:
  pass


# En[ ]:




