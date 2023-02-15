#!/usr/bin/env python
# codificación: utf-8

#
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Entrenamiento YOLOv4 Darknet**
#
# ## Introducción
#
#
# En este cuaderno, implementamos [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) para la capacitación en nuestro conjunto de datos de capturas de pantalla del sitio web.
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
# #### ![Marca de trabajo de Roboflow](https://github.com/rajeevratan84/ModernComputerVision/raw/main/website.png)
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


# ## PASO 1. Instalar cuDNN según la versión actual de CUDA
# Colab agregó cuDNN como una instalación inherente, por lo que no tiene que hacer nada, una gran victoria
#
#
#

# # Paso 2: Instalación de Darknet para YOLOv4 en Colab
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


get_ipython().run_line_magic('cd', '/content/darknet/')
get_ipython().run_line_magic('rm', 'Makefile')


# En[ ]:


#colab ocasionalmente cambia las dependencias, en el momento de la autoría, este Makefile funciona para construir Darknet en Colab

get_ipython().run_line_magic('%writefile', 'Makefile')
GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0

# establecer GPU=1 y CUDNN=1 para acelerar en GPU
# establecer CUDNN_HALF=1 para aumentar la velocidad 3 veces (precisión mixta en núcleos tensoriales) GPU: Volta, Xavier, Turing y superior
# establezca AVX=1 y OPENMP=1 para acelerar en la CPU (si ocurre un error, establezca AVX=0)
# establezca ZED_CAMERA=1 para habilitar ZED SDK 3.0 y superior
# establezca ZED_CAMERA_v2_8=1 para habilitar ZED SDK 2.X

USE_CPP=0
DEBUG=0

ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
	    -gencode arch=compute_61,code=[sm_61,compute_61] \
      -gencode arch=compute_37,code=sm_37

ARCH= -gencode arch=compute_60,code=sm_60

OS := $(shell uname)

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

ifeq ($(LIBSO), 1)
LIBNAMESO=libdarknet.so
APPNAMESO=uselib
endif

ifeq ($(USE_CPP), 1)
CC=g++
else
CC=gcc
endif

CPP=g++ -std=c++11
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -I3rdparty/stb/include
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC

ifeq ($(DEBUG), 1)
#OPTS= -O0 -g
#OPTS= -Y -g
COMMON+= -DDEBUG
CFLAGS+= -DDEBUG
else
ifeq ($(AVX), 1)
CFLAGS+= -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a
endif
endif

CFLAGS+=$(OPTS)

ifneq (,$(findstring MSYS_NT,$(OS)))
LDFLAGS+=-lws2_32
endif

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv`
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
LDFLAGS+= -lgomp
endif

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
ifeq ($(OS),Darwin) # MAC
LDFLAGS+= -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand
else
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
ifeq ($(OS),Darwin) # MAC
CFLAGS+= -DCUDNN -I/usr/local/cuda/include
LDFLAGS+= -L/usr/local/cuda/lib -lcudnn
else
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
endif
endif

ifeq ($(CUDNN_HALF), 1)
COMMON+= -DCUDNN_HALF
CFLAGS+= -DCUDNN_HALF
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]
endif

ifeq ($(ZED_CAMERA), 1)
CFLAGS+= -DZED_STEREO -I/usr/local/zed/include
ifeq ($(ZED_CAMERA_v2_8), 1)
LDFLAGS+= -L/usr/local/zed/lib -lsl_core -lsl_input -lsl_zed
#-lstdc++ -D_GLIBCXX_USE_CXX11_ABI=0
else
LDFLAGS+= -L/usr/local/zed/lib -lsl_zed
#-lstdc++ -D_GLIBCXX_USE_CXX11_ABI=0
endif
endif

OBJ=image_opencv.o http_stream.o gemm.o utils.o dark_cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o reorg_old_layer.o super.o voxel.o tree.o yolo_layer.o gaussian_yolo_layer.o upsample_layer.o lstm_layer.o conv_lstm_layer.o scale_channels_layer.o sam_layer.o
ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: $(OBJDIR) backup results setchmod $(EXEC) $(LIBNAMESO) $(APPNAMESO)

ifeq ($(LIBSO), 1)
CFLAGS+= -fPIC

$(LIBNAMESO): $(OBJDIR) $(OBJS) include/yolo_v2_class.hpp src/yolo_v2_class.cpp
	$(CPP) -shared -std=c++11 -fvisibility=hidden -DLIB_EXPORTS $(COMMON) $(CFLAGS) $(OBJS) src/yolo_v2_class.cpp -o $@ $(LDFLAGS)

$(APPNAMESO): $(LIBNAMESO) include/yolo_v2_class.hpp src/yolo_console_dll.cpp
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -o $@ src/yolo_console_dll.cpp $(LDFLAGS) -L ./ -l:$(LIBNAMESO)
endif

$(EXEC): $(OBJS)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)
backup:
	mkdir -p backup
results:
	mkdir -p results
setchmod:
	chmod +x *.sh

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) $(LIBNAMESO) $(APPNAMESO)


# En[ ]:


#instalar entorno desde el Makefile
#note si está en Colab Pro, esto funciona en una GPU P100
#si está en Colab gratis, es posible que deba cambiar el Makefile para la GPU K80
#esto se aplica a cualquier GPU, debe cambiar el Makefile para informar a darknet en qué GPU se está ejecutando.
#Tenga en cuenta que el Makefile anterior debería funcionar para usted, si necesita modificar, intente lo siguiente
get_ipython().run_line_magic('cd', '/content/darknet/')
get_ipython().system("sed -i 's/OPENCV=0/OPENCV=1/g' Makefile")
get_ipython().system("sed -i 's/GPU=0/GPU=1/g' Makefile")
get_ipython().system("sed -i 's/CUDNN=0/CUDNN=1/g' Makefile")
get_ipython().system('sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= ${ARCH_VALUE}/g" Makefile')
get_ipython().system('make')


# En[ ]:


#descargue los pesos ConvNet de yolov4 recién lanzados
get_ipython().run_line_magic('cd', '/content/darknet')
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137')


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


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Website+Screenshots.v1-raw.darknet.zip')
get_ipython().system("unzip -q 'Website Screenshots.v1-raw.darknet.zip'")


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
  for img in [f for f in os.listdir('test') if f.endswith('jpg')]:
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
print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

#Instrucciones del repositorio de darknet
#cambie la línea max_batches a (clases*2000 pero no menos del número de imágenes de entrenamiento, y no menos de 6000), ej. max_batches=6000 si entrenas para 3 clases
#cambiar pasos de línea al 80% y 90% de max_batches, p.e. pasos=4800,5400
if os.path.exists('./cfg/custom-yolov4-detector.cfg'): os.remove('./cfg/custom-yolov4-detector.cfg')


with open('./cfg/custom-yolov4-detector.cfg', 'a') as f:
  f.write('[net]' + '\n')
  f.write('batch=64' + '\n')
  # ####las subdivisiones más pequeñas ayudan a que la GPU funcione más rápido. 12 es óptimo, pero es posible que deba cambiar a 24,36,64####
  f.write('subdivisions=24' + '\n')
  f.write('width=416' + '\n')
  f.write('height=416' + '\n')
  f.write('channels=3' + '\n')
  f.write('momentum=0.949' + '\n')
  f.write('decay=0.0005' + '\n')
  f.write('angle=0' + '\n')
  f.write('saturation = 1.5' + '\n')
  f.write('exposure = 1.5' + '\n')
  f.write('hue = .1' + '\n')
  f.write('\n')
  f.write('learning_rate=0.001' + '\n')
  f.write('burn_in=1000' + '\n')
  # #####puede ajustar hacia arriba y hacia abajo para cambiar el tiempo de entrenamiento#####
  # #Darknet hace iteraciones con lotes, no con épocas####
  # max_lotes = num_clases*2000
  max_batches = 2000
  f.write('max_batches=' + str(max_batches) + '\n')
  f.write('policy=steps' + '\n')
  steps1 = .8 * max_batches
  steps2 = .9 * max_batches
  f.write('steps='+str(steps1)+','+str(steps2) + '\n')

#Instrucciones del repositorio de darknet
#cambie las clases de línea = 80 a su número de objetos en cada una de las 3 capas [yolo]:
#cambiar [filters=255] a filtros=(clases + 5)x3 en las 3 [convolucionales] antes de cada capa [yolo], ten en cuenta que solo tiene que ser la última [convolucional] antes de cada una de las [yolo] capas.

  with open('cfg/yolov4-custom2.cfg', 'r') as f2:
    content = f2.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 0,1,2' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')

  with open('cfg/yolov4-custom3.cfg', 'r') as f3:
    content = f3.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 3,4,5' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')

  with open('cfg/yolov4-custom4.cfg', 'r') as f4:
    content = f4.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 6,7,8' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')
    
  with open('cfg/yolov4-custom5.cfg', 'r') as f5:
    content = f5.readlines()
    for line in content:
      f.write(line)

print("file is written!")


# En[ ]:


#aquí está el archivo que se acaba de escribir.
#usted puede considerar ajustar ciertas cosas

#como la cantidad de subdivisiones 64 se ejecuta más rápido, pero la GPU de Colab puede no ser lo suficientemente grande
#si la memoria GPU de Colab es demasiado pequeña, deberá ajustar las subdivisiones a 16
get_ipython().run_line_magic('cat', 'cfg/custom-yolov4-detector.cfg')


# # Entrenar detector YOLOv4 personalizado

# En[ ]:


get_ipython().system('./darknet detector train data/obj.data cfg/custom-yolov4-detector.cfg yolov4.conv.137 -dont_show -map')
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
get_ipython().system('./darknet detect cfg/custom-yolov4-detector.cfg backup/custom-yolov4-detector_final.weights {img_path} -dont-show')
imShow('/content/darknet/predictions.jpg')


# En[ ]:




