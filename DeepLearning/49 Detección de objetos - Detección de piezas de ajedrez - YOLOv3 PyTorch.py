#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
#
# NOTA: para obtener la versión más actualizada de este cuaderno, copie de
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ntAL_zI68xfvZ4uCSAF6XT27g0U4mZbW#scrollTo=VHS_o3KGIyXm )
#
#
#
#
# ## **Entrenamiento de detección de objetos YOLOv3 en un conjunto de datos personalizado**
#
# ¡Edición PyTorch! Gracias a [Ultralytics](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) por simplificar esto.
#
# ### **Descripción general**
#
# Este cuaderno explica cómo entrenar un modelo de detección de objetos YOLOv3 en su propio conjunto de datos usando Roboflow y Colab.
#
# En este ejemplo específico, entrenaremos un modelo de detección de objetos para reconocer piezas de ajedrez en imágenes. **Para adaptar este ejemplo a su propio conjunto de datos, solo necesita cambiar una línea de código en este cuaderno.**
#
# ![Ejemplo de ajedrez](https://i.imgur.com/nkjobw1.png)
#
# ### **Nuestros datos**
#
# Nuestro conjunto de datos de 289 imágenes de ajedrez (¡y 2894 anotaciones!) está alojado públicamente en Roboflow [aquí] (https://public.roboflow.ai/object-detection/chess-full).
#
# ### **Nuestro Modelo**
#
# Estaremos entrenando un modelo YOLOv3 (Solo miras una vez). Este modelo específico es un aprendiz de una sola toma, lo que significa que cada imagen solo pasa a través de la red una vez para hacer una predicción, lo que permite que la arquitectura tenga un gran rendimiento, viendo hasta 60 cuadros por segundo en la predicción contra las transmisiones de video.
#
# El repositorio de GitHub que contiene la mayoría del código que usaremos está disponible [aquí](https://github.com/roboflow-ai/yolov3).
#
# ### **Capacitación**
#
# Google Colab proporciona recursos de GPU gratuitos. Haz clic en "Tiempo de ejecución" → "Cambiar tipo de tiempo de ejecución" → menú desplegable Acelerador de hardware a "GPU".
#
# Colab tiene limitaciones de memoria, y los cuadernos deben estar abiertos en su navegador para ejecutarse. Las sesiones se borran automáticamente después de 12 horas.
#
# ### **Inferencia**
#
# Aprovecharemos el script `detect.py --weights weights/last.pt` para producir predicciones. Los argumentos se especifican a continuación.
#
# ### **Acerca de**
#
# [Roboflow](https://roboflow.ai) hace que la gestión, el preprocesamiento, el aumento y el control de versiones de conjuntos de datos para la visión artificial sean fluidos.
#
# Los desarrolladores reducen el 50 % de su código repetitivo cuando usan el flujo de trabajo de Roboflow, ahorran tiempo de capacitación y aumentan la reproducibilidad del modelo.
#
# #### ![Marca de trabajo de Roboflow](https://i.imgur.com/WHFqYSJ.png)
#
#
#
#
#
#
#

# En 1]:


import os
import torch
from IPython.display import Image, clear_output 
print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# En 2]:


get_ipython().system('git clone https://github.com/roboflow-ai/yolov3  # clon')


# ## Obtener datos de Roboflow
#
# Crea una exportación desde Roboflow. **Seleccione "YOLO Darknet" como tipo de exportación.**
#
# Nuestras etiquetas tendrán el formato de la arquitectura de nuestro modelo.

# En[7]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Chess+Pieces.v24-416x416_aug.darknet.zip')
get_ipython().system("unzip -q 'Chess+Pieces.v24-416x416_aug.darknet.zip'")


# ## Organizar datos y etiquetas para la implementación de Ultralytics YOLOv3
#
# La implementación de YOLOv3 de Ultalytics exige [una gestión de archivos específica](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) donde nuestras imágenes están en una carpeta llamada `images` y las etiquetas correspondientes en un carpeta llamada `etiquetas`. Los nombres de la imagen y la etiqueta deben coincidir de forma idéntica. Afortunadamente, nuestros archivos tienen el nombre apropiado de Roboflow.
#
# Necesitamos reorganizar ligeramente la estructura de carpetas.

# En[4]:


get_ipython().run_line_magic('cd', 'train')


# En[5]:


get_ipython().run_line_magic('ls', '')


# En[ ]:


get_ipython().run_line_magic('mkdir', 'labels')
get_ipython().run_line_magic('mkdir', 'images')


# En[ ]:


get_ipython().run_line_magic('mv', '*.jpg ./images/')
get_ipython().run_line_magic('mv', '*.txt ./labels/')


# En[ ]:


get_ipython().run_line_magic('cd', 'images')


# En[ ]:


# crear un archivo de texto específico de Ultralytics de imágenes de entrenamiento
file = open("train_images_roboflow.txt", "w") 
for root, dirs, files in os.walk(""):
    for filename in files:
      # imprimir("../tren/imágenes/" + nombre de archivo)
      if filename == "train_images_roboflow.txt":
        pass
      else:
        file.write("../train/images/" + filename + "\n")
file.close()


# En[ ]:


get_ipython().run_line_magic('cat', 'train_images_roboflow.txt')


# En[ ]:


get_ipython().run_line_magic('cd', '../../valid')


# En[ ]:


get_ipython().run_line_magic('mkdir', 'labels')
get_ipython().run_line_magic('mkdir', 'images')


# En[ ]:


get_ipython().run_line_magic('mv', '*.jpg ./images/')
get_ipython().run_line_magic('mv', '*.txt ./labels/')


# En[ ]:


get_ipython().run_line_magic('cd', 'images')


# En[ ]:


# crear un archivo de texto específico de Ultralytics de imágenes de validación
file = open("valid_images_roboflow.txt", "w") 
for root, dirs, files in os.walk(""):
    for filename in files:
      # imprimir("../tren/imágenes/" + nombre de archivo)
      if filename == "valid_images_roboflow.txt":
        pass
      else:
        file.write("../valid/images/" + filename + "\n")
file.close()


# En[ ]:


get_ipython().run_line_magic('cat', 'valid_images_roboflow.txt')


# ## Configurar la configuración del modelo
#
# Deberíamos configurar nuestro modelo para entrenamiento.
#
# Esto requiere editar el archivo `roboflow.data`, que le dice a nuestro modelo dónde encontrar nuestros datos, nuestra cantidad de clases y los nombres de las etiquetas de nuestras clases.
#
# Nuestras rutas para nuestras etiquetas e imágenes son correctas.
#
# Pero necesitamos actualizar los nombres de nuestras clases. Eso se maneja a continuación..
#
#
#

# En[ ]:


get_ipython().run_line_magic('cd', '../../yolov3/data')


# En[ ]:


# mostrar etiquetas de clase importadas de Roboflow
get_ipython().run_line_magic('cat', '../../train/_darknet.labels')


# En[ ]:


# convertir .labels a .names para la especificación de Ultralytics
get_ipython().run_line_magic('cat', '../../train/_darknet.labels > ../../train/roboflow_data.names')


# En[ ]:


def get_num_classes(labels_file_path):
    classes = 0
    with open(labels_file_path, 'r') as f:
      for line in f:
        classes += 1
    return classes


# En[ ]:


# actualizar el archivo roboflow.data con el número correcto de clases
import re

num_classes = get_num_classes("../../train/_darknet.labels")
with open("roboflow.data") as f:
    s = f.read()
with open("roboflow.data", 'w') as f:
    
    # Establecer el número de clases num_classes.
    s = re.sub('classes=[0-9]+',
               'classes={}'.format(num_classes), s)
    f.write(s)


# En[ ]:


# mostrar el número actualizado de clases
get_ipython().run_line_magic('cat', 'roboflow.data')


# ## Entrenando a nuestro modelo
#
# Una vez que tengamos nuestros datos preparados, entrenaremos nuestro modelo usando el script de entrenamiento.
#
# Por defecto, este script entrena para 300 épocas.

# En[ ]:


get_ipython().run_line_magic('cd', '../')


# En[ ]:


get_ipython().system('python3 train.py --data data/roboflow.data --epochs 300')


# En[ ]:





# En[ ]:





# ## Muestra el rendimiento del entrenamiento
#
# Usaremos una secuencia de comandos predeterminada para mostrar los resultados de la imagen. **Por ejemplo:**
#
# ![resultados de ejemplo](https://user-images.githubusercontent.com/26833433/63258271-fe9d5300-c27b-11e9-9a15-95038daf4438.png)

# En[ ]:


from utils import utils; utils.plot_results()


# ## Realizar inferencias y mostrar resultados
#
#

# ### Realizar inferencias
#
# El siguiente script tiene algunos argumentos clave que estamos usando:
# - **Pesas**: estamos especificando que las pesas a usar para nuestro modelo deben ser las que usamos más recientemente en el entrenamiento
# - **Fuente**: estamos especificando las imágenes de origen que queremos usar para nuestras predicciones
# - **Nombres**: estamos definiendo los nombres que queremos usar. Aquí, hacemos referencia a `roboflow_data.names`, que creamos a partir de nuestro texto en cursiva `_darknet.labels` de Roboflow arriba.

# En[ ]:


get_ipython().system('python3 detect.py --weights weights/last.pt --source=../test --names=../train/roboflow_data.names')


# En[ ]:





# En[ ]:





# ### Mostrando nuestros resultados
#
# Ultralytics genera predicciones que incluyen etiquetas y cuadros delimitadores "impresos" directamente en la parte superior de nuestras imágenes. Se guardan en nuestro directorio `output` dentro del repositorio YOLOv3 que clonamos arriba.

# En[ ]:


# importar bibliotecas para mostrar
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Image
from glob import glob
import random
import PIL


# En[ ]:


# traza solo una predicción de imagen aleatoria
filename = random.choice(os.listdir('./output'))
print(filename)
Image('./output/' + filename)


# En[ ]:


# tomar todas las imágenes de nuestro directorio de salida
images = [ PIL.Image.open(f) for f in glob('./output/*') ]


# En[ ]:


# convertir imágenes a numPy
def img2array(im):
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    return np.fromstring(im.tobytes(), dtype='uint8').reshape((im.size[1], im.size[0], 3))


# En[ ]:


# crear una matriz de imágenes numPy
np_images = [ img2array(im) for im in images ]


# En[ ]:


# trace TODOS los resultados en el directorio de prueba (NOTA: ajuste figsize como quiera)
for img in np_images:
    plt.figure(figsize=(8, 6))
    plt.imshow(img)


# En[ ]:





# ## Salvemos nuestros pesos
#
# Podemos guardar los pesos de nuestro modelo para usarlos para inferencias en el futuro, o retomar el entrenamiento donde lo dejamos.
#
# Primero podemos guardarlos localmente. Conectaremos nuestro Google Drive y los guardaremos allí.
#

# En[ ]:


# guardar localmente
from google.colab import files
files.download('./weights/last.pt')


# En[ ]:





# En[ ]:


# conectar Google Drive
from google.colab import drive
drive.mount('/content/gdrive')


# En[ ]:


get_ipython().run_line_magic('pwd', '')


# En[ ]:


# crear una copia del archivo de pesos con una fecha y hora
# y mueve ese archivo a tu propia unidad
get_ipython().run_line_magic('cp', './weights/last.pt ./weights/last_copy.pt')
get_ipython().run_line_magic('mv', './weights/last_copy.pt /content/gdrive/My\\ Drive')


# En[ ]:




