#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# # **Máscara CNN TensorFlow Demostración de Matterport**
#
#

# En 1]:


get_ipython().system('pip uninstall h5py -y')
get_ipython().system('pip install h5py==2.10.0')


# En 2]:


get_ipython().run_line_magic('tensorflow_version', '1.x')


# En 3]:


get_ipython().system('git clone https://github.com/matterport/Mask_RCNN.git')


# En[4]:


get_ipython().run_line_magic('cd', 'Mask_RCNN/samples')


# En[5]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Directorio raíz del proyecto
ROOT_DIR = os.path.abspath("../Cursos/Modern Computer vision/")

#Importar Máscara RCNN
sys.path.append(ROOT_DIR)  # Para encontrar la versión local de la biblioteca
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Importar configuración de COCO
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # Para encontrar la versión local
import coco

get_ipython().run_line_magic('matplotlib', 'inline')

# Directorio para guardar registros y modelo entrenado
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Ruta local al archivo de pesas entrenadas
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Descargue los pesos entrenados de COCO de Versiones si es necesario
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directorio de imágenes para ejecutar la detección
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# En[6]:


class InferenceConfig(coco.CocoConfig):
    # Establezca el tamaño del lote en 1, ya que ejecutaremos la inferencia en
    # una imagen a la vez. Tamaño del lote = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## **Crear modelo y cargar pesos entrenados**

# En[7]:


# Crear objeto modelo en modo de inferencia.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Cargue pesos entrenados en MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# En[8]:


# Nombres de clase COCO
# El índice de la clase en la lista es su ID. Por ejemplo, para obtener la identificación de
# la clase de oso de peluche, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# En[9]:


# Cargue una imagen aleatoria de la carpeta de imágenes
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Ejecutar detección
results = model.detect([image], verbose=1)

# Visualizar resultados
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

