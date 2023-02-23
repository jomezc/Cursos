#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Detectron2 - BodyPose y segmentación de instancias panópticas**
#
# ---
#
# <img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">
#
# ¡Bienvenido a detectron2! Este es el tutorial oficial de colab de detectron2. Aquí, repasaremos algunos usos básicos de detectron2, incluidos los siguientes:
# * Ejecutar inferencia en imágenes o videos, con un modelo detectron2 existente
# * Entrenar un modelo detectron2 en un nuevo conjunto de datos
#
#
#

# # **Instalar detector2**

# En[ ]:


get_ipython().system('pip install pyyaml==5.1')
# Esta es la versión actual de pytorch en Colab. Descomente esto si Colab cambia su versión de pytorch
# !pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Instale detectron2 que coincida con la versión de pytorch anterior
# Consulte https://detectron2.readthedocs.io/tutorials/install.html para obtener instrucciones
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html')
# exit(0) # Después de la instalación, debe "reiniciar el tiempo de ejecución" en Colab. Esta línea también puede reiniciar el tiempo de ejecución


# En[ ]:


# comprobar la instalación de pytorch:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # instale manualmente torch 1.9 si Colab cambia su versión predeterminada


# En[ ]:


# Algunas configuraciones básicas:
# Configurar registrador detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# importar algunas bibliotecas comunes
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# importar algunas utilidades comunes de detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# # **Vamos a cargar un Modelo Pre-entrenado e Implementar BodyPose usando Detectron2**
#
# Mostramos demostraciones simples de otros tipos de modelos a continuación:

# En[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/footballer.jpg  -q')

im = cv2.imread("./footballer.jpg")
cv2_imshow(im)


# En[ ]:


# Inferencia con un modelo de detección de puntos clave
cfg = get_cfg()   # obtener una configuración nueva y fresca

cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # establecer umbral para este modelo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2_imshow(out.get_image()[:, :, ::-1])


# ## **Hagamos una segmentación de instancias usando el panóptico**

# En[ ]:


# Inferencia con un modelo de segmentación panóptico
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

cv2_imshow(out.get_image()[:, :, ::-1])


# # **Ejecutar segmentación panóptica en un video**

# En[ ]:


# Este es el video que vamos a procesar
from IPython.display import YouTubeVideo, display
video = YouTubeVideo("ll8TgCZ0plk", width=500)
display(video)


# En[ ]:


# Instale dependencias, descargue el video y recorte 5 segundos para procesar
get_ipython().system('pip install youtube-dl')
get_ipython().system('youtube-dl https://www.youtube.com/watch?v=ll8TgCZ0plk -f 22 -o video.mp4')
get_ipython().system('ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4')


# En[ ]:


# Ejecute la demostración de inferencia cuadro por cuadro en este video (toma de 3 a 4 minutos) con la herramienta "demo.py" que proporcionamos en el repositorio.
get_ipython().system('git clone https://github.com/facebookresearch/detectron2')
# Nota: actualmente está ROTO debido a que falta el códec. Consulte https://github.com/facebookresearch/detectron2/issues/2901 para obtener una solución alternativa.
get_ipython().run_line_magic('run', 'detectron2/demo/demo.py --config-file detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv    --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl')


# En[ ]:


# Descarga los resultados
from google.colab import files
files.download('video-output.mkv')


# En[ ]:




