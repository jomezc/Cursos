#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **DeepSORT - Seguimiento usando YOLOv5**
#
# ---
#
# En esta lección, aprenderá cómo integrar DeepSORT con cualquier modelo YOLOv5.
# 1. Descarga y explora nuestros datos
# 2. Cargue nuestro modelo VGG16 preentrenado
# 3. Extraiga nuestras características usando VGG16
# 4. Entrena un Clasificador LR usando esas características
# 5. Prueba algunas inferencias
#
# ### **Cambiar a GPU para aumentar el rendimiento.**
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)

# ## **Configuración**

# En[ ]:


# enlace de repositorio de respaldo: https://github.com/rajeevratan84/Yolov5_DeepSort_Pytorch.git
get_ipython().system('git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git  # clonar repositorio')
get_ipython().run_line_magic('cd', 'Yolov5_DeepSort_Pytorch')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # instalar dependencias')
get_ipython().system('pip install youtube-dl')

import torch
from IPython.display import Image, clear_output  # para mostrar imágenes

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


# ## **Descargue nuestro video de prueba y modelos**
#
# El código para descargar de youtube y cortar segmentos del video se comenta a continuación. Simplemente resalte el código y presione CTRL+L para descomentar el bloque.
#

# En[ ]:


# YOUTUBE_ID = 'uCj6glLYW5g'
# #!rm -rf youtube.mp4 https://www.youtube.com/watch?v=uCj6glLYW5g
# # descarga el youtube con la ID dada
#! youtube-dl -f 22 --salida "youtube.%(ext)s" https://www.youtube.com/watch?v=$YOUTUBE_ID

# # cortar los primeros 15 segundos
#! ffmpeg -y -loglevel info -i youtube.mp4 -t 30 ped_track.mp4
# !y | ffmpeg -ss 00:00:00 -i youtube.mp4 -t 00:00:15 -c copiar youtube_out.avi


# En[ ]:


get_ipython().run_line_magic('mkdir', 'yolov5/weights/')

# Obtenga el modelo yolov5 entrenado en el conjunto de datos coco128
get_ipython().system('wget -nc https://github.com/rajeevratan84/ModernComputerVision/raw/main/yolov5s.pt -O /content/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5s.pt')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/youtube_out.avi')

# entrenar el modelo yolov5m en el conjunto de datos multitud-humano
#!wget -nc https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v.2.0/crowdhuman_yolov5m.pt -O /content/Yolov5_DeepSort_Pytorch/yolov5/weights/crowdhuman_yolov5m.pt
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/crowdhuman_yolov5m.pt')

# obtener el video de prueba del repositorio
#!wget -nc https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v.2.0/test.avi
# extraer 3 segundos de fotogramas de video
#!y | ffmpeg -ss 00:00:00 -i prueba.avi -t 00:00:02 -c copia.avi


# ## **Ejecutar inferencia en video**
#
# Por lo tanto, elegimos guardarlo en un archivo en este cuaderno. Localmente puede usar el indicador ``--show-vid`` para visualizar el seguimiento en tiempo real

# En[ ]:


get_ipython().system('python track.py --yolo_model /content/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5s.pt --source youtube_out.avi --save-vid')


# ## **Mostrar resultados**
#
# Convierta avi a mp4 y luego podemos reproducir video dentro de colab. Esto toma ~25 segundos.

# En[ ]:


get_ipython().system('ffmpeg -i /content/Yolov5_DeepSort_Pytorch/runs/track/exp/youtube_out.avi output.mp4 -y')


# Obtener el contenido del archivo en data_url

# En[ ]:


from IPython.display import HTML
from base64 import b64encode
mp4 = open('output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# Mostrarlo con HTML

# En[ ]:


HTML("""<controles de vídeo>
<fuente src="%s" type="video/mp4">
</vídeo>
""" % URL_datos)


# En[ ]:




