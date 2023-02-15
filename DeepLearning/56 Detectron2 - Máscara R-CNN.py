#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Detectron2 - MáscaraCNN**
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
#
# ¡Reiniciar tiempo de ejecución después!

# En 1]:


get_ipython().system('pip install pyyaml==5.1')
get_ipython().system('pip uninstall torch -y')
get_ipython().system('pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html')

# Instale detectron2 que coincida con la versión de pytorch anterior
# Consulte https://detectron2.readthedocs.io/tutorials/install.html para obtener instrucciones
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html')
# exit(0) # Después de la instalación, debe "reiniciar el tiempo de ejecución" en Colab. Esta línea también puede reiniciar el tiempo de ejecución


# En 1]:


# comprobar la instalación de pytorch:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # instale manualmente torch 1.9 si Colab cambia su versión predeterminada


# En 2]:


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


# # **Ejecute un modelo de detectron2 previamente entrenado**

# Primero descargamos una imagen del conjunto de datos COCO:

# En 3]:


get_ipython().system('wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg')
im = cv2.imread("./input.jpg")
cv2_imshow(im)


# Luego, creamos una configuración de detectron2 y un `DefaultPredictor` de detectron2 para ejecutar la inferencia en esta imagen.

# En[4]:


cfg = get_cfg()
# agregue la configuración específica del proyecto (por ejemplo, TensorMask) aquí si no está ejecutando un modelo en la biblioteca central de detectron2
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # establecer umbral para este modelo
# Encuentre un modelo del zoológico de modelos de detectron2. También puede usar la URL https://dl.fbaipublicfiles...
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# En[5]:


# mira las salidas. Consulte https://detectron2.readthedocs.io/tutorials/models.html#model-output-format para conocer las especificaciones
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


# En[6]:


# Podemos usar `Visualizer` para dibujar las predicciones en la imagen.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])


# # **Entrenar en un conjunto de datos personalizado**

# En esta sección, mostramos cómo entrenar un modelo detectron2 existente en un conjunto de datos personalizado en un nuevo formato.
#
# Usamos [el conjunto de datos de segmentación de globo](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
# que solo tiene una clase: globo.
# Entrenaremos un modelo de segmentación de globos a partir de un modelo existente previamente entrenado en el conjunto de datos COCO, disponible en el zoológico modelo de detectron2.
#
# Tenga en cuenta que el conjunto de datos COCO no tiene la categoría "globo". Podremos reconocer esta nueva clase en unos minutos.
#
# ## Preparar el conjunto de datos

# En[7]:


# descargar, descomprimir los datos
get_ipython().system('wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip')
get_ipython().system('unzip balloon_dataset.zip > /dev/null')


# Registre el conjunto de datos del globo en detectron2, siguiendo el [tutorial del conjunto de datos personalizado de detectron2](https://detectron2.readthedocs.io/tutorials/datasets.html).
# Aquí, el conjunto de datos está en su formato personalizado, por lo tanto, escribimos una función para analizarlo y prepararlo en el formato estándar de detectron2. El usuario debe escribir dicha función cuando use un conjunto de datos en formato personalizado. Vea el tutorial para más detalles.
#

# En[8]:


# si su conjunto de datos está en formato COCO, esta celda se puede reemplazar por las siguientes tres líneas:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")


# Para verificar que la carga de datos sea correcta, visualicemos las anotaciones de muestras seleccionadas aleatoriamente en el conjunto de entrenamiento:
#
#

# En[9]:


dataset_dicts = get_balloon_dicts("balloon/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])


# ## **Tren**
#
# Ahora, ajustemos un modelo R-CNN de máscara R50-FPN preentrenado por COCO en el conjunto de datos del globo. Se necesitan ~2 minutos para entrenar 300 iteraciones en una GPU P100.
#

# En[ ]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Deje que el entrenamiento se inicialice desde el zoológico modelo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # elige un buen LR
cfg.SOLVER.MAX_ITER = 300    # 300 iteraciones parece lo suficientemente bueno para este conjunto de datos de juguete; necesitará entrenar más tiempo para un conjunto de datos práctico
cfg.SOLVER.STEPS = []        # no decae la tasa de aprendizaje
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # más rápido y lo suficientemente bueno para este conjunto de datos de juguete (predeterminado: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # solo tiene una clase (globo). (ver https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTA: esta configuración significa el número de clases, pero algunos tutoriales populares no oficiales usan incorrectamente num_classes+1 aquí.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# En[ ]:


# Mira las curvas de entrenamiento en tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir output')


# ## **Inferencia y evaluación usando el modelo entrenado**
# Ahora, ejecutemos la inferencia con el modelo entrenado en el conjunto de datos de validación del globo. Primero, creemos un predictor usando el modelo que acabamos de entrenar:
#
#

# En[ ]:


# La inferencia debe usar la configuración con parámetros que se usan en el entrenamiento
# cfg ahora ya contiene todo lo que hemos configurado anteriormente. Lo cambiamos un poco por inferencia:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # ruta al modelo que acabamos de entrenar
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # establecer un umbral de prueba personalizado
predictor = DefaultPredictor(cfg)


# Luego, seleccionamos aleatoriamente varias muestras para visualizar los resultados de la predicción.

# En[ ]:


from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # El formato está documentado en https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # eliminar los colores de los píxeles no segmentados. Esta opción solo está disponible para modelos de segmentación
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


# También podemos evaluar su rendimiento utilizando la métrica AP implementada en la API de COCO.
# Esto da un AP de ~70. ¡No está mal!

# En[ ]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("balloon_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "balloon_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# otra forma equivalente de evaluar el modelo es usar `trainer.test`


# # Otros tipos de modelos incorporados
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


# # Ejecutar segmentación panóptica en un video

# En[10]:


# Este es el video que vamos a procesar
from IPython.display import YouTubeVideo, display
video = YouTubeVideo("ll8TgCZ0plk", width=500)
display(video)


# En[11]:


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

