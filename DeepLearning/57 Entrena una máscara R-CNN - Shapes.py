#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# ## **Entrena una máscara de formas CNN**
#
# ---
#
#

# ### **Configurar e instalar los paquetes necesarios**

# En 1]:


get_ipython().system('pip uninstall tensorflow --yes')
get_ipython().system('pip install tensorflow-gpu==1.13.1')
get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow
print(tensorflow.version)
get_ipython().system('pip uninstall keras-nightly --yes')
get_ipython().system('pip uninstall keras --yes')
get_ipython().system('pip install h5py==2.10.0')
get_ipython().system('pip install q keras==2.1.0')


# En 1]:


get_ipython().system('pip uninstall h5py -y')
get_ipython().system('pip install h5py==2.10.0')
get_ipython().system('pip3 install keras==2.1.0')


# En 2]:


get_ipython().run_line_magic('tensorflow_version', '1.x')


# En 3]:


get_ipython().system('git clone https://github.com/matterport/Mask_RCNN.git')


# En[4]:


get_ipython().run_line_magic('cd', 'Mask_RCNN/samples/shapes/')


# ## **Máscara R-CNN - Conjunto de datos Train on Shapes**
# Este cuaderno muestra cómo entrenar a Mask R-CNN en su propio conjunto de datos. Para simplificar las cosas, utilizamos un conjunto de datos sintéticos de formas (cuadrados, triángulos y círculos) que permite un entrenamiento rápido. Sin embargo, aún necesitaría una GPU, porque la columna vertebral de la red es una Resnet101, que sería demasiado lenta para entrenar en una CPU. En una GPU, puede comenzar a obtener buenos resultados en unos minutos y buenos resultados en menos de una hora.
#
# El código del conjunto de datos Shapes se incluye a continuación. Genera imágenes sobre la marcha, por lo que no requiere descargar ningún dato. Y puede generar imágenes de cualquier tamaño, por lo que elegimos un tamaño de imagen pequeño para entrenar más rápido.

# En[5]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Directorio raíz del proyecto
ROOT_DIR = os.path.abspath("../Training/")

#Importar Máscara RCNN
sys.path.append(ROOT_DIR)  # Para encontrar la versión local de la biblioteca
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

get_ipython().run_line_magic('matplotlib', 'inline')

# Directorio para guardar registros y modelo entrenado
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Ruta local al archivo de pesas entrenadas
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Descargue los pesos entrenados de COCO de Versiones si es necesario
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## **Configuraciones**

# En[6]:


class ShapesConfig(Config):
    """ Configuración para el entrenamiento en el conjunto de datos de formas de juguetes.Deriva de la clase Config base y anula los valores específicos
al conjunto de datos de formas de juguetes.
    """
    # Dale a la configuración un nombre reconocible
    NAME = "shapes"

    # Entrene en 1 GPU y 8 imágenes por GPU. Podemos poner múltiples imágenes en cada
    # GPU porque las imágenes son pequeñas. El tamaño del lote es 8 (GPU * imágenes/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Número de clases (incluidos los antecedentes)
    NUM_CLASSES = 1 + 3  # fondo + 3 formas

    # Use imágenes pequeñas para un entrenamiento más rápido. Establecer los límites del lado pequeño
    # el lado grande, y eso determina la forma de la imagen.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use anclas más pequeñas porque nuestra imagen y objetos son pequeños
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # lado del ancla en píxeles

    # Reduzca los ROI de entrenamiento por imagen porque las imágenes son pequeñas y tienen
    # pocos objetos. Intente permitir que el muestreo de ROI recoja un 33 % de ROI positivos.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use una época pequeña ya que los datos son simples
    STEPS_PER_EPOCH = 100

    # use pequeños pasos de validación ya que la época es pequeña
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()


# ### **Preferencias de portátiles**

# En[7]:


def get_ax(rows=1, cols=1, size=8):
    """ Devuelve una matriz de ejes de Matplotlib para usar entodas las visualizaciones en el cuaderno. Proporcionar una
punto central para controlar los tamaños de los gráficos.

Cambie el atributo de tamaño predeterminado para controlar el tamaño
de imágenes renderizadas
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## **Conjunto de datos**
#
# Crear un conjunto de datos sintético
#
# Amplíe la clase Dataset y agregue un método para cargar el conjunto de datos de formas, load_shapes(), y anule los siguientes métodos:
#
# - cargar imagen()
# - cargar_máscara()
# - imagen_referencia()

# En[8]:


class ShapesDataset(utils.Dataset):
    """ Genera el conjunto de datos sintético de formas. El conjunto de datos consta deformas (triángulos, cuadrados, círculos) colocadas al azar en una superficie en blanco.
Las imágenes se generan sobre la marcha. No se requiere acceso al archivo.
    """

    def load_shapes(self, count, height, width):
        """ Generar el número solicitado de imágenes sintéticas.count: número de imágenes a generar.
alto, ancho: el tamaño de las imágenes generadas.
        """
        # Agregar clases
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        #  Añadir imágenes
        # Genere especificaciones aleatorias de imágenes (es decir, color y
        # lista de formas, tamaños y ubicaciones). Esto es más compacto que
        # imágenes reales. Las imágenes se generan sobre la marcha en load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """ Genera una imagen a partir de las especificaciones de la ID de imagen dada.Normalmente, esta función carga la imagen desde un archivo, pero
en este caso genera la imagen sobre la marcha a partir del
especificaciones en image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """ Devuelve los datos de las formas de la imagen."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """ Genera máscaras de instancia para formas de la ID de imagen dada.        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Manejar oclusiones
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Asigne nombres de clase a ID de clase.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """ Dibuja una forma a partir de las especificaciones dadas."""
        # Obtener el centro x, y y el tamaño s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """ Genera especificaciones de una forma aleatoria que se encuentra dentrolos límites dados de alto y ancho.
Devuelve una tupla de tres valores:
* El nombre de la forma (cuadrado, círculo, ...)
* Color de la forma: una tupla de 3 valores, RGB.
* Dimensiones de la forma: una tupla de valores que definen el tamaño de la forma
y ubicación Difiere según el tipo de forma.
        """
        #  Forma
        shape = random.choice(["square", "circle", "triangle"])
        #  Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        #  Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        #  Tamaño
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """ Crea especificaciones aleatorias de una imagen con múltiples formas.Devuelve el color de fondo de la imagen y una lista de formas.
especificaciones que se pueden utilizar para dibujar la imagen.
        """
        # Elija un color de fondo aleatorio
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Genere algunas formas aleatorias y registre sus
        # cuadros delimitadores
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Aplique supresión no máxima con un umbral de 0.3 para evitar
        # formas que se cubren entre sí
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


# En[9]:


# Conjunto de datos de entrenamiento
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Conjunto de datos de validación
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# En[10]:


# Cargar y mostrar muestras aleatorias
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# ## **Crear modelo**

# En[11]:


# Crear modelo en modo entrenamiento
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# En[12]:


# ¿Con qué pesos empezar?
init_with = "coco"  # imagenet, coco o last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Cargue pesos entrenados en MS COCO, pero salte capas que
    # son diferentes debido al diferente número de clases
    # Consulte LÉAME para obtener instrucciones sobre cómo descargar los pesos de COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Cargue el último modelo que entrenó y continúe entrenando
    model.load_weights(model.find_last(), by_name=True)


# ## **Capacitación**
#
# Entrena en dos etapas:
#
# - Solo las cabezas. Aquí estamos congelando todas las capas de la columna vertebral y entrenando solo las capas inicializadas aleatoriamente (es decir, las que no usamos pesos preentrenados de MS COCO). Para entrenar solo las capas de la cabeza, pase layers='heads' a la función entrenar().
# - Afinar todas las capas. Para este ejemplo simple no es necesario, pero lo estamos incluyendo para mostrar el proceso. Simplemente pase layers="all para entrenar todas las capas.

# En[15]:


#!python3 setup.py instalar


# En[14]:


# Entrena las ramas de la cabeza
# Pasar capas="cabezas" congela todas las capas excepto la cabeza
# capas. También puede pasar una expresión regular para seleccionar
# qué capas entrenar por patrón de nombre.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


# ## **Detección**

# En[16]:


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recrea el modelo en modo inferencia
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Obtener la ruta a los pesos guardados
# Establezca una ruta específica o encuentre los últimos pesos entrenados
# model_path = os.path.join(ROOT_DIR, ".h5 nombre de archivo aquí")
model_path = model.find_last()

# Carga pesos entrenados
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# En[17]:


# Prueba en una imagen aleatoria
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# En[18]:


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


# ## **Evaluación**

# En 19]:


# Calcular mapa estilo VOC @ IoU=0.5
# Corriendo en 10 imágenes. Aumentar para una mejor precisión.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Cargue la imagen y los datos reales del terreno
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Ejecutar detección de objetos
    results = model.detect([image], verbose=0)
    r = results[0]
    # Calcular AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

