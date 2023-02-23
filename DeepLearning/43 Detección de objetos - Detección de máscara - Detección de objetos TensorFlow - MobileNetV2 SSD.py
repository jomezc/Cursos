#!/usr/bin/env python
# codificación: utf-8

#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
#
# ## **Entrenamiento de detección de objetos MobileNetSSD en un conjunto de datos de máscara**
#
# ¡Nota! Para obtener la versión más actualizada de este cuaderno, asegúrese de copiar de:
#
# 💡 Recomendación: [Abra esta publicación de blog](https://blog.roboflow.ai/training-a-tensorflow-object-detection-model-with-a-custom-dataset/) para continuar.
#
# ### **Descripción general**
#
# Este cuaderno explica cómo entrenar un modelo de detección de objetos de MobileNet mediante la API de detección de objetos de TensorFlow 1.5.
#
# En este ejemplo específico, entrenaremos un modelo de detección de objetos para reconocer tipos de células: glóbulos blancos, glóbulos rojos y plaquetas. **Para adaptar este ejemplo para entrenar en su propio conjunto de datos, solo necesita cambiar dos líneas de código en este cuaderno.**
#
# Todo en este cuaderno también está alojado en este [repositorio de GitHub] (https://github.com/josephofiowa/tensorflow-object-detection).
#
# ![Salida del detector de máscara](https://github.com/rajeevratan84/ModernComputerVision/raw/main/mask.png)
#
# **Crédito para [DLology](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) y [Tony607](https:/ /github.com/Tony607)**, quien escribió el primer cuaderno en el que se basa gran parte de este ejemplo.
#
# ### **Nuestros datos**
#
# Usaremos un conjunto de datos de uso de máscaras de código abierto que contiene 149 imágenes y está alojado públicamente en Roboflow [aquí] (https://public.roboflow.com/object-detection/mask-wearing).
#
# Al adaptar este ejemplo a sus propios datos, cree dos conjuntos de datos en Roboflow: `train` y `test`. Use Roboflow para generar TFRecords para cada uno, reemplace sus URL en este cuaderno y podrá entrenar en su propio conjunto de datos personalizado.
#
# ### **Nuestro Modelo**
#
# Estaremos entrenando un MobileNetSSDv2 (detector de disparo único). Este modelo específico es un aprendiz de un solo paso, lo que significa que cada imagen solo pasa a través de la red una vez para hacer una predicción, lo que permite que la arquitectura tenga un gran rendimiento para el hardware móvil.
#
# La arquitectura del modelo es una de las muchas disponibles a través del [zoológico modelo] de TensorFlow (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).
#
# Como nota, este portátil asume TensorFlow 1.5, ya que TensorFlow 2.0 aún no es totalmente compatible con la API de detección de objetos.
#
# ### **Capacitación**
#
# Google Colab proporciona recursos de GPU gratuitos. Haz clic en "Tiempo de ejecución" → "Cambiar tipo de tiempo de ejecución" → menú desplegable Acelerador de hardware a "GPU".
#
# Colab tiene limitaciones de memoria, y los cuadernos deben estar abiertos en su navegador para ejecutarse. Las sesiones se borran automáticamente después de 12 horas.
#
# ### **Inferencia**
#
# Ejecutaremos la inferencia directamente en este cuaderno y en tres imágenes de prueba contenidas en la carpeta "prueba" de nuestro repositorio de GitHub.
#
# Cuando se adapte a su propio conjunto de datos, deberá agregar imágenes de prueba a la carpeta `test` ubicada en `tensorflow-object-detection/test`.
#
# ### **Acerca de**
#
# [Roboflow](https://roboflow.ai) hace que la gestión, el preprocesamiento, el aumento y el control de versiones de conjuntos de datos para la visión artificial sean fluidos.
#
# Los desarrolladores reducen el 50 % de su código repetitivo cuando usan el flujo de trabajo de Roboflow, automatizan el control de calidad del etiquetado, ahorran tiempo de capacitación y aumentan la reproducibilidad del modelo.
#
# #### ![Marca de trabajo de Roboflow](https://i.imgur.com/WHFqYSJ.png)
#
#
#
#
#
#

# ## Configuraciones e hiperparámetros
#
# Admite una variedad de modelos, puede encontrar más modelos preentrenados en [Zoológico de modelos de detección de Tensorflow: modelos entrenados por COCO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo. md#coco-trained-models), así como sus archivos de configuración de pipline en [object_detection/samples/configs/](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) .

# En[ ]:


# Si bifurcó el repositorio, puede reemplazar el enlace.
repo_url = 'https://github.com/roboflow-ai/tensorflow-object-detection-faster-rcnn'

# Número de pasos de entrenamiento: 1000 entrenarán muy rápido, pero más pasos aumentarán la precisión.
num_steps = 30000  #200000 para mejorar

# Número de pasos de evaluación.
num_eval_steps = 50

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    },    
}

# Elija el modelo que desea usar
# Selecciona un modelo en `MODELS_CONFIG`.
selected_model = 'ssd_mobilenet_v2'

# Nombre del modelo de detección de objetos a utilizar.
MODEL = MODELS_CONFIG[selected_model]['model_name']

# Nombre del archivo pipline en la API de detección de objetos de tensorflow.
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

# El tamaño del lote de entrenamiento cabe en la memoria GPU Tesla K80 de Colabe para el modelo seleccionado.
batch_size = MODELS_CONFIG[selected_model]['batch_size']


# En[ ]:


# use TF 1.x para las API de detección de objetos, ya que aún no están migrados a TF 2.0
get_ipython().run_line_magic('tensorflow_version', '1.x')


# ## Clone el repositorio `tensorflow-object-detection` o su bifurcación.

# En[ ]:


import os

get_ipython().run_line_magic('cd', '/content')

repo_dir_path = os.path.abspath(os.path.join('', os.path.basename(repo_url)))

get_ipython().system('git clone {repo_url}')
get_ipython().run_line_magic('cd', '{repo_dir_path}')
get_ipython().system('git pull')


# ## Instalar los paquetes necesarios

# En[ ]:


get_ipython().run_line_magic('cd', '/content')
get_ipython().system('git clone --quiet https://github.com/tensorflow/models.git')

get_ipython().system('pip install tf_slim')

get_ipython().system('apt-get install -qq protobuf-compiler python-pil python-lxml python-tk')

get_ipython().system('pip install -q Cython contextlib2 pillow lxml matplotlib')

get_ipython().system('pip install -q pycocotools')

get_ipython().system('pip install lvis')

get_ipython().run_line_magic('cd', '/content/models/research')
get_ipython().system('protoc object_detection/protos/*.proto --python_out=.')

import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'

get_ipython().system('python object_detection/builders/model_builder_test.py')


# ## Prepara archivos `tfrecord`
#
# ¡Roboflow crea automáticamente nuestros archivos TFRecord y label_map que necesitamos!
#
# **Generar sus propios TFRecords es el único paso que debe cambiar para su propio conjunto de datos personalizado.**
#
# Debido a que necesitamos un archivo TFRecord para nuestros datos de entrenamiento y un archivo TFRecord para nuestros datos de prueba, crearemos dos conjuntos de datos separados en Roboflow y generaremos un conjunto de TFRecords para cada uno.
#
# Para crear un conjunto de datos en Roboflow y generar TFRecords, siga [esta guía paso a paso](https://blog.roboflow.ai/getting-started-with-roboflow/).

# En[ ]:


get_ipython().run_line_magic('cd', '/content/tensorflow-object-detection-faster-rcnn/data')


# En 1]:


# Descarga nuestro conjunto de datos
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Mask+Wearing.v4-raw.tensorflow.zip')
get_ipython().system("unzip -q 'Mask Wearing.v4-raw.tensorflow.zip'")


# En[ ]:


# conjunto de entrenamiento
get_ipython().run_line_magic('ls', 'train')


# En[ ]:


# equipo de prueba
get_ipython().run_line_magic('ls', 'test')


# En[ ]:


# NOTA: ¡Actualice estos nombres de TFRecord de "cells" y "cells_label_map" a sus archivos!
test_record_fname = '/content/tensorflow-object-detection-faster-rcnn/data/test/People.tfrecord'
train_record_fname = '/content/tensorflow-object-detection-faster-rcnn/data/train/People.tfrecord'
label_map_pbtxt_fname = '/content/tensorflow-object-detection-faster-rcnn/data/train/People_label_map.pbtxt'


# ## Descargar modelo base

# En[ ]:


get_ipython().run_line_magic('cd', '/content/models/research')

import os
import shutil
import glob
import urllib.request
import tarfile
MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = '/content/models/research/pretrained_model'

if not (os.path.exists(MODEL_FILE)):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar = tarfile.open(MODEL_FILE)
tar.extractall()
tar.close()

os.remove(MODEL_FILE)
if (os.path.exists(DEST_DIR)):
    shutil.rmtree(DEST_DIR)
os.rename(MODEL, DEST_DIR)


# En[ ]:


get_ipython().system('echo {DEST_DIR}')
get_ipython().system('ls -alh {DEST_DIR}')


# En[ ]:


fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
fine_tune_checkpoint


# ## Configuración de una canalización de entrenamiento

# En[ ]:


import os
pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)

assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)


# En[ ]:


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


# En[ ]:


import re

num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    
    # fine_tune_punto de control
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # Los archivos tfrecord entrenan y prueban.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # etiqueta_mapa_ruta
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Establecer el tamaño del lote de entrenamiento.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Establecer pasos de entrenamiento, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Establecer el número de clases num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)


# En[ ]:


get_ipython().system('cat {pipeline_fname}')


# En[ ]:


model_dir = 'training/'
# Opcionalmente, elimine el contenido en el directorio del modelo de salida para comenzar de nuevo.
get_ipython().system('rm -rf {model_dir}')
os.makedirs(model_dir, exist_ok=True)


# ## Ejecutar Tensorboard (Opcional)

# En[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip -o ngrok-stable-linux-amd64.zip')


# En[ ]:


LOG_DIR = model_dir
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)


# En[ ]:


get_ipython().system_raw('./ngrok http 6006 &')


# ### Obtener enlace de Tensorboard

# En[ ]:


get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c      "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# ## Entrenar al modelo

# En[ ]:


get_ipython().system('python /content/models/research/object_detection/model_main.py      --pipeline_config_path={pipeline_fname}      --model_dir={model_dir}      --alsologtostderr      --num_train_steps={num_steps}      --num_eval_steps={num_eval_steps}')


# En[ ]:


get_ipython().system('ls {model_dir}')


# ## Exportación de un gráfico de inferencia entrenado
# Una vez que se complete su trabajo de entrenamiento, debe extraer el gráfico de inferencia recién entrenado, que luego se usará para realizar la detección de objetos. Esto puede hacerse de la siguiente manera:

# En[ ]:


import re
import numpy as np

output_directory = './fine_tuned_model'

lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join(model_dir, last_model)
print(last_model_path)
get_ipython().system('python /content/models/research/object_detection/export_inference_graph.py      --input_type=image_tensor      --pipeline_config_path={pipeline_fname}      --output_directory={output_directory}      --trained_checkpoint_prefix={last_model_path}')


# En[ ]:


get_ipython().system('ls {output_directory}')


# ## Descarga el archivo `.pb` del modelo

# En[ ]:


import os

pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph2.pb")
assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)


# En[ ]:


get_ipython().system('ls -alh {pb_fname}')


# ### Opción 1: sube el archivo `.pb` a tu Google Drive
# Luego descárguelo de su Google Drive al sistema de archivos local.
#
# Durante este paso, se le pedirá que ingrese el token.

# En[ ]:


# Instale el envoltorio de PyDrive y las bibliotecas de importación.
# Esto solo debe hacerse una vez en un cuaderno.
get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# Autenticar y crear el cliente PyDrive.
# Esto solo debe hacerse una vez en un cuaderno.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

fname = os.path.basename(pb_fname)
# Crear y cargar un archivo de texto.
uploaded = drive.CreateFile({'title': fname})
uploaded.SetContentFile(pb_fname)
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))


# ### Opción 2: Descargue el archivo `.pb` directamente a su sistema de archivos local
# Es posible que este método no sea estable al descargar archivos grandes como el archivo modelo `.pb`. Pruebe la **opción 1** en su lugar si no funciona.

# En[ ]:


from google.colab import files
files.download(pb_fname)


# ### OPCIONAL: Descargue el archivo `label_map.pbtxt`

# En[ ]:


from google.colab import files
files.download(label_map_pbtxt_fname)


# ### OPCIONAL: Descargar el archivo pipline modificado
# Si tiene previsto utilizar el kit de herramientas OpenVINO para convertir el archivo `.pb` en una inferencia más rápida en el hardware de Intel (CPU/GPU, Movidius, etc.)

# En[ ]:


files.download(pipeline_fname)


# En[ ]:


# !tar cfz fine_tuned_model.tar.gz fine_tuned_model
# de archivos de importación de google.colab
# archivos.descargar('fine_tuned_model.tar.gz')


# ## Ejecutar prueba de inferencia
# Prueba con imágenes en el directorio `tensorflow-object-detection/test` del repositorio.
#
# **Para probar con sus propias imágenes, ¡debe colocar sus imágenes dentro del directorio `test` en este cuaderno de Colab!** Más sobre esto a continuación.

# En[ ]:


import os
import glob

# Camino al gráfico de detección congelado. Este es el modelo real que se utiliza para la detección de objetos.
PATH_TO_CKPT = pb_fname

# Lista de las cadenas que se utilizan para agregar la etiqueta correcta para cada caja.
PATH_TO_LABELS = label_map_pbtxt_fname

# Si desea probar el código con sus imágenes, simplemente agregue archivos de imágenes a PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR =  repo_dir_path + "/data/test/"
sample_img = 'https://storage.googleapis.com/roboflow-platform-transforms/Ly2DeBzbwsemGd2ReHk4BFxy8683/cf5ed147e4f2675fbabbc9b0db750ecf/transformed.jpg'
import urllib.request
urllib.request.urlretrieve(sample_img, 
                           PATH_TO_TEST_IMAGES_DIR + "cell.jpg")


assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
print(TEST_IMAGE_PATHS)


# En[ ]:


get_ipython().run_line_magic('cd', '/content/models/research/object_detection')

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Esto es necesario ya que el cuaderno está almacenado en la carpeta object_detection.
sys.path.append("../Cursos/Modern Computer vision")
from object_detection.utils import ops as utils_ops


# Esto es necesario para mostrar las imágenes.
get_ipython().run_line_magic('matplotlib', 'inline')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# En[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Tamaño, en pulgadas, de las imágenes de salida.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Obtenga identificadores para los tensores de entrada y salida
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # El siguiente procesamiento es solo para una sola imagen
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Se requiere reencuadrar para traducir la máscara de las coordenadas del cuadro a las coordenadas de la imagen y ajustar el tamaño de la imagen.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Siga la convención agregando nuevamente la dimensión del lote
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Ejecutar inferencia
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # todas las salidas son matrices numpy float32, así que convierta los tipos según corresponda
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# En[ ]:


# Esto es necesario para mostrar las imágenes.
get_ipython().run_line_magic('matplotlib', 'inline')


# En[ ]:


for image_path in TEST_IMAGE_PATHS:
  try:
    image = Image.open(image_path)
    print(image_path)
    # la representación basada en matriz de la imagen se usará más adelante para preparar la
    # imagen de resultado con cuadros y etiquetas en ella.
    image_np = load_image_into_numpy_array(image)
    # Expanda las dimensiones ya que el modelo espera que las imágenes tengan forma: [1, Ninguno, Ninguno, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Detección real.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualización de los resultados de una detección.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()
  except Exception:
    pass


# En[ ]:


### Agregar sus propias imágenes a tensorflow-object-detection/data
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())


# En[ ]:


# navegar a la carpeta correcta
get_ipython().run_line_magic('cd', '/content/tensorflow-object-detection-faster-rcnn/data/test/')

# función de llamada para cargar
upload_files()


# En[ ]:


while True:
  pass


# En[ ]:




