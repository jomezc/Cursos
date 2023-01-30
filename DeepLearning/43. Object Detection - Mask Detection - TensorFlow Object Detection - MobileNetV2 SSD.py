#!/usr/bin/env python
# coding: utf-8

# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# 
# ## **Training MobileNetSSD Object Detection on a Mask Dataset**
# 
# Note! For a most up to date version of this notebook, make sure you copy from:
# 
# 💡 Recommendation: [Open this blog post](https://blog.roboflow.ai/training-a-tensorflow-object-detection-model-with-a-custom-dataset/) to continue.
# 
# ### **Overview**
# 
# This notebook walks through how to train a MobileNet object detection model using the TensorFlow 1.5 Object Detection API.
# 
# In this specific example, we'll training an object detection model to recognize cells types: white blood cells, red blood cells and platelets. **To adapt this example to train on your own dataset, you only need to change two lines of code in this notebook.**
# 
# Everything in this notebook is also hosted on this [GitHub repo](https://github.com/josephofiowa/tensorflow-object-detection).
# 
# ![Mask Detector Output](https://github.com/rajeevratan84/ModernComputerVision/raw/main/mask.png)
# 
# **Credit to [DLology](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) and [Tony607](https://github.com/Tony607)**, whom wrote the first notebook on which much of this is example is based. 
# 
# ### **Our Data**
# 
# We'll be using an open source Mask Wearing Dataset which contains 149 images and is hosted publicly on Roboflow [here](https://public.roboflow.com/object-detection/mask-wearing).
# 
# When adapting this example to your own data, create two datasets in Roboflow: `train` and `test`. Use Roboflow to generate TFRecords for each, replace their URLs in this notebook, and you're able to train on your own custom dataset.
# 
# ### **Our Model**
# 
# We'll be training a MobileNetSSDv2 (single shot detector). This specific model is a one-short learner, meaning each image only passes through the network once to make a prediction, which allows the architecture to be very performant for mobile hardware.
# 
# The model arechitecture is one of many available via TensorFlow's [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).
# 
# As a note, this notebook presumes TensorFlow 1.5 as TensorFlow 2.0 has yet to fully support the object detection API.
# 
# ### **Training**
# 
# Google Colab provides free GPU resources. Click "Runtime" → "Change runtime type" → Hardware Accelerator dropdown to "GPU."
# 
# Colab does have memory limitations, and notebooks must be open in your browser to run. Sessions automatically clear themselves after 12 hours.
# 
# ### **Inference**
# 
# We'll run inference directly in this notebook, and on three test images contained in the "test" folder from our GitHub repo. 
# 
# When adapting to your own dataset, you'll need to add test images to the `test` folder located at `tensorflow-object-detection/test`.
# 
# ### **About**
# 
# [Roboflow](https://roboflow.ai) makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless.
# 
# Developers reduce 50% of their boilerplate code when using Roboflow's workflow, automate labelling quality assurance, save training time, and increase model reproducibility.
# 
# #### ![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png)
# 
# 
# 
# 
# 
# 

# ## Configs and Hyperparameters
# 
# Support a variety of models, you can find more pretrained model from [Tensorflow detection model zoo: COCO-trained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models), as well as their pipline config files in [object_detection/samples/configs/](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

# In[ ]:


# If you forked the repo, you can replace the link.
repo_url = 'https://github.com/roboflow-ai/tensorflow-object-detection-faster-rcnn'

# Number of training steps - 1000 will train very quickly, but more steps will increase accuracy.
num_steps = 30000  # 200000 to improve

# Number of evaluation steps.
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

# Pick the model you want to use
# Select a model in `MODELS_CONFIG`.
selected_model = 'ssd_mobilenet_v2'

# Name of the object detection model to use.
MODEL = MODELS_CONFIG[selected_model]['model_name']

# Name of the pipline file in tensorflow object detection API.
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

# Training batch size fits in Colabe's Tesla K80 GPU memory for selected model.
batch_size = MODELS_CONFIG[selected_model]['batch_size']


# In[ ]:


# use TF 1.x for Object Detection APIs as they are not ported to TF 2.0 yet
get_ipython().run_line_magic('tensorflow_version', '1.x')


# ## Clone the `tensorflow-object-detection` repository or your fork.

# In[ ]:


import os

get_ipython().run_line_magic('cd', '/content')

repo_dir_path = os.path.abspath(os.path.join('', os.path.basename(repo_url)))

get_ipython().system('git clone {repo_url}')
get_ipython().run_line_magic('cd', '{repo_dir_path}')
get_ipython().system('git pull')


# ## Install required packages

# In[ ]:


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


# ## Prepare `tfrecord` files
# 
# Roboflow automatically creates our TFRecord and label_map files that we need!
# 
# **Generating your own TFRecords the only step you need to change for your own custom dataset.**
# 
# Because we need one TFRecord file for our training data, and one TFRecord file for our test data, we'll create two separate datasets in Roboflow and generate one set of TFRecords for each.
# 
# To create a dataset in Roboflow and generate TFRecords, follow [this step-by-step guide](https://blog.roboflow.ai/getting-started-with-roboflow/).

# In[ ]:


get_ipython().run_line_magic('cd', '/content/tensorflow-object-detection-faster-rcnn/data')


# In[1]:


# Download our dataset
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Mask+Wearing.v4-raw.tensorflow.zip')
get_ipython().system("unzip -q 'Mask Wearing.v4-raw.tensorflow.zip'")


# In[ ]:


# training set
get_ipython().run_line_magic('ls', 'train')


# In[ ]:


# test set
get_ipython().run_line_magic('ls', 'test')


# In[ ]:


# NOTE: Update these TFRecord names from "cells" and "cells_label_map" to your files!
test_record_fname = '/content/tensorflow-object-detection-faster-rcnn/data/test/People.tfrecord'
train_record_fname = '/content/tensorflow-object-detection-faster-rcnn/data/train/People.tfrecord'
label_map_pbtxt_fname = '/content/tensorflow-object-detection-faster-rcnn/data/train/People_label_map.pbtxt'


# ## Download base model

# In[ ]:


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


# In[ ]:


get_ipython().system('echo {DEST_DIR}')
get_ipython().system('ls -alh {DEST_DIR}')


# In[ ]:


fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
fine_tune_checkpoint


# ## Configuring a Training Pipeline

# In[ ]:


import os
pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)

assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)


# In[ ]:


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


# In[ ]:


import re

num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)


# In[ ]:


get_ipython().system('cat {pipeline_fname}')


# In[ ]:


model_dir = 'training/'
# Optionally remove content in output model directory to fresh start.
get_ipython().system('rm -rf {model_dir}')
os.makedirs(model_dir, exist_ok=True)


# ## Run Tensorboard(Optional)

# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip -o ngrok-stable-linux-amd64.zip')


# In[ ]:


LOG_DIR = model_dir
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)


# In[ ]:


get_ipython().system_raw('./ngrok http 6006 &')


# ### Get Tensorboard link

# In[ ]:


get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c      "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# ## Train the model

# In[ ]:


get_ipython().system('python /content/models/research/object_detection/model_main.py      --pipeline_config_path={pipeline_fname}      --model_dir={model_dir}      --alsologtostderr      --num_train_steps={num_steps}      --num_eval_steps={num_eval_steps}')


# In[ ]:


get_ipython().system('ls {model_dir}')


# ## Exporting a Trained Inference Graph
# Once your training job is complete, you need to extract the newly trained inference graph, which will be later used to perform the object detection. This can be done as follows:

# In[ ]:


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


# In[ ]:


get_ipython().system('ls {output_directory}')


# ## Download the model `.pb` file

# In[ ]:


import os

pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph2.pb")
assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)


# In[ ]:


get_ipython().system('ls -alh {pb_fname}')


# ### Option1 : upload the `.pb` file to your Google Drive
# Then download it from your Google Drive to local file system.
# 
# During this step, you will be prompted to enter the token.

# In[ ]:


# Install the PyDrive wrapper & import libraries.
# This only needs to be done once in a notebook.
get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

fname = os.path.basename(pb_fname)
# Create & upload a text file.
uploaded = drive.CreateFile({'title': fname})
uploaded.SetContentFile(pb_fname)
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))


# ### Option2 :  Download the `.pb` file directly to your local file system
# This method may not be stable when downloading large files like the model `.pb` file. Try **option 1** instead if not working.

# In[ ]:


from google.colab import files
files.download(pb_fname)


# ### OPTIONAL: Download the `label_map.pbtxt` file

# In[ ]:


from google.colab import files
files.download(label_map_pbtxt_fname)


# ### OPTIONAL: Download the modified pipline file
# If you plan to use OpenVINO toolkit to convert the `.pb` file to inference faster on Intel's hardware (CPU/GPU, Movidius, etc.)

# In[ ]:


files.download(pipeline_fname)


# In[ ]:


# !tar cfz fine_tuned_model.tar.gz fine_tuned_model
# from google.colab import files
# files.download('fine_tuned_model.tar.gz')


# ## Run inference test
# Test with images in repository `tensorflow-object-detection/test` directory.
# 
# **To test with your own images, you need to place your images inside the `test` directory in this Colab notebook!** More on this below.

# In[ ]:


import os
import glob

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = pb_fname

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = label_map_pbtxt_fname

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
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


# In[ ]:


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

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../Cursos/Modern Computer vision")
from object_detection.utils import ops as utils_ops


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# In[ ]:


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

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
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
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
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
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# In[ ]:


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for image_path in TEST_IMAGE_PATHS:
  try:
    image = Image.open(image_path)
    print(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
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


# In[ ]:


### Adding your own images to tensorflow-object-detection/data
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())


# In[ ]:


# navigate to correct folder
get_ipython().run_line_magic('cd', '/content/tensorflow-object-detection-faster-rcnn/data/test/')

# call function to upload
upload_files()


# In[ ]:


while True:
  pass


# In[ ]:




