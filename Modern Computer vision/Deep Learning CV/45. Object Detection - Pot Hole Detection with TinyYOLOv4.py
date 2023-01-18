#!/usr/bin/env python
# coding: utf-8

# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# ## **Training TinyYOLOv4 using a Pot Hole Dataset**
# 
# ## Introduction
# 
# 
# In this notebook, we implement the tiny version of [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) for training our public [Pot Hole Dataset] which consists of 665 images(https://public.roboflow.com/object-detection/pothole), [YOLOv4 tiny](https://github.com/AlexeyAB/darknet/issues/6067).
# 
# We also recommend reading our blog post on [Training YOLOv4 on custom data](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) side by side.
# 
# We will take the following steps to implement YOLOv4 on our custom data:
# * Configure our GPU environment on Google Colab
# * Install the Darknet YOLOv4 training environment
# * Download our custom dataset for YOLOv4 and set up directories
# * Configure a custom YOLOv4 training config file for Darknet
# * Train our custom YOLOv4 object detector
# * Reload YOLOv4 trained weights and make inference on test images
# 
# When you are done you will have a custom detector that you can use. It will make inference like this:
# 
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/pothole.png)
# 
# 
# ### **Reach out for support**
# 
# If you run into any hurdles on your own data set or just want to share some cool results in your own domain, [reach out!](https://roboflow.ai) 
# 
# 
# 
# #### ![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png)

# # Configuring cuDNN on Colab for YOLOv4
# 
# 

# In[ ]:


# CUDA: Let's check that Nvidia CUDA drivers are already pre-installed and which version is it.
get_ipython().system('/usr/local/cuda/bin/nvcc --version')
# We need to install the correct cuDNN according to this output


# In[ ]:


#take a look at the kind of GPU we have
get_ipython().system('nvidia-smi')


# In[ ]:


# This cell ensures you have the correct architecture for your respective GPU
# If you command is not found, look through these GPUs, find the respective
# GPU and add them to the archTypes dictionary

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# Tesla K80 
# ARCH= -gencode arch=compute_37,code=sm_37

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61

# GP100/Tesla P100 - DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]
import os
os.environ['GPU_TYPE'] = str(os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read())

def getGPUArch(argument):
  try:
    argument = argument.strip()
    # All Colab GPUs
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


# # Installing Darknet for YOLOv4 on Colab
# 
# 
# 

# In[ ]:


get_ipython().run_line_magic('cd', '/content/')
get_ipython().run_line_magic('rm', '-rf darknet')


# In[ ]:


#we clone the fork of darknet maintained by roboflow
#small changes have been made to configure darknet for training
get_ipython().system('git clone https://github.com/roboflow-ai/darknet.git')


# In[ ]:


#install environment from the Makefile
get_ipython().run_line_magic('cd', '/content/darknet/')
# compute_37, sm_37 for Tesla K80
# compute_75, sm_75 for Tesla T4
# !sed -i 's/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= -gencode arch=compute_75,code=sm_75/g' Makefile

#install environment from the Makefile
#note if you are on Colab Pro this works on a P100 GPU
#if you are on Colab free, you may need to change the Makefile for the K80 GPU
#this goes for any GPU, you need to change the Makefile to inform darknet which GPU you are running on.
get_ipython().system("sed -i 's/OPENCV=0/OPENCV=1/g' Makefile")
get_ipython().system("sed -i 's/GPU=0/GPU=1/g' Makefile")
get_ipython().system("sed -i 's/CUDNN=0/CUDNN=1/g' Makefile")
get_ipython().system('sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= ${ARCH_VALUE}/g" Makefile')
get_ipython().system('make')


# In[ ]:


#download the newly released yolov4-tiny weights
get_ipython().run_line_magic('cd', '/content/darknet')
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights')
get_ipython().system('wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29')


# # Set up Custom Dataset for YOLOv4

# We'll use Roboflow to convert our dataset from any format to the YOLO Darknet format. 
# 
# 1. To do so, create a free [Roboflow account](https://app.roboflow.ai).
# 2. Upload your images and their annotations (in any format: VOC XML, COCO JSON, TensorFlow CSV, etc).
# 3. Apply preprocessing and augmentation steps you may like. We recommend at least `auto-orient` and a `resize` to 416x416. Generate your dataset.
# 4. Export your dataset in the **YOLO Darknet format**.
# 5. Copy your download link, and paste it below.
# 
# See our [blog post](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) for greater detail.
# 
# In this example, I used the open source [BCCD Dataset](https://public.roboflow.ai/object-detection/bccd). (You can `fork` it to your Roboflow account to follow along.)

# In[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/Pothole.v1-raw.darknet.zip')
get_ipython().system('unzip -q Pothole.v1-raw.darknet.zip')


# In[ ]:


#if you already have YOLO darknet format, you can skip this step
#otherwise we recommend formatting in Roboflow
get_ipython().run_line_magic('cd', '/content/darknet')
get_ipython().system('curl -L "https://public.roboflow.com/ds/I2ZXTaUHUY?key=BBFNVcFack" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip')


# In[ ]:


#Set up training file directories for custom dataset
get_ipython().run_line_magic('cd', '/content/darknet/')
get_ipython().run_line_magic('cp', 'train/_darknet.labels data/obj.names')
get_ipython().run_line_magic('mkdir', 'data/obj')
#copy image and labels
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

#write train file (just the image list)
import os

with open('data/train.txt', 'w') as out:
  for img in [f for f in os.listdir('train') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

#write the valid file (just the image list)
import os

with open('data/valid.txt', 'w') as out:
  for img in [f for f in os.listdir('valid') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')


# # Write Custom Training Config for YOLOv4

# In[ ]:


#we build config dynamically based on number of classes
#we build iteratively from base config files. This is the same file shape as cfg/yolo-obj.cfg
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

#Instructions from the darknet repo
#change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
#change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
if os.path.exists('./cfg/custom-yolov4-tiny-detector.cfg'): os.remove('./cfg/custom-yolov4-tiny-detector.cfg')


#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# In[ ]:


get_ipython().run_cell_magic('writetemplate', './cfg/custom-yolov4-tiny-detector.cfg', '[net]\n# Testing\n#batch=1\n#subdivisions=1\n# Training\nbatch=64\nsubdivisions=24\nwidth=416\nheight=416\nchannels=3\nmomentum=0.9\ndecay=0.0005\nangle=0\nsaturation = 1.5\nexposure = 1.5\nhue=.1\n\nlearning_rate=0.00261\nburn_in=1000\nmax_batches = {max_batches}\npolicy=steps\nsteps={steps_str}\nscales=.1,.1\n\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=2\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -1,-2\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -6,-1\n\n[maxpool]\nsize=2\nstride=2\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -1,-2\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -6,-1\n\n[maxpool]\nsize=2\nstride=2\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -1,-2\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[route]\nlayers = -6,-1\n\n[maxpool]\nsize=2\nstride=2\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n##################################\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters={num_filters}\nactivation=linear\n\n\n\n[yolo]\nmask = 3,4,5\nanchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319\nclasses={num_classes}\nnum=6\njitter=.3\nscale_x_y = 1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\nignore_thresh = .7\ntruth_thresh = 1\nrandom=0\nnms_kind=greedynms\nbeta_nms=0.6\n\n[route]\nlayers = -4\n\n[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\nactivation=leaky\n\n[upsample]\nstride=2\n\n[route]\nlayers = -1, 23\n\n[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[convolutional]\nsize=1\nstride=1\npad=1\nfilters={num_filters}\nactivation=linear\n\n[yolo]\nmask = 1,2,3\nanchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319\nclasses={num_classes}\nnum=6\njitter=.3\nscale_x_y = 1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\nignore_thresh = .7\ntruth_thresh = 1\nrandom=0\nnms_kind=greedynms\nbeta_nms=0.6\n')


# In[ ]:


#here is the file that was just written. 
#you may consider adjusting certain things

#like the number of subdivisions 64 runs faster but Colab GPU may not be big enough
#if Colab GPU memory is too small, you will need to adjust subdivisions to 16
get_ipython().run_line_magic('cat', 'cfg/custom-yolov4-tiny-detector.cfg')


# # Train Custom YOLOv4 Detector

# In[ ]:


get_ipython().system('./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map')
#If you get CUDA out of memory adjust subdivisions above!
#adjust max batches down for shorter training above


# # Infer Custom Objects with Saved YOLOv4 Weights

# In[ ]:


#define utility function
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
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()


# In[ ]:


#check if weigths have saved yet
#backup houses the last weights for our detector
#(file yolo-obj_last.weights will be saved to the build\darknet\x64\backup\ for each 100 iterations)
#(file yolo-obj_xxxx.weights will be saved to the build\darknet\x64\backup\ for each 1000 iterations)
#After training is complete - get result yolo-obj_final.weights from path build\darknet\x64\bac
get_ipython().system('ls backup')
#if it is empty you haven't trained for long enough yet, you need to train for at least 100 iterations


# In[ ]:


#coco.names is hardcoded somewhere in the detector
get_ipython().run_line_magic('cp', 'data/obj.names data/coco.names')


# In[ ]:


#/test has images that we can test our detector on
test_images = [f for f in os.listdir('test') if f.endswith('.jpg')]
import random
img_path = "test/" + random.choice(test_images);

#test out our detector!
get_ipython().system('./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights {img_path} -dont-show')
imShow('/content/darknet/predictions.jpg')


# In[ ]:


while True:
  pass


# In[ ]:




