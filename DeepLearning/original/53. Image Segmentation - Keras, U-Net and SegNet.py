#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Semantic Segmentation - U-Net and SegNet**
# 
# ---
# We are going to use the image-segmentation-keras to load pretrained models, train them via transfer learning and run inference on images.

# ## **Install the package**

# you should modify the line 77 in image-segmentation-keras/keras_segmentation/models/vgg16.py by replacing:
# VGG_Weights_path = keras.utils.get_file( by VGG_Weights_path = tf.keras.utils.get_file(
# do not forgot to add import tensorflow as tf and reinstall the image-segmentation-keras library

# In[1]:


#!git clone https://github.com/divamgupta/image-segmentation-keras
get_ipython().system('git clone https://github.com/rajeevratan84/image-segmentation-keras.git')


# In[2]:


get_ipython().run_line_magic('cd', 'image-segmentation-keras')
get_ipython().system('python setup.py install')


# ### **Download the dataset**

# In[3]:


get_ipython().system('wget https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip && unzip -q dataset1.zip')


# ### **Initialize the model**

# In[4]:


from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640)


# ### **Train the model**

# In[5]:


model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5  )


# In[6]:


out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png")


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt


# In[8]:


plt.imshow(out)


# In[9]:


from IPython.display import Image
Image('/tmp/out.png')


# ## **Display with Legend**

# In[10]:


o = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    class_names = [ "Sky", "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"])


# In[11]:


from IPython.display import Image
Image('/tmp/out.png')


# ## **Now let's load and train a SegNet Model**

# In[ ]:


from keras_segmentation.models.segnet import segnet

model = segnet(n_classes=50 ,  input_height=320, input_width=640)

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5)


# In[ ]:


from IPython.display import Image

out = model.predict_segmentation(
    inp = "dataset1/images_prepped_test/0016E5_07965.png",
    out_fname = "out.png")

Image('out.png')


# In[ ]:




