#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **GradCAM, GradCAM++ and Faster-ScoreCAM Visualisations**
# 
# ---
# 
# 
# In this lesson, we use **Keras with a TensorFlow 2.0** to visualise the following (see below). This helps you gain a better understanding of what's going on under the hood and de-mystifies some of the deep learning aspects.
# 
# 1. Learn to use the GradCAM, GradCAM++, ScoreCAM and Faster-ScoreCAM to see where our CNN is 'looking'
# 
# **References:**
# 
# https://github.com/keisen/tf-keras-vis
# 

# #### **Install Libraries**
# 
# Firstly, we need to install tf-keras-vis.

# In[1]:


get_ipython().system('pip install --upgrade tf-keras-vis tensorflow')


# #### **Load our libraries**
# 
# 

# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))


# ### **Load a pretrained VGG16 model.**

# In[8]:


from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Load images**
# 
# tf-keras-vis support evaluating batch-wisely that includes multiple images. Here, we load three pictures of goldfish, bear and assault-rifle as inputs data.

# In[9]:


get_ipython().system('wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/goldfish.jpg')
get_ipython().system('wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/bear.jpg')
get_ipython().system('wget https://github.com/keisen/tf-keras-vis/raw/master/docs/examples/images/soldiers.jpg')


# In[10]:


from tensorflow.keras.preprocessing.image import load_img

# Image titles
image_titles = ['Goldfish', 'Bear', 'Assault rifle']

# Load images
img1 = load_img('goldfish.jpg', target_size=(224, 224))
img2 = load_img('bear.jpg', target_size=(224, 224))
img3 = load_img('soldiers.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data
X = preprocess_input(images)

# Rendering
subplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.show()


# 
# The Imagenet Classes - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# ## **Define necessary functions** 
# 
# #### **Define Loss functions** 
# 
# You MUST define loss function that return target scores. Here, it returns the scores corresponding Goldfish, Bear, Assault Rifle.

# In[11]:


# The `output` variable refer to the output of the model,
# so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][1], output[1][294], output[2][413])


# #### **Define Model-Modifier function**
# 
# Then, when the softmax activation function is applied to the last layer of model, it may obstruct generating the attention images, so you need to replace the function to a linear function. Here, we does so using model_modifier.
# 
# 

# In[12]:


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


# ## **GradCAM**
# 
# GradCAM is another way of visualizing attention over input. Instead of using gradients with respect to model outputs, it uses penultimate (pre Dense layer) Conv layer output.

# In[13]:


get_ipython().run_cell_magic('time', '', "from tensorflow.keras import backend as K\nfrom tf_keras_vis.utils import normalize\nfrom matplotlib import cm\nfrom tf_keras_vis.gradcam import Gradcam\n\n# Create Gradcam object\ngradcam = Gradcam(model,\n                  model_modifier=model_modifier,\n                  clone=False)\n\n# Generate heatmap with GradCAM\ncam = gradcam(loss,\n              X,\n              penultimate_layer=-1, # model.layers number\n             )\ncam = normalize(cam)\n\nf, ax = plt.subplots(**subplot_args)\nfor i, title in enumerate(image_titles):\n    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)\n    ax[i].set_title(title, fontsize=14)\n    ax[i].imshow(images[i])\n    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay\nplt.tight_layout()\nplt.show()\n")


# ## **GradCAM++**
# 
# GradCAM++ can provide better visual explanations of CNN model predictions. In tf-keras-vis, GradcamPlusPlus (GradCAM++) class has most of compatibility with Gradcam. So you can use GradcamPlusPlus if you just replace classname from Gradcam to GradcamPlusPlus.
# 
# 

# In[14]:


get_ipython().run_cell_magic('time', '', '\nfrom tf_keras_vis.gradcam import GradcamPlusPlus\n\n# Create GradCAM++ object, Just only repalce class name to "GradcamPlusPlus"\n# gradcam = Gradcam(model, model_modifier, clone=False)\ngradcam = GradcamPlusPlus(model,\n                          model_modifier,\n                          clone=False)\n\n# Generate heatmap with GradCAM++\ncam = gradcam(loss,\n              X,\n              penultimate_layer=-1, # model.layers number\n             )\ncam = normalize(cam)\n\nf, ax = plt.subplots(**subplot_args)\nfor i, title in enumerate(image_titles):\n    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)\n    ax[i].set_title(title, fontsize=14)\n    ax[i].imshow(images[i])\n    ax[i].imshow(heatmap, cmap=\'jet\', alpha=0.5)\nplt.tight_layout()\nplt.show()\n')


# As you can see above, Now, the visualized attentions almost completely cover the target objects!
# 
# ## **ScoreCAM**
# 
# Lastly, Here, we show you ScoreCAM. SocreCAM is an another method that generate Class Activation Map. The characteristic is that it's the gradient-free CAM method unlike GradCAM/GradCAM++.
# 
# In default, this method takes too much time, so in the cell below ScoreCAM is NOT run with CPU.
# 
# 

# In[15]:


get_ipython().run_cell_magic('time', '', '\nfrom tf_keras_vis.scorecam import ScoreCAM\n\n# Create ScoreCAM object\nscorecam = ScoreCAM(model, model_modifier, clone=False)\n\n# This cell takes toooooooo much time, so only doing with GPU.\nif gpus > 0:\n    # Generate heatmap with ScoreCAM\n    cam = scorecam(loss,\n                   X,\n                   penultimate_layer=-1, # model.layers number\n                  )\n    cam = normalize(cam)\n\n    f, ax = plt.subplots(**subplot_args)\n    for i, title in enumerate(image_titles):\n        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)\n        ax[i].set_title(title, fontsize=14)\n        ax[i].imshow(images[i])\n        ax[i].imshow(heatmap, cmap=\'jet\', alpha=0.5)\n    plt.tight_layout()\n    plt.show()\nelse:\n    print("NOTE: Change to GPU to see visual output\\n")\n')


# ##**Faster-ScoreCAM**
# 
# As you see above, ScoreCAM need huge processing power, but there is a good news for us. Faster-ScorecAM that makes ScoreCAM to be more efficient was devised by @tabayashi0117.
# 
# https://github.com/tabayashi0117/Score-CAM/blob/master/README.md#faster-score-cam
# 
# > We thought that several channels were dominant in generating the final heat map. Faster-Score-CAM adds the processing of “use only channels with large variances as mask images” to Score-CAM. (max_N = -1 is the original Score-CAM).

# In[16]:


get_ipython().run_cell_magic('time', '', "\n# Create ScoreCAM object\nscorecam = ScoreCAM(model, model_modifier, clone=False)\n\n# Generate heatmap with Faster-ScoreCAM\ncam = scorecam(loss,\n               X,\n               penultimate_layer=-1, # model.layers number\n               max_N=10\n              )\ncam = normalize(cam)\n\nf, ax = plt.subplots(**subplot_args)\nfor i, title in enumerate(image_titles):\n    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)\n    ax[i].set_title(title, fontsize=14)\n    ax[i].imshow(images[i])\n    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)\nplt.tight_layout()\nplt.show()\n")


# In[ ]:




