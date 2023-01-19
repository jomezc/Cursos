#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Filter & Class Maximisation**
# 
# ---
# 
# 
# In this lesson, we use **Keras with a TensorFlow 2.0** to visualise the following (see below). This helps you gain a better understanding of what's going on under the hood and de-mystifies some of the deep learning aspects.**
# 1. Filter Maximisation
# 2. Class Maximisation
# 
# **References:**
# 
# https://github.com/keisen/tf-keras-vis
# 

# 
# ## **Maximizing Filter Activations**
# 
# The process is relatively simple in principle.
# 1. You’ll build a loss function that maximizes the value of a given filter in a given convolution layer
# 2. You’ll use Stochastic Gradient Descent to adjust the values of the input image so as to maximize this activation value. 
# 
# **NOTE** This is easier to implement in TF1.14 so we'll downgrade our Tensorflow package to make this work.

# # **Visualising Conv Filters Maximisations**
# 
# Firstly, we need to install tf-keras-vis. https://github.com/keisen/tf-keras-vis

# In[1]:


get_ipython().system('pip install --upgrade tf-keras-vis tensorflow')


# **Import our libraries**

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

# In[3]:


from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Firstly, we define a function to modify the model**
# 
# Define modifier to replace the model output to target layer's output that has filters you want to visualize.

# In[4]:


layer_name = 'block5_conv3' # The target layer that is the last layer of VGG16.

def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs,
                               outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


# ### **Create ActivationMaximization Instance** 
# 
# If clone argument is True(default), the model will be cloned, so the model instance will be NOT modified, but it takes a machine resources.

# In[5]:


from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model, model_modifier, clone=False)


# ### **Define Loss function**
# You MUST define Loss function that return arbitrary filter values. Here, it returns the value corresponding third filter in block5_conv3 layer. ActivationMaximization will maximize the filter value.

# In[6]:


filter_number = 7
def loss(output):
    return output[..., filter_number]


# ### **Visualize**
# ActivationMaximization will maximize the model output value that is computed by the loss function. Here, we try to visualize a convolutional filter.

# In[7]:


get_ipython().run_cell_magic('time', '', "from tf_keras_vis.utils.callbacks import Print\n\n# Generate max activation\nactivation = activation_maximization(loss, callbacks=[Print(interval=50)])\nimage = activation[0].astype(np.uint8)\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),\n                 'subplot_kw': {'xticks': [], 'yticks': []} }\n                 \nf, ax = plt.subplots(**subplot_args)\nax.imshow(image)\nax.set_title('filter[{:03d}]'.format(filter_number), fontsize=14)\nplt.tight_layout()\nplt.show()\n")


# ## **Now let's visualize multiple convolutional filters**
# 
# #### **Define Loss function**
# When visualizing multiple convolutional filters, you MUST define Loss function that return arbitrary filter values for each layer.

# In[8]:


filter_numbers = [63, 132, 320]

# Define loss function that returns multiple filter outputs.
def loss(output):
    return (output[0, ..., 63], output[1, ..., 132], output[2, ..., 320])


# #### **Create SeedInput values** 
# 
# And then, you MUST prepare seed-input value. In default, when visualizing a conv filter, tf-keras-vis automatically generate seed-input for generating a image. When visualizing multiple conv filters, you MUST manually generate seed-input whose samples-dim is as many as the number of the filters you want to generate.

# In[9]:


# Define seed inputs whose shape is (samples, height, width, channels).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualize** 
# 
# Here, we will visualize 3 images while steps option is to be 512 to get clear images.

# In[10]:


get_ipython().run_cell_magic('time', '', "\n# Generate max activation\nactivations = activation_maximization(loss,\n                                      seed_input=seed_input, # To generate multiple images\n                                      callbacks=[Print(interval=50)])\nimages = [activation.astype(np.uint8) for activation in activations]\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),\n                 'subplot_kw': {'xticks': [], 'yticks': []} }\nf, ax = plt.subplots(**subplot_args)\nfor i, filter_number in enumerate(filter_numbers):\n    ax[i].set_title('filter[{:03d}]'.format(filter_number), fontsize=14)\n    ax[i].imshow(images[i])\n    \nplt.tight_layout()\nplt.show()\n")


# # **Class Maximisation** 
# 
# Finding an input that maximizes a specific class of VGGNet.
# 
# #### **Load libaries and load your pretrained VGG16 Model**
# 
# Load tf.keras.Model¶
# This tutorial use VGG16 model in tf.keras but if you want to use other tf.keras.Models, you can do so by modifing section below.
# 

# In[11]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('{} GPUs'.format(gpus))

from tensorflow.keras.applications.vgg16 import VGG16 as Model

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()


# #### **Define a function to modify the model**
# 
# Define modifier to replace a softmax function of the last layer to a linear function.

# In[12]:


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


# #### **Create ActivationMaximization Instance**
# 
# If clone argument is True(default), the model will be cloned, so the model instance will be NOT modified, but it takes a machine resources.

# In[13]:


from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)


# #### **Define Loss function**
# 
# You MUST define Loss function that return arbitrary category value. Here, we try to visualize a category as defined No.20 (ouzel) of imagenet.
# 
# 

# In[14]:


def loss(output):
    return output[:, 20]


# ### **Visualise**
# 
# The Imagenet Classes - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# In[15]:


get_ipython().run_cell_magic('time', '', "\nfrom tf_keras_vis.utils.callbacks import Print\n\nactivation = activation_maximization(loss,\n                                     callbacks=[Print(interval=50)])\nimage = activation[0].astype(np.uint8)\n\nsubplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (3, 3),\n                 'subplot_kw': {'xticks': [], 'yticks': []} }\nf, ax = plt.subplots(**subplot_args)\nax.imshow(image)\nax.set_title('Ouzel', fontsize=14)\nplt.tight_layout()\nplt.show()\n")


# ### **Visualizing specific output categories** 
# 
# Now, let's visualize multiple categories at once!
# 
# #### **Define Loss function**
# 
# You MUST define loss function that return arbitrary category values. Here, we try to visualize categories as defined No.1 (Goldfish), No.294 (Bear) and No.413 (Assault rifle) of imagenet.
# 

# In[16]:


image_titles = ['Goldfish', 'Bear', 'Assault rifle']

def loss(output):
    return (output[0, 1], output[1, 294], output[2, 413])


# #### **Create SeedInput values** 
# And then, you MUST prepare seed-input value. In default, when visualizing a conv filter, tf-keras-vis automatically generate seed-input for generating a image. When visualizing multiple conv filters, you MUST manually generate seed-input whose samples-dim is as many as the number of the filters you want to generate.

# In[17]:


# Define seed inputs whose shape is (samples, height, width, channels).

seed_input = tf.random.uniform((3, 224, 224, 3), 0, 255)


# #### **Visualize**
# 
# Here, we will visualize 3 images while steps option is to be 512 to get clear images.

# In[18]:


get_ipython().run_cell_magic('time', '', "\n# Do 500 iterations and Generate an optimizing animation\nactivations = activation_maximization(loss,\n                                      seed_input=seed_input,\n                                      steps=512,\n                                      callbacks=[ Print(interval=50)])\nimages = [activation.astype(np.uint8) for activation in activations]\n\n# Render\nsubplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),\n                 'subplot_kw': {'xticks': [], 'yticks': []} }\nf, ax = plt.subplots(**subplot_args)\nfor i, title in enumerate(image_titles):\n    ax[i].set_title(title, fontsize=14)\n    ax[i].imshow(images[i])\nplt.tight_layout()\n\nplt.show()\n")


# In[ ]:




