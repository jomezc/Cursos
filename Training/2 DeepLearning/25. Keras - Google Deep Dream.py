#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Googe Deep Dream in Keras Tensorflow 2.0**
# 
# ---
# 
# 
# In this lesson, we learn to implement the **Google Deep Dream Algorithm** using Keras with Tensorflow 2.0. This method was first introduced by Alexander Mordvintsev from Google in July 2015.
# 
# It enables us to provide the 'Deep Dream' effect which produces hallucinogenic-like visuals. 
# 
# In this tutorial we:
# 
# 1. Load our modules and base image
# 2. Create our image preprocessing and deprocessing utilities
# 3. Compute the Deep Dream loss
# 4. Set up the gradient ascent loop for one octave
# 
# Source - https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/deep_dream.ipynb#scrollTo=jIdSV7PumdQh

# ## **1. Load modules and base image**

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3

base_image_path = keras.utils.get_file("castara-tobago.jpeg", "https://github.com/rajeevratan84/ModernComputerVision/raw/main/castara-tobago.jpeg")
result_prefix = "sky_dream"

# These are the names of the layers for which we try to maximize activation,
# as well as their weight in the final loss we try to maximize.
# You can tweak these setting to obtain new visual effects.
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 15.0


# ### **Our Base Image**

# In[2]:


from IPython.display import Image, display

display(Image(base_image_path))


# ## **The Deep Dream Process**
# 
# 1. Load the original image.
# 2. Define a number of processing scales ("octaves"), from smallest to largest.
# 3. Resize the original image to the smallest scale.
# 4. For every scale, starting with the smallest (i.e. current one):
#   - Run gradient ascent
#   - Upscale image to the next scale
#   - Reinject the detail that was lost at upscaling time
# 5. Stop when we are back to the original size. To obtain the detail lost during upscaling, we simply take the original image, shrink it down, upscale it, and compare the result to the (resized) original image.

# ## **2. Create our image preprocessing and deprocessing utilities**
# 

# In[3]:


def preprocess_image(image_path):
    '''Util function to open, resize and format pictures
     into appropriate arrays.'''
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    '''Util function to convert a NumPy array into a valid image.'''
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception v3 preprocessing
    x /= 2.0
    x += 0.5
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# ## **3. Compute the Deep Dream loss**
# 
# We first build a feature extraction model to retrieve the activations of our target layers given an input image.
# 

# In[8]:


# Build an InceptionV3 model loaded with pre-trained ImageNet weights
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# Set up a model that returns the activation values for every target layer
# (as a dict)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)


# The actual loss computation is very simple:

# In[4]:


def compute_loss(input_image):
    features = feature_extractor(input_image)
    ''' Initialize the loss '''
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


# ## **4. Set up the gradient ascent loop for one octave**

# In[5]:


@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img


# ### **Now run the training loop, iterating over different octave**

# In[9]:


original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  # Make a copy
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.preprocessing.image.save_img(result_prefix + ".png", deprocess_image(img.numpy()))


# ## **Display your final output**

# In[10]:


display(Image(result_prefix + ".png"))


# In[ ]:




