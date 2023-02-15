#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Neural Style Transfer in Keras Tensorflow 2.0**
# 
# ---
# 
# 
# In this lesson, we first learn to implement the **Neural Style Transfer Algorithm** using Keras with Tensorflow 2.0. We also start learning to load and use models from the TF-Hub.
# 
# We apply the technique known as *neural style transfer* shown in the research published here <a href="https://arxiv.org/abs/1508.06576" class="external">A Neural Algorithm of Artistic Style</a> (Gatys et al.). 
# 
# In this tutorial we demonstrate the original style-transfer algorithm. It optimizes the image content to a particular style. Modern approaches train a model to generate the stylized image directly (similar to [cyclegan](cyclegan.ipynb)). This approach is much faster (up to 1000x).
# 
# 1. Setup, load modules and helper function
# 2. Fast Style Transfer using TF-Hub
# 3. Implementing our model from scratch
# 4. Build the model
# 5. Extract style and content
# 6. Running Gradient Descent
# 7. Total variation loss - Reducing high frequency artifacts
# 8. Re-run the optimisation
# 
# Source - https://www.tensorflow.org/tutorials/generative/style_transfer
# 
# 

# ## **1. Setup, load modules and helper functions**
# 
# 
# 

# In[36]:


import os
import tensorflow as tf

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# In[37]:


# Set our image plot parameters and import some modules
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


# In[38]:


# function that transforms a tensor to image
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


# In[39]:


# load our content and style images
content_path = tf.keras.utils.get_file('labrador.jpeg', 'https://github.com/rajeevratan84/ModernComputerVision/raw/main/labrador.jpeg')
style_path = tf.keras.utils.get_file('the_wave.jpg','https://github.com/rajeevratan84/ModernComputerVision/raw/main/the_wave.jpg')


# mosaic - https://github.com/rajeevratan84/ModernComputerVision/raw/main/mosaic.jpg
# feathers - https://github.com/rajeevratan84/ModernComputerVision/raw/main/feathers.jpg


# Define a function to load an image and limit its maximum dimension to 512 pixels.

# In[40]:


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


# Create a simple function to display an image:

# In[41]:


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# In[42]:


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


# ## **2. Fast Style Transfer using TF-Hub**
# 
# Before we implement the algorithm on our own, let's try using a simple pretrained model founder on TensorFlow Hub. 
# 
# [TensorFlow Hub model](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

# In[43]:


# import tensorflow hub which allows us to directly download pretrained models
import tensorflow_hub as hub

# Get our model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# input or style and content images
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# convert the returned tensor to an image
tensor_to_image(stylized_image)


# ## **3. Implementing our model from scratch**
# 
# ### **Define content and style representations**
# 
# We use the intermediate layers of the model to get the *content* and *style* representations of the image. 
# 
# Starting from the network's input layer, the first few layer activations represent low-level features like edges and textures. As you step through the network, the final few layers represent higher-level features—object parts like *wheels* or *eyes*. 
# 
# Here we'll be using the VGG19 network architecture, a pretrained image classification network. These intermediate layers are necessary to define the representation of content and style from the images. For an input image, try to match the corresponding style and content target representations at these intermediate layers.
# 

# In[44]:


# Load VGG19 without the head
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# print a list of layer names
for layer in vgg.layers:
  print(layer.name)


# Choose intermediate layers from the network to represent the style and content of the image:

# In[45]:


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# #### **Why choose Intermediate layers for style and content representations?**
# 
# So why do these intermediate outputs within our pretrained image classification network allow us to define style and content representations?
# 
# At a high level, in order for a network to perform image classification (which this network has been trained to do), it must understand the image. This requires taking the raw image as input pixels and building an internal representation that converts the raw image pixels into a complex understanding of the features present within the image.
# 
# This is also a reason why convolutional neural networks are able to generalize well: **they’re able to capture the invariances and defining features within classes** (e.g. cats vs. dogs) that are agnostic to background noise and other nuisances. Thus, somewhere between where the raw image is fed into the model and the output classification label, the model serves as a complex feature extractor. By accessing intermediate layers of the model, you're able to describe the content and style of input images.

# ## **4. Build the model**
# 
# The networks in `tf.keras.applications` are designed so you can easily extract the intermediate layer values using the Keras functional API.
# 
# To define a model using the functional API, specify the inputs and outputs:
# 
# `model = Model(inputs, outputs)`
# 
# This following function builds a VGG19 model that returns a list of intermediate layer outputs:

# In[46]:


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  # 
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


# Now we use the function above to get our style extractor and style outputs

# In[47]:


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()


# ### **Calculate style**
# 
# The content of an image is represented by the values of the intermediate feature maps.
# 
# It turns out, **the style of an image can be described by the means and correlations across the different feature maps.** 
# 
# We can use this to calculate a **Gram matrix** that includes this information by taking the outer product of the feature vector with itself at each location, and averaging that outer product over all locations. This Gram matrix can be calculated for a particular layer as:
# 
# $$G^l_{cd} = \frac{\sum_{ij} F^l_{ijc}(x)F^l_{ijd}(x)}{IJ}$$
# 
# This can be implemented concisely using the `tf.linalg.einsum` function:

# In[55]:


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


# ## **5. Extract style and content**
# Build a model that returns the style and content tensors.

# In[49]:


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


# When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`:

# In[56]:


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())


# ## **6. Running Gradient Descent**
# 
# With this style and content extractor, you can now implement the style transfer algorithm!
# 
# Do this by calculating the mean square error for your image's output relative to each target, then take the weighted sum of these losses.

# In[16]:


# Set your style and content target values
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# Define a `tf.Variable` to contain the image to optimize. To make this quick, initialize it with the content image (the `tf.Variable` must be the same shape as the content image):

# In[17]:


image = tf.Variable(content_image)


# Since this is a float image, define a function to keep the pixel values between 0 and 1:

# In[18]:


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Create an optimizer. The paper recommends LBFGS, but `Adam` works okay, too:

# In[19]:


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# To optimize this, use a weighted combination of the two losses to get the total loss:

# In[20]:


style_weight=1e-2
content_weight=1e4


# In[57]:


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# Use `tf.GradientTape` to update the image.
# 

# In[58]:


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# Now run a few steps to test:

# In[59]:


train_step(image)
train_step(image)
train_step(image)
tensor_to_image(image)


# **It works!**
# 
# Since it's working, perform a longer optimization:

# In[60]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))


# ## **7. Total variation loss - Reducing high frequency artifacts**
# 
# One downside to this basic implementation is that it produces a lot of high frequency artifacts. Decrease these using an explicit regularization term on the high frequency components of the image. In style transfer, this is often called the *total variation loss*:

# In[25]:


def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var


# ## Visually see the high frequency components

# In[26]:


x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")


# This shows how the high frequency components have increased.
# 
# Also, this high frequency component is basically an edge-detector. You can get similar output from the Sobel edge detector, for example:

# In[27]:


plt.figure(figsize=(14, 10))

sobel = tf.image.sobel_edges(content_image)
plt.subplot(1, 2, 1)
imshow(clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
plt.subplot(1, 2, 2)
imshow(clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")


# The regularization loss associated with this is the sum of the squares of the values:

# In[61]:


def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


# That demonstrated what it does. But there's no need to implement it yourself, TensorFlow includes a standard implementation:

# In[63]:


tf.image.total_variation(image).numpy()


# ## **8. Re-run the optimization**
# 
# Choose a weight for the `total_variation_loss`, we'll choose 30

# In[65]:


total_variation_weight=30


# Now include it in the `train_step` function:

# In[66]:


@tf.function()
def train_step(image, total_variation_weight):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# In[67]:


image = tf.Variable(content_image)


# And run the optimization:

# In[68]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100
total_variation_weight=30

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image, total_variation_weight)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))


# ### **Much better!!**
# #### **Save the result and show off your friends**

# In[70]:


file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download(file_name)


# In[ ]:




