#!/usr/bin/env python
# coding: utf-8

# 
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Generative Adversarial Networks (GANs) in Keras - Deep Convolutional GANs or DCGAN with MNIST**
# 
# ---
# 
# 
# In this lesson, we first learn to implement a basic **Generative Adversarial Networks (GANs) or DCGAN** with Keras using the MNIST dataset.
# 
# In this tutorial we demonstrate the original style-transfer algorithm. It optimizes the image content to a particular style. Modern approaches train a model to generate the stylized image directly (similar to [cyclegan](cyclegan.ipynb)). This approach is much faster (up to 1000x).
# 
# 1. Setup & load and prepare the dataset
# 2. Load and prepare the dataset
# 3. Define our Generator Model
# 4. Define our Discriminator Model
# 5. Define the loss and optimizers
# 6. Define the training loop
# 7. Training the model
# 
# 

# ## **A Recap on GANs**
# [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (GANs) are a really cool domain of Deep Learning where we produce new data that plausibly came from the dataset we train on.
# 
# To do this we use a *generator* network that learns to create images that look real, while a *discriminator* network learns to tell real images apart from fakes.
# 
# As training progresses the *generator* becomes better at making images that look real, while simultaenously the *discriminator* becomes better at telling them apart. 
# 
# The process reaches equilibrium when the *discriminator* can no longer distinguish real images from fakes. In this tutorial we demonstrate this process on the MNIST dataset. 
# 
# 
# ![A second diagram of a generator and discriminator](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202021-06-03%20at%209.54.37%402x.png)
# 
# 

# ## **1. Setup & load and prepare the dataset**

# In[1]:


# To generate GIFs
get_ipython().system('pip install imageio')
get_ipython().system('pip install git+https://github.com/tensorflow/docs')


# In[2]:


import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


# ## **2. Load and prepare the dataset**
# 
# We will train our GAN using the MNIST dataset, in the end we want our generator to be able to generate handwritten digits resembling those of the MNIST dataset.

# In[3]:


# Download MNIST
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshape and Normalize
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Set Batch and Buffer Size
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# ## **3. Define our Generator Model**
# 
# Both the generator and discriminator are defined using the Keras Sequential API.
# 
# The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.
# 

# In[5]:


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Now let's see how good our untrained generator is at making a 28x28 grayscale image (like MNIST).

# In[6]:


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# Not very good....

# ## **4. Define our Discriminator Model**
# 
# The discriminator is simply CNN-based image classifier.

# In[7]:


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Note the output is a single binary node

    return model


# Now lets use the untrained discriminator to classify the generated images as real or fake. 
# 
# The model will be trained to output positive values for real images, and negative values for fake images.

# In[9]:


discriminator = make_discriminator_model()
# Use the generated image produced by our untrained Generator
decision = discriminator(generated_image)
print(decision)


# ## **5. Define the loss and optimizers**
# 
# Define loss functions and optimizers for both models.
# 

# In[10]:


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# ### **Discriminator loss**
# 
# This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.

# In[11]:


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# ### **Generator loss**
# The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, compare the discriminators decisions on the generated images to an array of 1s.

# In[12]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator optimizers are different since you will train two networks separately.

# In[13]:


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# ### **Save checkpoints**
# This notebook also demonstrates how to save and restore models, which can be helpful in case a long running training task is interrupted.

# In[14]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## **6. Define the training loop**
# 

# In[15]:


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# The training loop begins with generator receiving a **random seed** as input. 
# 
# That seed is used to produce an image. 
# 
# The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
# 
# The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

# In[16]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# ### **Generate and save images**
# 

# In[18]:


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# In[19]:


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# ## **7. Training the model**
# Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).
# 
# At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.

# In[20]:


train(train_dataset, EPOCHS)


# Restore the latest checkpoint.

# In[21]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## **7. Output - Let's create a GIF**
# 

# In[22]:


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# In[23]:


display_image(EPOCHS)


# Use `imageio` to create an animated gif using the images saved during training.

# In[24]:


anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


# In[25]:


import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)


# ## Next steps
# 

# This tutorial has shown the complete code necessary to write and train a GAN. As a next step, you might like to experiment with a different dataset, for example the Large-scale Celeb Faces Attributes (CelebA) dataset [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset). To learn more about GANs see the [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).
# 
