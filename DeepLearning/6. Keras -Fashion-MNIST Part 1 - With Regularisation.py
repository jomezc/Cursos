#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Regularisation in Keras - Part 2 - With Regularisation**
# ### **First we train a CNN on the Fashion-MNIST Dataset usng NO Regularisation Methods**
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/CleanShot%202020-12-02%20at%204.01.54%402x.png)
# ---
# 
# 
# 
# ---
# 
# 
# In this lesson, we first learn to create a **simple Convolutional Neural Network model** using Keras with TensorFlow 2.0 and train it to **classify images in the Fashion-MNIST Dataset**, without the use of any regularisation methods. 
# 1. Loading, Inspecting and Visualising our data
# 2. Preprocessing our data and defining our **Data Augmentation**
# 3. Build a Simple CNN with Regularisation 
#   - L2 Regularisation
#   - Data Augmentation
#   - Dropout
#   - BatchNorm
# 4. Train our CNN with Regularisation
# 
# 

# # **Loading, Inspecting and Visualising our data**

# In[1]:


# We load our data directly from the included datasets in tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist

# loads the Fashion-MNIST training and test dataset 
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()

# Our Class Names, when loading data from .datasets() our classes are integers
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# #### **Check to see if we're using the GPU**

# In[2]:


# Check to see if we're using the GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


# ### **Inspect our Data**

# In[3]:


# Display the number of samples in x_train, x_test, y_train, y_test
print("Initial shape or dimensions of x_train", str(x_train.shape))

# Print the number of samples in our data
print ("Number of samples in our training data: " + str(len(x_train)))
print ("Number of labels in our training data: " + str(len(y_train)))
print ("Number of samples in our test data: " + str(len(x_test)))
print ("Number of labels in our test data: " + str(len(y_test)))

# Print the image dimensions and no. of labels in our Training and Test Data
print("\n")
print ("Dimensions of x_train:" + str(x_train[0].shape))
print ("Labels in x_train:" + str(y_train.shape))
print("\n")
print ("Dimensions of x_test:" + str(x_test[0].shape))
print ("Labels in y_test:" + str(y_test.shape))


# ### **Visualizing some of our sample Data**
# 
# Let's plot 50 sample images.

# In[4]:


# Let's view the 50 first images of the MNIST training dataset
import matplotlib.pyplot as plt

# Create figure and change size
figure = plt.figure()
plt.figure(figsize=(16,10))

# Set how many images we wish to see
num_of_images = 50 

# iterate index from 1 to 51
for index in range(1, num_of_images + 1):
    class_names = classes[y_train[index]]
    plt.subplot(5, 10, index).set_title(f'{class_names}')
    plt.axis('off')
    plt.imshow(x_train[index], cmap='gray_r')


# # **2. Data Preprocessing using ImageDataGenerator**
# 
# First we reshape and change our data types as we had done previously.

# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras import backend as K

# Reshape our data to be in the format [number of samples, width, height, color_depth]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Change datatype to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# **We gather our image size, shape and Normalize our Test Data**
# 
# We will use the ImageDataGenerator to Normalize and provide Data Augmentations for our **Training Data**.

# In[6]:


# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# Normalize our data between 0 and 1
x_test /= 255.0


# ### **One Hot Encode our Labels**

# In[7]:


from tensorflow.keras.utils import to_categorical

# Now we one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix 
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# # **3. Building Our Model**
# 
# This is the same CNN we used previously for the MNIST classification project.

# In[8]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras import regularizers

L2 = 0.001

# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_regularizer = regularizers.l2(L2),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(L2)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu',kernel_regularizer = regularizers.l2(L2)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.SGD(0.001, momentum=0.9),
              metrics = ['accuracy'])

print(model.summary())


# # **Training Our Model**

# In[9]:


# Define Data Generator for Augmentation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# Here we fit the data generator to some sample data.
#train_datagen.fit(x_train)

batch_size = 32
epochs = 15

# Fit the model
# Notice we use train_datagen.flow, this takes data & label arrays, generates batches of augmented data.
history = model.fit(train_datagen.flow(x_train, y_train, batch_size = batch_size),
                              epochs = epochs,
                              validation_data = (x_test, y_test),
                              verbose = 1,
                              steps_per_epoch = x_train.shape[0] // batch_size)

# We obtain our accuracy score using the evalute function
# Score holds two values, our Test loss and Accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




