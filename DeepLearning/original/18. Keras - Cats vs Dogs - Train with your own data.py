#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Keras Cats vs Dogs - Training using your own data**
# 
# ---
# 
# In this lesson, we learn how to setup data generators to load our own dataset and train a classifier using Keras. 
# 1. Download and Explore our data
# 2. Create a simple CNN
# 3. Create our Data Generators
# 4. Train our model
# 5. Test some inferences 
# 6. Checkpoints

# In[5]:


# import our packages

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# ## **1. Download and Explore our Data**

# In[6]:


get_ipython().system('gdown --id 1Dvw0UpvItjig0JbnzbTgYKB-ibMrXdxk')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# In[2]:


# Define our images sizes
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# ### **Loading our data and it's labels into a dataframe**
# 
# There are many ways we can do this, this way is relatively simple to follow.
# 
# `dog.1034234.jpg`

# In[7]:


filenames = os.listdir("./train")

categories = []

for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'class': categories
})
df.head()


# ### **Check the counts in each class**

# In[ ]:


df['class'].value_counts().plot.bar()


# #### **View a sample image**

# In[9]:


sample = random.choice(filenames)
image = load_img("./train/" + sample)
plt.imshow(image)


# ## **2. Create our Model**

# In[11]:


from keras.models import Sequential
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# ## **3. Create our Data Generators**

# In[12]:


df["class"] = df["class"].replace({0: 'cat', 1: 'dog'}) 
df.head()


# #### **Split our dataset using train_test_split**

# In[13]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=7)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[15]:


train_df.head()


# In[17]:


validate_df.head()


# ### **Create our Training Data Generator**

# In[18]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./train/", 
    x_col = 'filename',
    y_col = 'class',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size
)


# ### **Create our Validation Data Generator**

# In[19]:


validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./train/", 
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# #### **Creating an example Data Generator to load just a single image**

# In[20]:


example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "./train/", 
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# #### **Preview that image**

# In[21]:


plt.figure(figsize=(6, 6))

for X_batch, Y_batch in example_generator:
    image = X_batch[0]
    plt.imshow(image)
    break

plt.tight_layout()
plt.show()


# ## **4. Start Training Our Model**

# In[22]:


epochs = 10

history = model.fit(
    train_generator, 
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = 5000//batch_size,
    steps_per_epoch = 20000//batch_size,
)


# In[ ]:


# Save our model
model.save_weights("cats_vs_dogs_10_epochs.h5")


# In[ ]:


# Viewour for performance plots

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# #### **Get the predictions for our validation images**

# In[ ]:


# We run our prediction on all images in the validation set
predict = model.predict_generator(validation_generator, steps = np.ceil(5000/batch_size))


# #### **Add it to our dataframe for eacy viewing**

# In[ ]:


validate_df['predicted'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
validate_df['predicted'] = validate_df['predicted'].replace(label_map)
validate_df


# ## **Inference on a batch of images from our validation dataset**

# In[ ]:


sample_test = validate_df.head(18)
sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['predicted']
    img = load_img("./train/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()
plt.show()


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


checkpoint = ModelCheckpoint("MNIST_Checkpoint.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 5, #Number of epochs we wait before stopping 
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped


# ### **Another useful callback is Reducing our learning Rate on Plateau**
# 
# We can avoid having our oscillate around the global minimum by attempting to reduce the Learn Rate by a certain fact. If no improvement is seen in our monitored metric (val_loss typically), we wait a certain number of epochs (patience) then this callback reduces the learning rate by a factor

# In[ ]:


from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)


# In[ ]:


# we put our call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]


# In[ ]:


epochs = 10

history = model.fit(
    train_generator, 
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = 5000//batch_size,
    steps_per_epoch = 20000//batch_size,
)


# In[ ]:


while True:
  pass


# In[ ]:




