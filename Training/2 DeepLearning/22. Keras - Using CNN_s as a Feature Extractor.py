#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Keras Cats vs Dogs - Feature Extraction**
# 
# ---
# 
# In this lesson, we learn how to use a pretrained network as a feature extractor. We'll then use those feautres as the input for our Logistic Regression Clasifier.
# 1. Download and Explore our data
# 2. Load our pretrained VGG16 Model
# 3. Extract our Features using VGG16
# 4. Train a LR Classifier using those features
# 5. Test some inferences 
# 
# ### **You will need to use High-RAM and GPU (for speed increase).**
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)

# ## **1. Download and Explore our data**

# In[1]:


from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications import VGG16, imagenet_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random
import tqdm
import os


# In[2]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/dogs-vs-cats.zip')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# ### **Loading our data and it's labels into a dataframe**
# 
# There are many ways we can do this, this way is relatively simple to follow.

# In[3]:


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


# ## **2. Load our pretrained VGG16 Model** 

# In[4]:


model = VGG16(weights="imagenet", include_top=False)


# In[5]:


model.summary()


# ## **What exactly are we doing?**
# 
# We're taking the output of the last CONV-POOL layer (see below). 
# 
# The output shape at this layer is **7 x 7 x 512**
# 
# ![feat_extraction](https://appliedmachinelearning.files.wordpress.com/2021/05/ef54e-vgg16.png?w=612&zoom=2)
# Image referenced from [here](https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/)

# ### **Store our Image Paths and Label names**

# ## **3. Extract our Features using VGG16**

# In[28]:


batch_size = 32
image_features = []
image_labels = []

# loop over each batch
for i in range(0, len(image_paths)//batch_size):
  # extract our batches
  batch_paths = image_paths[i:i + batch_size]
  batch_labels = labels[i:i + batch_size]
  batch_images = []

  # iterate over each image and extract our image features
  for image_path in batch_paths:
    # load images using Keras's load_img() and resize to 224 x 244
    image = load_img(image_path, target_size = (224, 224))
    image = img_to_array(image)

    # We expand the dimensions and then subtract the mean RGB pixel intensity of ImageNet
    # using the imagenet_utils.preprocess_input() function
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # append our image features to our batch list
    batch_images.append(image)

  # we take our batch of images and get in the right format with vstack
  batch_images = np.vstack(batch_images)

  # we then that batch and run it throuhg our predict function
  features = model.predict(batch_images, batch_size = batch_size)

  # we then take the output shape 7x7x512 and flatten it
  features = np.reshape(features,(-1, 7*7*512))

  # store our features and corresponding labels
  image_features.append(features)
  image_labels.append(batch_labels)


# In[29]:


# lets look at the image imageFeatures
print(image_features[0].shape)
image_features[0]


# In[30]:


image_labels


# ## **4. Train a LR Classifier using those features**
# 
# First let's store our extracted feature info in a format that can loaded directly by sklearn.

# In[ ]:


# take our list of batches and reduce the dimernsion so that it's now a list 25088 features x 25000 rows (25000 x 1 for our labels)
imageLabels_data =  [lb for label_batch in image_labels for lb in label_batch]
imageFeatures_data = [feature for feature_batch in image_features for feature in feature_batch]

# Convert to numpy arrays
image_labels_data = np.array(imageLabels_data)
image_features_data = np.array(imageFeatures_data)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

y = image_labels_data 

# Split our model into a test and training dataset to train our LR classifier
X_train, X_test, y_train, y_test = train_test_split(image_features_data, y, test_size=0.2, random_state = 7)

glm = LogisticRegression(C=0.1)
glm.fit(X_train,y_train)


# In[ ]:


# Get Accruacy on the 20% we split from our training dataset
accuracy = glm.score(X_test, y_test)
print(f'Accuracy on validation set using Logistic Regression: {accuracy*100}%')


# ## **5. Test some inferences**

# In[ ]:


image_names_test = os.listdir("./test1")
image_paths_test = ["./test1/"+ x for x in image_names_test]


# In[ ]:


import random 

test_sample = random.sample(image_paths_test, 12)

def test_img():
    result_lst = []
    for path in test_sample:
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        features = model.predict(image)
        features = np.reshape(features,(-1,7*7*512))
        result = glm.predict(features)
        result = 'dog' if float(result) >0.5 else 'cat'
        result_lst.append(result)
    return result_lst


# In[ ]:


# get test predictions from all models
pred_results = test_img()
pred_results


# In[ ]:


plt.figure(figsize=(15, 15))

for i in range(0, 12):
    plt.subplot(4, 3, i+1)
    result = pred_results[i]
    img = test_sample[i]
    image = load_img(img, target_size=(256,256))
    plt.text(72, 248, f'Feature Extractor CNN: {result}', color='lightgreen',fontsize= 12, bbox=dict(facecolor='black', alpha=0.9))
    plt.imshow(image)

plt.tight_layout()
plt.show()


# ## **How do we compare to Kaggle's top 10?**
# https://www.kaggle.com/c/dogs-vs-cats/leaderboard
# 
# We just got 98.34%, second place! Not too shabby :)
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%208.09.25%20pm.png)

# In[ ]:




