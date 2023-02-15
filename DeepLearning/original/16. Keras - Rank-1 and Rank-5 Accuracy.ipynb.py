#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Using Pre-trained Models in Keras to get Rank-1 and Rank-5 Accuracy**
# 1. We'll first load the pre-trained ImageNet model MobileNetV2
# 2. We'll get the top 5 classes from a single image inference
# 3. Next we'll construct a function to give us the rank-N Accuracy using a few test images
# 
# ---
# 

# In[ ]:


# Load our pre-trained MobileNetV2 Model

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

model = MobileNetV2(weights='imagenet')
model.summary()


# In[ ]:


# Get the imageNet Class label names and test images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')
get_ipython().system('rm -rf ./images/class1/.DS_Store')


# In[ ]:


import cv2
from os import listdir
from os.path import isfile, join

# Get images located in ./images folder    
mypath = "./images/class1/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_names


# In[ ]:


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,16))
all_top_classes = []

# Loop through images run them through our classifer
for (i,file) in enumerate(file_names):

    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    #load image using opencv
    img2 = cv2.imread(mypath+file)
    #imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC) 
    
    # Get Predictions
    preds = model.predict(x)
    preditions = decode_predictions(preds, top=10)[0]
    all_top_classes.append([x[1] for x in preditions])
    # Plot image
    sub = fig.add_subplot(len(file_names),1, i+1)
    sub.set_title(f'Predicted {str(preditions)}')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.show()


# In[ ]:


preditions


# In[ ]:


all_top_classes


# In[ ]:


# Create our ground truth labels
ground_truth = ['basketball',
                'German shepherd',
                'limousine, limo',
                'spider_web',
                'burrito',
                'beer_glass',
                'doormat',
                'Christmas_stocking',
                'collie']


# In[ ]:


def getScore(all_top_classes, ground_truth, N):
  # Calcuate rank-N score
  in_labels = 0
  for (i,labels) in enumerate(all_top_classes):
    if ground_truth[i] in labels[:N]:
      in_labels += 1
  return f'Rank-{N} Accuracy = {in_labels/len(all_top_classes)*100:.2f}%'


# ## **Get Rank-5 Accuracy**

# In[ ]:


getScore(all_top_classes, ground_truth, 5)


# ## **Get Rank-1 Accuracy**

# In[ ]:


getScore(all_top_classes, ground_truth, 1)


# In[ ]:


getScore(all_top_classes, ground_truth, 10)


# In[ ]:




