#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **DeepFace - Age, Gender, Expression, Headpose and Recognition**
# 
# ---
# 
# 
# In this lesson, we use the **DeepFace API for Age, Gender, Expression Facial and Recognition. We even use the headpose library to obtain head direction/tilt**. DeepFace is an easy to use python module that provides access to several Facial Detection and Recognition models. It's very simple to use to let's dive in.
# 
# 1. Install the necessary modules and download our files
# 2. Demonstrate facial landmarks
# 3. Obtain Age, Gender, Emotional Expression and Ethnicity using DeepFace
# 4. Perform Facial Similarity
# 5. Perform Facial Recognition
# 
# 
# **NOTE** Change to High-RAM setting.
# 

# ## **1. Install the necessary modules and download our files**

# In[ ]:


get_ipython().system('pip install deepface')
get_ipython().system('pip install dlib')


# #### **Define our imshow function**

# In[ ]:


# Some imports and our image viewing function
import dlib
import tarfile
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# In[ ]:


# Download facial landmarks
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/shape_predictor_68_face_landmarks.dat')


# In[ ]:


# Download our test images and a test pic
get_ipython().system('gdown --id 1RDw1BqsuZv4auJNv3iJ4ToIOnBq9WsVZ')
get_ipython().system('unzip -q face_recognition.zip')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/me.jpg')


# ## **2. Demonstrate facial landmarks**

# In[ ]:


from imutils import face_utils

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('me.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# Get faces 
rects = detector(gray, 0)

# For each detected face, find the landmark.
for (i, rect) in enumerate(rects):
    # Make the prediction and transfom it to numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw on our image, all the finded cordinate points (x,y) 
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Show the image
imshow("Output", image)


# ## **3. Obtain Age, Gender, Emotional Expression and Ethnicity using DeepFace**
# 
# **Download our models**

# In[ ]:


get_ipython().system('gdown --id 1Id32-d-nS9BooBLLkw1PQhvLWWAukCsq')
get_ipython().system('gdown --id 1txWignSWdELl8cWdZHYqIlSE2ZRjI8WI')
get_ipython().system('gdown --id 1d_tQRWjvQ5i4lZyUfFEfRj7LzXWXseBY')
get_ipython().system('gdown --id 1kWp2CVg_xTIFqdZAwfN_86A3grim9NyI')

get_ipython().system('mv facial_expression_model_weights.zip /root/.deepface/weights/facial_expression_model_weights.zip')
get_ipython().system('mv age_model_weights.h5 /root/.deepface/weights/age_model_weights.h5')
get_ipython().system('mv gender_model_weights.h5 /root/.deepface/weights/gender_model_weights.h5')
get_ipython().system('mv race_model_single_batch.zip /root/.deepface/weights/race_model_single_batch.zip')


# In[ ]:


from deepface import DeepFace

obj = DeepFace.analyze(img_path =  "./training_faces/Nidia_1.jpg", actions = ['age', 'gender', 'race', 'emotion'])
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])


# In[ ]:


from deepface import DeepFace
import pprint

img_path = "./training_faces/Nidia_1.jpg"
image = cv2.imread(img_path)

obj = DeepFace.analyze(img_path = img_path,
                       actions = ['age', 'gender', 'race', 'emotion'])
imshow("Face Analysis", image)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Create a simple function to display our results on the image**

# In[ ]:


import cv2

def drawFace(img_path, obj):
  image = cv2.imread(img_path)
  x = obj['region']['x'] 
  y = obj['region']['y'] 
  h = obj['region']['h'] 
  w = obj['region']['w'] 
  age = obj['age']
  gender = obj['gender']
  gender = 'F' if gender == 'Woman' else 'M'
  dominant_emotion = obj['dominant_emotion']
  dominant_race = obj['dominant_race']
  dominant_emotion = obj['dominant_emotion']
  description = f'{age}{gender} - {dominant_emotion}'
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(image, description, (x,y-10) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  cv2.putText(image, dominant_race, (x,y+h+30) , cv2.FONT_HERSHEY_PLAIN,2, (0,255,0), 3)
  imshow("Face Analysis", image)


# #### **Test on another image**

# In[ ]:


from deepface import DeepFace
import pprint

img_path = "training_faces/Nidia_4.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'])
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# #### **Change backends of face detection**

# In[ ]:


from deepface import DeepFace

# backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

img_path = "me.jpg"
image = cv2.imread(img_path)
obj = DeepFace.analyze(img_path = "me.jpg", actions = ['age', 'gender', 'race', 'emotion'], detector_backend = 'mtcnn')
drawFace(img_path, obj)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(obj)


# ## **4. Perform Facial Similarity**

# In[ ]:


result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg")
print("Is verified: ", result["verified"])
result


# ### **We can even use different Distance Metrics**

# In[ ]:


#metrics = ["cosine", "euclidean", "euclidean_l2"]

result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg", distance_metric = 'euclidean')
print("Is verified: ", result["verified"])
result


# In[ ]:


#metrics = ["cosine", "euclidean", "euclidean_l2"]

result  = DeepFace.verify("training_faces/Nidia_1.jpg", "training_faces/Nidia_3.jpg", distance_metric = 'euclidean_l2')
print("Is verified: ", result["verified"])
result


# ### **Download models as the existing DeepFace downloader has stopped working**

# In[ ]:


get_ipython().system('gdown --id 1OdJNKL85CCYStVi9XtJRpHhXo2FU6Gf1')
get_ipython().system('gdown --id 1GWIuvW3Vm3wMpGGEyTT7sU-c1cVWZIEc')
get_ipython().system('mv vgg_face_weights.h5 /root/.deepface/weights/vgg_face_weights.h5')
get_ipython().system('mv facenet_weights.h5 /root/.deepface/weights/facenet_weights.h5')


# ## **5. Perform Facial Recognition**

# In[ ]:


from deepface import DeepFace
import pandas as pd

df = DeepFace.find(img_path = "./training_faces/Nidia_1.jpg", db_path = './training_faces/', detector_backend = 'ssd')
df


# ## **We can even try a few different models**

# In[ ]:


from deepface import DeepFace
import pandas as pd

dfs = []
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

for model in models:
   df = DeepFace.find(img_path = "./training_faces/Nidia_1.jpg", db_path = './training_faces/', model_name = model,  detector_backend = 'ssd')
   df['model'] = model
   dfs.append(df)

pd.concat(dfs)


# In[ ]:


imshow('1', cv2.imread('./training_faces/Nidia_1.jpg'))
imshow('1', cv2.imread('./training_faces/Nidia_5.jpg'))


# Here's a great tutorial on building a MongoDB based system for facial recognition https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/
