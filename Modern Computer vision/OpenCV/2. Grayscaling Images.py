#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Grayscaling Images**
# 
# In this lesson we'll learn to:
# 1. Convert a color image to grayscale
# 2. See the change in dimensions between grayscale and color images
# 
# 

# ### **Dowloading Images**
# 
# If using Google Colab, we'll have to **upload our image**. 
# 
# Colab is a Jupyther Notebook environment that runs on the **cloud** using Google's servers. As such, any file we wish to use needs to be uploaded to their servers.

# In[1]:


import cv2
from matplotlib import pyplot as plt


# In[2]:


# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')


# In[3]:


# Load our input image
image = cv2.imread('./images/castara.jpeg')

imshow("Castara, Tobago", image)


# In[4]:


image.shape[:2] # (1200, 1920) (height, width)
def imshow(title = "", image = None, size = 10):
 
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
 
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

imshow("Castara, Tobago", image)


# In[ ]:


# We use cvtColor, to convert to grayscale
# It takes 2 arguments, the first being the input image
# The second being the color space conversion code 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imshow("Converted to Grayscale", gray_image)


# ### **Grayscale Image Dimensions**
# 
# Remember RGB color images have 3 dimensions, one for each primary color. Grayscale just has 1, which is the intensity of gray.
# 
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/gray.png)

# In[ ]:


image.shape


# In[ ]:


gray_image.shape

