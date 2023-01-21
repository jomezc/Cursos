#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Comparing Images**
# 
# ####**In this lesson we'll learn:**
# 1. Compare Images using Mean Squared Error (MSE)
# 2. UCompare Images using Structual Similarity

# In[ ]:


# Our Setup, Import Libaries, Create our Imshow Function and Download our Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

# Define our imshow function 
def imshow(title = "Image", image = None, size = 8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')

get_ipython().system('unzip -qq images.zip')


# #### **Mean Squared Error (MSE)**
# 
# The MSE between the two images is the sum of the squared difference between the two images. This can easily be implemented with numpy.
# 
# The lower the MSE the more similar the images are.

# In[ ]:


def mse(image1, image2):
	# Images must be of the same dimension
	error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
	error /= float(image1.shape[0] * image1.shape[1])

	return error


# #### **Let's get 3 images**
# 
# 1. Fireworks1
# 2. Fireworks1 with brightness enhanced
# 3. Fireworks2

# In[ ]:


fireworks1 = cv2.imread('images/fireworks.jpeg')
fireworks2 = cv2.imread('images/fireworks2.jpeg')

M = np.ones(fireworks1.shape, dtype = "uint8") * 100 
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)


# In[ ]:


def compare(image1, image2):
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  print('MSE = {:.2f}'.format(mse(image1, image2)))
  print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


# In[ ]:


# When they're the same
compare(fireworks1, fireworks1)


# In[ ]:


compare(fireworks1, fireworks2)


# In[ ]:


compare(fireworks1, fireworks1b)


# In[ ]:


compare(fireworks2, fireworks1b)

