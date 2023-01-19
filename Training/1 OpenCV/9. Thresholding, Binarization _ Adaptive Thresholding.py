#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Thresholding, Binarization & Adaptive Thresholding**
# 
# ####**In this lesson we'll learn:**
# 1. Binarized Images
# 2. Thresholding Methods
# 3. Adaptive Thresholding
# 4. SkImage's Threshold Local

# In[1]:


# Our Setup, Import Libaries, Create our Imshow Function and Download our Images
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ### **Thresholding Methods** 
# 
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.57.55%20am.png)
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/Screenshot%202020-11-17%20at%2012.58.09%20am.png)
# 
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

# In[2]:


get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/scan.jpeg')


# In[5]:


# Load our image as greyscale 
image = cv2.imread('scan.jpeg',0)
imshow("Original", image)

# Values below 127 goes to 0 or black, everything above goes to 255 (white)
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('1 Threshold Binary @ 127', thresh1)

# Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
ret,thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
imshow('2 Threshold Binary Inverse @ 127', thresh2)

# Values above 127 are truncated (held) at 127 (the 255 argument is unused)
ret,thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
imshow('3 THRESH TRUNC @ 127', thresh3)

# Values below 127 go to 0, above 127 are unchanged  
ret,thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
imshow('4 THRESH TOZERO @ 127', thresh4)

# Reverse of the above, below 127 is unchanged, above 127 goes to 0
ret,thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
imshow('5 THRESH TOZERO INV @ 127', thresh5)


# #### **Adaptive Thresholding**
# 
# 1. ADAPTIVE_THRESH_MEAN_C
# 2. THRESH_OTSU
# 
# #### **cv2.adaptiveThreshold Parameters**
# 
# ``**cv2.adaptiveThreshold**(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst``
# 
# - **src** – Source 8-bit single-channel image.
# - **dst** – Destination image of the same size and the same type as src .
# - **maxValue** – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
# - **adaptiveMethod** – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
# - **thresholdType** – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
# - **blockSize** – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# - **C** – Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
# 

# In[4]:


image = cv2.imread('scan.jpeg',0)
imshow("Original", image)

# Values below 127 goes to 0 (black, everything above goes to 255 (white)
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('Threshold Binary', thresh1)

# It's good practice to blur images as it removes noise
#image = cv2.GaussianBlur(image, (3, 3), 0)

# Using adaptiveThreshold
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 
imshow("Adaptive Mean Thresholding", thresh) 

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Otsu's Thresholding", th2) 

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Guassian Otsu's Thresholding", th3) 


# ### **SkImage Threshold Local**
# 
# 
# ``threshold_local(image, block_size, offset=10)``
# 
# The threshold_local function, calculates thresholds in regions with a characteristic size ``block_size`` surrounding each pixel (i.e. local neighborhoods). Each threshold value is the weighted mean of the local neighborhood minus an ``offset`` value
# 
# 
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

# In[7]:


from skimage.filters import threshold_local

image = cv2.imread('scan.jpeg')

# We get the Value component from the HSV color space 
# then we apply adaptive thresholdingto 
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Apply the threshold operation 
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)


# ### **Why Bluring is Important before Thresholding?** 
# 
# ## **Answer - Noise**
# 
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/otsu.jpg)
# 
# https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
