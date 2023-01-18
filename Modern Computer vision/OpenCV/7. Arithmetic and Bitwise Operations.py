#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Arithmetic and Bitwise Operations**
# 
# #### **In this lesson we'll learn:**
# 1. Arithmetic Operations
# 2. Bitwise Operations

# In[ ]:


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

# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')


# ## **Arithmetic Operations**
# 
# These are simple operations that allow us to directly add or subract to the color intensity.
# 
# Calculates the per-element operation of two arrays. The overall effect is increasing or decreasing brightness.

# In[ ]:


# Adding comma zero in cv2.imread loads our image in as a grayscaled image
image = cv2.imread('images/liberty.jpeg', 0)
imshow("Grayscaled Image",image)
print(image)

# Create a matrix of ones, then multiply it by a scaler of 100 
# This gives a matrix with same dimesions of our image with all values being 100
M = np.ones(image.shape, dtype = "uint8") * 100 



# In[ ]:


print(M)


# #### **Increasing Brightness**

# In[ ]:


# We use this to add this matrix M, to our image
# Notice the increase in brightness
added = cv2.add(image, M)
imshow("Increasing Brightness", added)

# Now if we just added it, look what happens
added2 = image + M 
imshow("Simple Numpy Adding Results in Clipping", added2)


# #### **Decreasing Brightness**

# In[ ]:


# Likewise we can also subtract
# Notice the decrease in brightness
subtracted = cv2.subtract(image, M)
imshow("Subtracted", subtracted)

subtracted = image - M 
imshow("Subtracted 2", subtracted)


# ## **Bitwise Operations and Masking**
# 
# To demonstrate these operations let's create some simple images

# In[ ]:


# If you're wondering why only two dimensions, well this is a grayscale image, 

# Making a square
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
imshow("square", square)

# Making a ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
imshow("ellipse", ellipse)


# ### **Experimenting with some bitwise operations such as AND, OR, XOR and NOT**

# In[ ]:


# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)
imshow("AND", And)

# Shows where either square or ellipse is 
bitwiseOr = cv2.bitwise_or(square, ellipse)
imshow("bitwiseOr", bitwiseOr)

# Shows where either exist by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
imshow("bitwiseXor", bitwiseXor)

# Shows everything that isn't part of the square
bitwiseNot_sq = cv2.bitwise_not(square)
imshow("bitwiseNot_sq", bitwiseNot_sq)

# Notice the last operation inverts the image totally


# In[ ]:




