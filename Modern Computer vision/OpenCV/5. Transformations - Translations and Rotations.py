#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# 
# # **Transformations - Translations and Rotations**
# 
# In this lesson we'll learn to:
# 1. Perform Image Translations
# 2. Rotations with getRotationMatrix2D
# 3. Rotations with Transpose
# 4. Flipping Images

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


# ### **Translations**
# 
# This an affine transform that simply shifts the position of an image. (left or right).
# 
# We use cv2.warpAffine to implement these transformations.
# 
# ```cv2.warpAffine(image, T, (width, height))```
# 
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/warp.png)

# In[ ]:


# Load our image
image = cv2.imread('images/Volleyball.jpeg')
imshow("Original", image)

# Store height and width of the image
height, width = image.shape[:2]

# We shift it by quarter of the height and width
quarter_height, quarter_width = height/4, width/4

# Our Translation
#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T is our translation matrix
T = np.float32([[1, 0, quarter_width], [0, 1,quarter_height]])

# We use warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image, T, (width, height))
imshow("Translated", img_translation)


# In[ ]:


# What does T look like
print(T)

print(height, width )


# ### **Rotations**
# 
# ```cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle of rotation, scale)```
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/rotation.png)
# 

# In[ ]:


# Load our image
image = cv2.imread('images/Volleyball.jpeg')
height, width = image.shape[:2]

# Divide by two to rototate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

# Input our image, the rotation matrix and our desired final width and height
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 1", rotated_image)


# In[ ]:


# Divide by two to rototate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)
print(rotation_matrix)
# Input our image, the rotation matrix and our desired final width and height
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 0.5", rotated_image)


# Notice all the black space surrounding the image.
# 
# We could now crop the image as we can calculate it's new size (we haven't learned cropping yet!).
# 
# ### **Rotations with cv2.transpose** (less flexible)
# 
# 
# 

# In[ ]:


rotated_image = cv2.transpose(image)
imshow("Original", image)
imshow("Rotated using Transpose", rotated_image)


# In[ ]:


rotated_image = cv2.transpose(image)
rotated_image = cv2.transpose(rotated_image)

imshow("Rotated using Transpose", rotated_image)


# In[ ]:


# Let's now to a horizontal flip.
flipped = cv2.flip(image, 1)
imshow("Horizontal Flip", flipped)


# In[ ]:




