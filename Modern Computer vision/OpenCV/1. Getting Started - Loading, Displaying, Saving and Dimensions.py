#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# 
# # **Getting Started**
# 
# Welcome to your first OpenCV Lesson. Here we'll learn to:
# 1. Import the OpenCV Model in Python
# 2. Load Images 
# 3. Display Images
# 4. Save Images
# 5. Getting the Image Dimensions
# 

# In[ ]:


# This is how we import OpenCV, we can't use OpenCV's functions without first doing this
import cv2


# In[ ]:


# Let's see what version we're running
print(cv2.__version__)


# ### **Dowloading Images**
# 
# If using Google Colab, we'll have to **upload our image**. 
# 
# Colab is a Jupyther Notebook environment that runs on the **cloud** using Google's servers. As such, any file we wish to use needs to be uploaded to their servers.

# In[ ]:


# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')


# ### **Loading Images**

# In[ ]:


# Load an image using 'imread' specifying the path to image
image = cv2.imread('./images/castara.jpeg')


# ### **Displaying Images**

# In[ ]:


from matplotlib import pyplot as plt

#Show the image with matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


# Let's create a simple function to make displaying our images simpler and easier

# In[ ]:


def imshow(title = "", image = None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# In[ ]:


# Let's test it out
imshow("Displaying Our First Image", image)


# ### **Saving Images**

# In[ ]:


# Simply use 'imwrite' specificing the file name and the image to be saved
cv2.imwrite('output.jpg', image)


# In[ ]:


# Or save it as a PNG (Portable Network Graphics) which is a Lossless bitmap image format
cv2.imwrite('output.png', image)


# ### **Displaying Image Dimensions**
# 
# Remember Images are arrays:
# 
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/array.png?token=ADLZD2HNEL33JAKTYRM3B5C7WMIV4)
# 
# 

# We need to use numpy to perform this operation. Don't worry, numpy will become one of your best friends if you're learning a Data Science and Computer Vision.

# In[ ]:


# Import numpy
import numpy as np

print(image.shape)


# In[ ]:


# To access a dimension, simply index it by using 0, 1 or 2.
image.shape[0]


# You can see the first dimension is the height and it's 960 pixels
# 
# The second dimension is the width which is 1280 pixels.
# 
# We can print them out nicely like this:

# In[ ]:


print('Height of Image: {} pixels'.format(int(image.shape[0])))
print('Width of Image: {} pixels'.format(int(image.shape[1])))
print('Depth of Image: {} colors components'.format(int(image.shape[2])))


# In[ ]:




