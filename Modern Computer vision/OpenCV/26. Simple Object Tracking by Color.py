#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# # **Simple Object Tracking by Color**
# 
# ####**In this lesson we'll learn:**
# 1. How to use an HSV Color Filter to Create a Mask and then Track our Desired Object
# 

# In[4]:


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

get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/bmwm4.mp4')


# In[4]:


#Object Tracking
import cv2
import numpy as np

# Initalize camera
#cap = cv2.VideoCapture(0)

# define range of color in HSV
lower = np.array([20,50,90])
upper = np.array([40,255,255])

# Create empty points array
points = []

# Get default camera window size

# Load video stream, long clip
cap = cv2.VideoCapture('bmwm4.mp4')

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('bmwm4_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0
radius = 0

while True:
  
    # Capture webcame frame
    ret, frame = cap.read()
    if ret:
      hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # Threshold the HSV image to get only green colors
      mask = cv2.inRange(hsv_img, lower, upper)
      #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
      
      contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      # Create empty centre array to store centroid center of mass
      center =   int(Height/2), int(Width/2)

      if len(contours) > 0:
          
          # Get the largest contour and its center 
          c = max(contours, key=cv2.contourArea)
          (x, y), radius = cv2.minEnclosingCircle(c)
          M = cv2.moments(c)
          
          # Sometimes small contours of a point will cause a divison by zero error
          try:
              center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

          except:
              center =   int(Height/2), int(Width/2)

          # Allow only countors that have a larger than 25 pixel radius
          if radius > 25:
              
              # Draw cirlce and leave the last center creating a trail
              cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
              cv2.circle(frame, center, 5, (0, 255, 0), -1)
              
          # Log center points 
          points.append(center)
      
      # If radius large enough, we use 25 pixels
      if radius > 25:
          
          # loop over the set of tracked points
          for i in range(1, len(points)):
              try:
                  cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
              except:
                  pass
              
          # Make frame count zero
          frame_count = 0
              
      out.write(frame)
    else:
      break

# Release camera and close any open windows
cap.release()
out.release()


# In[ ]:


get_ipython().system('ffmpeg -i /content/bmwm4_output.avi bmwm4_output.mp4 -y')


# In[ ]:


from IPython.display import HTML
from base64 import b64encode

mp4 = open('bmwm4_output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# In[ ]:


HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# In[ ]:




