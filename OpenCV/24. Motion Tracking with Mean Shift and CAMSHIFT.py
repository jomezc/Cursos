#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Motion Tracking with Mean Shift and CAMSHIFT.**
# 
# ####**In this lesson we'll learn two Object Tracking Algorithms:**
# 1. How to use the Mean Shift Algorithm in OpenCV
# 2. Use CAMSHIFT in OpenCV

# In[1]:


# Our Setup, Import Libaries, Create our Imshow Function and Download our Images
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
      
get_ipython().system('wget https://github.com/makelove/OpenCV-Python-Tutorial/raw/master/data/slow.flv')


# ## **Meanshif Object Tracking** 
# 
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_basics.jpg)
# 
# The intuition behind the meanshift is simple. Consider you have a set of points. (It can be a pixel distribution like histogram backprojection). You are given a small window ( may be a circle) and you have to move that window to the area of maximum pixel density (or maximum number of points). It is illustrated in the simple image given below:
# 
# ![](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/meanshift_face.gif)
# 
# Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region until convergence. Every shift is defined by a mean shift vector. The mean shift vector always points toward the direction of the maximum increase in the density. 
# ![](https://upload.wikimedia.org/wikipedia/commons/b/bd/Meanshiftred.gif)
# 
# Read Paper Here - https://ieeexplore.ieee.org/document/732882
# 
# Animation Source - https://fr.wikipedia.org/wiki/Camshift

# In[2]:


cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('car_tracking_mean_shift.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255),2)
        out.write(img2)
        #imshow('Tracking', img2)

    else:
        break

cap.release()
out.release()


# In[3]:


get_ipython().system('ffmpeg -i /content/car_tracking_mean_shift.avi car_tracking_mean_shift.mp4 -y')


# In[4]:


from IPython.display import HTML
from base64 import b64encode

mp4 = open('car_tracking_mean_shift.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# In[5]:


HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# ## **Camshift in OpenCV** 
# It is almost same as meanshift, but it returns a rotated rectangle (that is our result) and box parameters (used to be passed as search window in next iteration). 
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/8/86/CamshiftStillImage.gif)
# 
# Read Paper Here - https://ieeexplore.ieee.org/document/732882
# 
# Animation Source - https://fr.wikipedia.org/wiki/Camshift

# In[6]:


cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# Get the height and width of the frame (required to be an interger)
width = int(cap.get(3)) 
height = int(cap.get(4))

# Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
out = cv2.VideoWriter('car_tracking_cam_shift.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        out.write(img2)
        #imshow('img2',img2)

    else:
        break

cap.release()
out.release()


# In[8]:


get_ipython().system('ffmpeg -i /content/car_tracking_cam_shift.avi car_tracking_cam_shift.mp4 -y')


# In[9]:


from IPython.display import HTML
from base64 import b64encode

mp4 = open('car_tracking_cam_shift.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# In[11]:


HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

