#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **DeepSORT - Tracking using YOLOv5**
# 
# ---
# 
# In this lesson, you'll learn how to integrate DeepSORT with any YOLOv5 model.
# 1. Download and Explore our data
# 2. Load our pretrained VGG16 Model
# 3. Extract our Features using VGG16
# 4. Train a LR Classifier using those features
# 5. Test some inferences 
# 
# ### **Change to GPU for performance increase.**
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)

# ## **Setup**

# In[ ]:


# backup repo link - https://github.com/rajeevratan84/Yolov5_DeepSort_Pytorch.git
get_ipython().system('git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git  # clone repo')
get_ipython().run_line_magic('cd', 'Yolov5_DeepSort_Pytorch')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install dependencies')
get_ipython().system('pip install youtube-dl')

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


# ## **Download Our test video and models**
# 
# The code to download from youtube and cut segments of the video is commented out below. Just highlight the code and press CTRL+L to uncomment block.
# 

# In[ ]:


# YOUTUBE_ID = 'uCj6glLYW5g'
# #!rm -rf youtube.mp4 https://www.youtube.com/watch?v=uCj6glLYW5g
# # download the youtube with the given ID
# ! youtube-dl -f 22 --output "youtube.%(ext)s" https://www.youtube.com/watch?v=$YOUTUBE_ID

# # cut the first 15 seconds
# ! ffmpeg -y -loglevel info -i youtube.mp4 -t 30 ped_track.mp4
# !y | ffmpeg -ss 00:00:00 -i youtube.mp4 -t 00:00:15 -c copy youtube_out.avi


# In[ ]:


get_ipython().run_line_magic('mkdir', 'yolov5/weights/')

# Get yolov5 model trained on the coco128 dataset
get_ipython().system('wget -nc https://github.com/rajeevratan84/ModernComputerVision/raw/main/yolov5s.pt -O /content/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5s.pt')
get_ipython().system('wget https://github.com/rajeevratan84/ModernComputerVision/raw/main/youtube_out.avi')

# get yolov5m model trained on the crowd-human dataset
#!wget -nc https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v.2.0/crowdhuman_yolov5m.pt -O /content/Yolov5_DeepSort_Pytorch/yolov5/weights/crowdhuman_yolov5m.pt
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/crowdhuman_yolov5m.pt')

# get the test video from the repo
#!wget -nc https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v.2.0/test.avi
# extract 3 seconds worth of video frames of it
#!y | ffmpeg -ss 00:00:00 -i test.avi -t 00:00:02 -c copy out.avi


# ## **Run inference on video**
# 
# Hence we chose to save it to file in this notebook. Locally you can use the ``--show-vid`` flag in order visualize the tracking in real-time

# In[ ]:


get_ipython().system('python track.py --yolo_model /content/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5s.pt --source youtube_out.avi --save-vid')


# ## **Show results**
# 
# Convert avi to mp4 and then we can play video within colab. This takes ~25 seconds.

# In[ ]:


get_ipython().system('ffmpeg -i /content/Yolov5_DeepSort_Pytorch/runs/track/exp/youtube_out.avi output.mp4 -y')


# Get the file content into data_url

# In[ ]:


from IPython.display import HTML
from base64 import b64encode
mp4 = open('output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


# Display it with HTML

# In[ ]:


HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# In[ ]:




