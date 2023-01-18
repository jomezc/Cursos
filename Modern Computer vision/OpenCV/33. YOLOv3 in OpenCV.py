#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# # **YOLOv3 in using cv2.dnn.readNetFrom()**
# 
# ####**In this lesson we'll learn how to load a pre-trained YOLOV3 Model and use OpenCV to run inferences over a few images**
# 

# In[ ]:


# import the necessary packages
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Download and unzip our images and YOLO files
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/YOLO.zip')
get_ipython().system('unzip -qq YOLO.zip')


# ## **YOLO Object Detection**
# 
# ![](https://opencv-tutorial.readthedocs.io/en/latest/_images/yolo1_net.png)
# 
# **Steps Invovled**
# 
# 1. Use Pretrained YOLOV3 weights (237MB)- https://pjreddie.com/media/files/yolov3.weights
# 2. Create our blob object which is our loaded model
# 3. Set the backend that runs the model

# In[ ]:


# Load the COCO class labels our YOLO model was trained on
labelsPath = "YOLO/yolo/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
 
# We now need to initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("Loading YOLO weights...") 

weights_path = "YOLO/yolo/yolov3.weights" 
cfg_path = "YOLO/yolo/yolov3.cfg"

# Create our blob object
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Set our backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Our YOLO Layers")
ln = net.getLayerNames()

# There are 254 Layers
print(len(ln), ln)


# The input to the network is a called blob object. 
# 
# The function ```cv.dnn.blobFromImage(img, scale, size, mean)``` transforms the image into a blob:
# 
# ```blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)```
# 
# **It has the following parameters:**
# 
# 1. the image to transform
# 2. the scale factor (1/255 to scale the pixel values to [0..1])
# 3. the size, here a 416x416 square image
# 4. the mean value (default=0)
# 5. the option swapBR=True (since OpenCV uses BGR)
# 
# **Note** A blob is a 4D numpy array object (images, channels, width, height). The image below shows the red channel of the blob. You notice the brightness of the red jacket in the background.
# 
# 

# In[ ]:


print("Starting Detections...")
# Get images located in ./images folder    
mypath = "YOLO/images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through images run them through our classifer
for file in file_names:
    # load our input image and grab its spatial dimensions
    image = cv2.imread(mypath+file)
    (H, W) = image.shape[:2]
 
    # we want only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Now we contruct our blob from our input images
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # We set our input to our image blob
    net.setInput(blob)
    # Then we run a forward pass through the network
    layerOutputs = net.forward(ln)

    # we initialize our lists for our detected bounding boxes, confidences, and classes
    boxes = []
    confidences = []
    IDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:

        # Loop over each detection
        for detection in output:
            # Obtain class ID and probality of detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # We keep only the most probably predictions
            if confidence > 0.75:
                # We scale the bounding box coordinates relative to the image
                # Note: YOLO actually returns the center (x, y) of the bounding
                # box followed by the width and height of the box
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Get the the top and left corner of the bounding box
                # Remember we alredy have the width and height
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Append our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)

    # Now we apply non-maxima suppression to reduce overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # We proceed once a detection has been found
    if len(idxs) > 0:
        # iterate over the indexes we are keeping
        for i in idxs.flatten():
            # Get the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw our bounding boxes and put our class label on the image
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output image
    imshow("YOLO Detections", image, size = 12)


# ## **NOTE:** **How to Perform non maximum suppression given boxes and corresponding scores.**
# 
# ```indices	=	cv.dnn.NMSBoxes(	bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]```
# 
# 
# 
# **Parameters**
# - bboxes	a set of bounding boxes to apply NMS.
# - scores	a set of corresponding confidences.
# - score_threshold	a threshold used to filter boxes by score.
# - nms_threshold	a threshold used in non maximum suppression.
# indices	the kept indices of bboxes after NMS.
# - eta	a coefficient in adaptive threshold formula: nms_thresholdi+1=eta⋅nms_thresholdi.
# - top_k	if >0, keep at most top_k picked indices.
# 
