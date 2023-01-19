#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Using Pre-trained Models in PyTorch to get Rank-1 and Rank-5 Accuracy**
# 1. We'll first load the pre-trained ImageNet model VGG16
# 2. We'll get the top 5 classes from a single image inference
# 3. Next we'll construct a function to give us the rank-N Accuracy using a few test images
# 
# ---
# 

# In[ ]:


# Load our pre-trained VGG16
import torchvision.models as models

model = models.vgg16(pretrained=True)


# ## **Normalisation**
# 
# All pre-trained models expect input images **normalized** in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
# 
# `normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])`

# In[ ]:


from torchvision import datasets, transforms, models

data_dir = '/images'

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),])


# **NOTE**
# 
# ```net.eval()``` is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:

# In[ ]:


model.eval()


# ### **Download our ImageNet Class Name and our Test Images**

# In[ ]:


# Get the imageNet Class label names and test images
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/imageNetclasses.json')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')
get_ipython().system('rm -rf ./images/class1/.DS_Store')


# ## **Import our modules**

# In[ ]:


import torch
import json
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('imageNetclasses.json') as f:
  class_names = json.load(f)


# # **Load and run a single image through our pre-trained model**

# In[ ]:


from PIL import Image
import numpy as np

image = Image.open('./images/class1/1539714414867.jpg')

# Convert to Tensor
image_tensor = test_transforms(image).float()
image_tensor = image_tensor.unsqueeze_(0)
input = Variable(image_tensor)
input = input.to(device)
output = model(input)
index = output.data.cpu().numpy().argmax()
name = class_names[str(index)]

# Plot image
fig=plt.figure(figsize=(8,8))
plt.axis('off')
plt.title(f'Predicted {name}')
plt.imshow(image)
plt.show()


# ## **Get our Class Probabilities**

# In[ ]:


import torch.nn.functional as nnf

prob = nnf.softmax(output, dim=1)

top_p, top_class = prob.topk(5, dim = 1)
print(top_p, top_class)


# In[ ]:


# Convert to numpy array
top_class_np = top_class.cpu().data.numpy()[0]
top_class_np


# ## **Create a class that gives us our class names**

# In[ ]:


def getClassNames(top_classes):
  top_classes = top_classes.cpu().data.numpy()[0]
  all_classes = []
  for top_class in top_classes:
    all_classes.append(class_names[str(top_class)])
  return all_classes


# In[ ]:


getClassNames(top_class)


# # **Construct our function to give us our Rank-N Accuracy**

# In[ ]:


from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,16))

def getRankN(model, directory, ground_truth, N, show_images = True):
  # Get image names in directory
  onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

  # We'll store the top-N class names here
  all_top_classes = []

  # Iterate through our test images
  for (i,image_filename) in enumerate(onlyfiles):
    image = Image.open(directory+image_filename)

    # Convert to Tensor
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    # Get our probabilties and top-N class names
    prob = nnf.softmax(output, dim=1)
    top_p, top_class = prob.topk(N, dim = 1)
    top_class_names = getClassNames(top_class)
    all_top_classes.append(top_class_names)

    if show_images:
      # Plot image
      sub = fig.add_subplot(len(onlyfiles),1, i+1)
      x = " ,".join(top_class_names)
      print(f'Top {N} Predicted Classes {x}')
      plt.axis('off')
      plt.imshow(image)
      plt.show()

  return getScore(all_top_classes, ground_truth, N)

def getScore(all_top_classes, ground_truth, N):
  # Calcuate rank-N score
  in_labels = 0
  for (i,labels) in enumerate(all_top_classes):
    if ground_truth[i] in labels:
      in_labels += 1
  return f'Rank-{N} Accuracy = {in_labels/len(all_top_classes)*100:.2f}%'


# In[ ]:


# Create our ground truth labels
ground_truth = ['basketball',
                'German shepherd, German shepherd dog, German police dog, alsatian',
                'limousine, limo',
                "spider web, spider's web",
                'burrito',
                'beer_glass',
                'doormat, welcome mat',
                'Christmas stocking',
               'collie']


# ## **Get Rank-5 Accuracy**

# In[ ]:


getRankN(model,'./images/class1/', ground_truth, N=5)


# ## **Get Rank-1 Accuracy**

# In[ ]:


getRankN(model,'./images/class1/', ground_truth, N=1)


# ## **Get Rank-10 Accuracy**

# In[ ]:


getRankN(model,'./images/class1/', ground_truth, N=10)


# In[ ]:




