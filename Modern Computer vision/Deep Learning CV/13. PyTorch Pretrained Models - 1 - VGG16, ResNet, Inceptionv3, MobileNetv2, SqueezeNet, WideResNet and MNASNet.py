#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Using Pre-trained Models in PyTorch**
# ### **We'll load the weights of advanced pretrained models such as:**
# 
# ---
# 
# 
# 1. VGG16
# 2. ResNet
# 3. Inception v3
# 4. MobileNet v2
# 5. SqueezeNet
# 6. Wide ResNet
# 7. MNASNet
# 
# **View all the Models Available in PyTorch here** - https://pytorch.org/vision/main/models.html

# # **1. Loading VGG16**

# In[ ]:


import torchvision.models as models

model = models.vgg16(pretrained=True)


# ### **Let's take a look at its layers**

# In[ ]:


# Look at the model's layers
model


# ### **Check the number of parameters**

# In[ ]:


from torchsummary import summary 

summary(model, input_size = (3,224,224))


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
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])


# **NOTE**
# 
# ```net.eval()``` is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:

# In[ ]:


model.eval()


# ## **Run some inferences**

# In[ ]:


# Get the imageNet Class label names
get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/imageNetclasses.json')


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


def predict_image(images, class_names):
    to_pil = transforms.ToPILImage()
    fig=plt.figure(figsize=(16,16))

    for (i,image) in enumerate(images):
      # Convert to image and tensor
      image = to_pil(image)
      image_tensor = test_transforms(image).float()
      image_tensor = image_tensor.unsqueeze_(0)
      input = Variable(image_tensor)
      input = input.to(device)
      output = model(input)
      index = output.data.cpu().numpy().argmax()
      name = class_names[str(index)]
      
      # Plot image
      sub = fig.add_subplot(len(images),1, i+1)
      sub.set_title(f'Predicted {str(name)}')
      plt.axis('off')
      plt.imshow(image)
    plt.show()

def get_images(directory='./images'):
    data = datasets.ImageFolder(directory, transform=test_transforms)
    num_images = len(data)
    loader = torch.utils.data.DataLoader(data, batch_size=num_images)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images


# In[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/imagesDLCV.zip')
get_ipython().system('unzip imagesDLCV.zip')


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **2. Loading ResNet**

# In[ ]:


import torchvision.models as models

model = models.resnet18(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **3. Loading Inception**

# In[ ]:


import torchvision.models as models

model = models.inception_v3(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

# Note difference input sinze is expected with Inception
summary(model, input_size = (3,299,299))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **4. Loading MobileNet**
# 

# In[ ]:


import torchvision.models as models

model = models.mobilenet_v2(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **5. Loading SqueezeNet**
# 

# In[ ]:


import torchvision.models as models

model = models.squeezenet1_0(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **6. Loading Wide ResNet**
# 

# In[ ]:


import torchvision.models as models

model = models.wide_resnet50_2(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# # **7. Loading Wide MNASNet**
# 

# In[ ]:


import torchvision.models as models

model = models.mnasnet1_0(pretrained=True)


# In[ ]:


# Show Model Parameters
from torchsummary import summary 

summary(model, input_size = (3,224,224))


# In[ ]:


# Set to Eval and look at the model's layers
model.eval()


# In[ ]:


images = get_images('./images')
predict_image(images, class_names)


# In[ ]:




