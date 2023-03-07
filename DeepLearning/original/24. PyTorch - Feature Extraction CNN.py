#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **PyTorch Cats vs Dogs - Feature Extraction**
# 
# ---
# 
# In this lesson, we learn how to use a pretrained network as a feature extractor. We'll then use those feautres as the input for our Logistic Regression Clasifier.
# 1. Load our pretrained VGG16 Model
# 2. Download our data and setup our transformations
# 3. Extract our Features using VGG16
# 4. Train a LR Classifier using those features
# 5. Run some inferences on our Test Data
# ---
# ### **You will need to use High-RAM and GPU (for speed increase).**
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.55.52%20pm.png)
# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/Screenshot%202021-05-17%20at%207.57.25%20pm.png)
# 
# ---

# ## **1. Download our Pre-trained Models (VGG16)**

# In[ ]:


import torch
import os
import tqdm
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchsummary import summary 
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
model = model.to(device)

summary(model, input_size = (3,224,224))


# ### **Remove the top Dense Fully Connected Layers**

# In[ ]:


# remove last fully-connected layer
new_classifier = nn.Sequential(*list(model.classifier.children())[:-7])
model.classifier = new_classifier


# `python
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)`

# In[ ]:


summary(model, input_size = (3,224,224))


# ## **2. Download our data and setup our Transformers**

# In[ ]:


get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/dogs-vs-cats.zip')
get_ipython().system('unzip -q gatos_perros.zip')
get_ipython().system('unzip -q train.zip')
get_ipython().system('unzip -q test1.zip')


# In[ ]:


# Set directory paths for our files
train_dir = './train'
test_dir = './test1'

# Get files in our directories
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

print(f'Number of images in {train_dir} is {len(train_files)}')
print(f'Number of images in {test_dir} is {len(test_files)}')

transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])

class Dataset():
    def __init__(self, filelist, filepath, transform = None):
        self.filelist = filelist
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return int(len(self.filelist))

    def __getitem__(self, index):
        imgpath = os.path.join(self.filepath, self.filelist[index])
        img = Image.open(imgpath)

        if "dog" in imgpath:
            label = 1
        else:
            label = 0 

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

# Create our train and test dataset objects
train = Dataset(train_files, train_dir, transformations)
val = Dataset(test_files, test_dir, transformations)

# Create our dataloaders
train_dataset = torch.utils.data.DataLoader(dataset = train, batch_size = 32, shuffle=True)
val_dataset = torch.utils.data.DataLoader(dataset = val, batch_size = 32, shuffle=True)


# ## **3. Extract our Features using VGG16**

# In[ ]:


image_names = os.listdir("./train")
image_paths = ["./train/"+ x for x in image_names]


# In[ ]:


model.eval() 
model = model.cuda()

with torch.no_grad():
    features = None
    image_labels = None

    # loop over each batch and pass our input tensors to hte model
    for data, label in tqdm.tqdm(train_dataset):
        x = data.cuda()
        output = model(x)
        
        if features is not None:
            # Concatenates the given sequence of tensors in the given dimension.
            # cat needs at least two tensors so we only start to cat after the first loop
            features = torch.cat((features, output), 0)
            image_labels = torch.cat((image_labels, label), 0)
        else:
            features = output
            image_labels = label

    # reshape our tensor to 25000 x 25088 
    features = features.view(features.size(0), -1)


# In[ ]:


# Check that we have features for all 25000 images
features.size(0)


# In[ ]:


# Check that we have labels for all 25000 images
image_labels.shape


# In[ ]:


# Check the shape to ensure our features are a flattened 512*7*7 array
features.shape


# ## **4. Train a LR Classifier using those features**

# In[ ]:


# Convert our tensors to numpy arrays
features_np = features.cpu().numpy()
image_labels_np = image_labels.cpu().numpy()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split our model into a test and training dataset to train our LR classifier
X_train, X_test, y_train, y_test = train_test_split(features_np, image_labels_np, test_size=0.2, random_state = 7)

glm = LogisticRegression(C=0.1)
glm.fit(X_train,y_train)


# In[ ]:


# Get Accruacy
accuracy = glm.score(X_test, y_test)
print(f'Accuracy on validation set using Logistic Regression: {accuracy*100}%')


# ## **5. Run some inferences on our Test Data**

# In[ ]:


image_names_test = os.listdir("./test1")
image_paths_test = ["./test1/"+ x for x in image_names_test]


# In[ ]:


from torch.autograd import Variable

imsize = 224

loader = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


# In[ ]:


import random 

test_sample = random.sample(image_paths_test, 12)
model.eval() 

def test_img():
    result_lst = []
    for path in test_sample:
      image = image_loader(loader, path)
      output = model(image.to(device))
      output = output.cpu().detach().numpy() 
      result = glm.predict(output)
      result = 'dog' if float(result) >0.5 else 'cat'
      result_lst.append(result)
    return result_lst


# In[ ]:


# get test predictions from all models
pred_results = test_img()
pred_results


# In[ ]:


import cv2

plt.figure(figsize=(15, 15))

for i in range(0, 12):
    plt.subplot(4, 3, i+1)
    result = pred_results[i]
    img = test_sample[i]
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.text(72, 248, f'Feature Extractor CNN: {result}', color='lightgreen',fontsize= 12, bbox=dict(facecolor='black', alpha=0.9))
    plt.imshow(image)

plt.tight_layout()
plt.show()


# In[ ]:




