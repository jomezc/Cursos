#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Vision Transformer Tutorial in PyTorch**
# 
# Credit - Hiroto Honda  [homepage](https://hirotomusiker.github.io/)  
# 
# This notebook provides the simple walkthrough of the Vision Transformer. We hope you will be able to understand how it works by looking at the actual data flow during inference.  
# 
# citiations:
# - Paper: Alexey Dosovitskiy et al., "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale", 
# https://arxiv.org/abs/2010.11929
# - Model Implementation: this notebook loads (and is inspired by) Ross Wightman (@wightmanr)'s amazing module: https://github.com/rwightman/pytorch-image-models/tree/master/timm . For the detailed codes, please refer to the repo.
# - The copyright of figures and demo images belongs to Hiroto Honda.
# 
# 

# ## **Preparation**
# Excellent tutorial on timm - https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

# In[1]:


get_ipython().run_cell_magic('capture', '', '# PyTorch Image Models\n!pip install timm\n')


# In[2]:


import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model


# ## **Prepare Model and Data**

# In[3]:


# Load your model
model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
model = create_model(model_name, pretrained=True).to(device)


# In[4]:


# Define transforms for test
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)


# In[5]:


get_ipython().run_cell_magic('capture', '', "# ImageNet Labels\n!wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt\nimagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))\n\n# Demo Image\n!wget https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/santorini.png?raw=true -O santorini.png\nimg = PIL.Image.open('santorini.png')\nimg_tensor = transforms(img).unsqueeze(0).to(device)\n")


# ## **Inference**

# In[6]:


# end-to-end inference
output = model(img_tensor)
print(f"Inference Result: {imagenet_labels[int(torch.argmax(output))]}")
plt.imshow(img)


# # **Deeper into the Vision Transformer**
# 
# Let's look at the details of the Vision Transformer!
# 

# <img src='https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/vit_input.png?raw=true'>
# 
# Figure 1. Vision Transformer inference pipeline.  
# 1. Split Image into Patches  
# The input image is split into 14 x 14 vectors with dimension of 768 by Conv2d (k=16x16) with stride=(16, 16). 
# 2. Add Position Embeddings  
# Learnable position embedding vectors are added to the patch embedding vectors and fed to the transformer encoder. 
# 3. Transformer Encoder  
# The embedding vectors are encoded by the transformer encoder. The dimension of input and output vectors are the same. Details of the encoder are depicted in Fig. 2.
# 4. MLP (Classification) Head  
# The 0th output from the encoder is fed to the MLP head for classification to output the final classification results.
# 

# ### **1. Split Image into Patches**
# 
# The input image is split into N patches (N = 14 x 14 for ViT-Base)
# and converted to D=768 embedding vectors by learnable 2D convolution:
# ```
# Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
# ```

# In[7]:


patches = model.patch_embed(img_tensor)  # patch embedding convolution, 14 * 14 = 196
print("Image tensor: ", img_tensor.shape)
print("Patch embeddings: ", patches.shape)


# In[8]:


# This is NOT a part of the pipeline.
# Actually the image is divided into patch embeddings by Conv2d 
# with stride=(16, 16) shown above.
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of Patches", fontsize=24)
fig.add_axes()
img = np.asarray(img)
for i in range(0, 196):
    x = i % 14
    y = i // 14
    patch = img[y*16:(y+1)*16, x*16:(x+1)*16]
    ax = fig.add_subplot(14, 14, i+1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)


# ### **2. Add Position Embeddings**
# To make patches position-aware, learnable 'position embedding' vectors are added to the patch embedding vectors. The position embedding vectors learn distance within the image thus neighboring ones have high similarity.

# ### Visualization of position embeddings

# In[9]:


pos_embed = model.pos_embed
print(pos_embed.shape)


# In[10]:


# Visualize position embedding similarities.
# One cell shows cos similarity between an embedding and all the other embeddings.
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of position embedding similarities", fontsize=24)
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(14, 14, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)


# #### **Make Transformer Input**
# A learnable class token is prepended to the patch embedding vectors as the 0th vector.  
# 197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors.

# In[11]:


transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
print("Transformer input: ", transformer_input.shape)


# ### **3. Transformer Encoder**
# <img src='https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/transformer_encoder.png?raw=true'>
# 
# Figure 2. Detailed schematic of Transformer Encoder. 
# - N (=197) embedded vectors are fed to the L (=12) series encoders. 
# - The vectors are divided into query, key and value after expanded by an fc layer. 
# - q, k and v are further divided into H (=12) and fed to the parallel attention heads. 
# - Outputs from attention heads are concatenated to form the vectors whose shape is the same as the encoder input.
# - The vectors go through an fc, a layer norm and an MLP block that has two fc layers.
# 
# The Vision Transformer employs the Transformer Encoder that was proposed in the [attention is all you need paper](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 
# 
# Implementation Reference: 
# 
# - [tensorflow implementation](https://github.com/google-research/vision_transformer/blob/502746cb287a107f9911c061f9d9c2c0159c81cc/vit_jax/models.py#L62-L146)
# - [pytorch implementation (timm)](https://github.com/rwightman/pytorch-image-models/blob/198f6ea0f3dae13f041f3ea5880dd79089b60d61/timm/models/vision_transformer.py#L79-L143)
# 

# ### **Series Transformer Encoders**

# In[12]:


print("Input tensor to Transformer (z0): ", transformer_input.shape)
x = transformer_input.clone()
for i, blk in enumerate(model.blocks):
    print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)
x = model.norm(x)
transformer_output = x[:, 0]
print("Output vector from Transformer (z12-0):", transformer_output.shape)


# ### **How Attention Works**
# 
# In this part, we are going to see what the actual attention looks like.

# In[13]:


print("Transformer Multi-head Attention block:")
attention = model.blocks[0].attn
print(attention)
print("input of the transformer encoder:", transformer_input.shape)


# In[14]:


# fc layer to expand the dimension
transformer_input_expanded = attention.qkv(transformer_input)[0]
print("expanded to: ", transformer_input_expanded.shape)


# In[15]:


# Split qkv into mulitple q, k, and v vectors for multi-head attention
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
print("split qkv : ", qkv.shape)
q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
print("transposed ks: ", kT.shape)


# In[16]:


# Attention Matrix
attention_matrix = q @ kT
print("attention matrix: ", attention_matrix.shape)
plt.imshow(attention_matrix[3].detach().cpu().numpy())


# In[17]:


# Visualize attention matrix
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
fig.add_axes()
img = np.asarray(img)
ax = fig.add_subplot(2, 4, 1)
ax.imshow(img)
for i in range(7):  # visualize the 100th rows of attention matrices in the 0-7th heads
    attn_heatmap = attention_matrix[i, 100, 1:].reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(2, 4, i+2)
    ax.imshow(attn_heatmap)


# ## **4. MLP (Classification) Head**
# The 0-th output vector from the transformer output vectors (corresponding to the class token input) is fed to the MLP head.  
# The 1000-dimension classification result is the output of the whole pipeline.

# In[18]:


print("Classification head: ", model.head)
result = model.head(transformer_output)
result_label_id = int(torch.argmax(result))
plt.plot(result.detach().cpu().numpy()[0])
plt.title("Classification result")
plt.xlabel("class id")
print("Inference result : id = {}, label name = {}".format(
    result_label_id, imagenet_labels[result_label_id]))


# In[ ]:




