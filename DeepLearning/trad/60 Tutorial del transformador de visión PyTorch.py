#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Tutorial de Vision Transformer en PyTorch**
#
# Crédito: Hiroto Honda [página de inicio] (https://hirotomusiker.github.io/)
#
# Este cuaderno proporciona un recorrido sencillo del Transformador de visión. Esperamos que pueda comprender cómo funciona observando el flujo de datos real durante la inferencia.
#
# citas:
# - Artículo: Alexey Dosovitskiy et al., "Una imagen vale 16x16 palabras: transformadores para el reconocimiento de imágenes a escala",
# https://arxiv.org/abs/2010.11929
# - Implementación del modelo: este cuaderno carga (y está inspirado en) el increíble módulo de Ross Wightman (@wightmanr): https://github.com/rwightman/pytorch-image-models/tree/master/timm. Para obtener los códigos detallados, consulte el repositorio.
# - Los derechos de autor de las figuras y las imágenes de demostración pertenecen a Hiroto Honda.
#
#

# ## **Preparación**
# Excelente tutorial sobre timm - https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

# En 1]:


get_ipython().run_cell_magic('capture', '', '# Modelos de imagen de PyTorch\n!pip install timm\n')


# En 2]:


import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model


# ## **Preparar modelo y datos**

# En 3]:


# Carga tu modelo
model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# crear un modelo ViT: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
model = create_model(model_name, pretrained=True).to(device)


# En[4]:


# Definir transformaciones para prueba
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)


# En[5]:


get_ipython().run_cell_magic('capture', '', "# Etiquetas ImageNet\n!wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt\nimagennet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))\n\n# Imagen de demostración\n!wget https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/santorini.png?raw=true -O santorini.png\nimg = PIL.Image.open('santorini.png')\nimg_tensor = transforma(img).unsqueeze(0).to(dispositivo)\n")


# ## **Inferencia**

# En[6]:


# inferencia de extremo a extremo
output = model(img_tensor)
print(f"Inference Result: {imagenet_labels[int(torch.argmax(output))]}")
plt.imshow(img)


# # **Más profundo en el Transformador de Visión**
#
# ¡Veamos los detalles del Transformador de Visión!
#

# <img src='https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/vit_input.png?raw=true'>
#
# Figura 1. Tubería de inferencia de Vision Transformer.
# 1. Dividir imagen en parches
# La imagen de entrada se divide en vectores de 14 x 14 con una dimensión de 768 por Conv2d (k=16x16) con zancada=(16, 16).
# 2. Agregar incrustaciones de posición
# Los vectores de incrustación de posición aprendibles se agregan a los vectores de incrustación de parche y se alimentan al codificador del transformador.
# 3. Codificador de transformador
# Los vectores de incrustación son codificados por el codificador del transformador. Las dimensiones de los vectores de entrada y salida son las mismas. Los detalles del codificador se muestran en la Fig. 2.
# 4. MLP (Clasificación) Jefe
# La salida 0 del codificador se alimenta al cabezal MLP para la clasificación para generar los resultados de la clasificación final.
#

#### **1. Dividir imagen en parches**
#
# La imagen de entrada se divide en N parches (N = 14 x 14 para ViT-Base)
# y convertida a D=768 vectores incrustados por convolución 2D aprendible:
# ```
# Conv2d(3, 768, kernel_size=(16, 16), paso=(16, 16))
# ```

# En[7]:


patches = model.patch_embed(img_tensor)  # convolución de incrustación de parches, 14 * 14 = 196
print("Image tensor: ", img_tensor.shape)
print("Patch embeddings: ", patches.shape)


# En[8]:


# Esto NO es parte de la canalización.
# En realidad, la imagen se divide en incrustaciones de parches por Conv2d
# con zancada=(16, 16) como se muestra arriba.
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


#### **2. Agregar incrustaciones de posición**
# Para hacer que los parches reconozcan la posición, se agregan vectores de 'incrustación de posición' aprendibles a los vectores de incrustación de parches. Los vectores de incrustación de posición aprenden la distancia dentro de la imagen, por lo que los vecinos tienen una gran similitud.

# ### Visualización de incrustaciones de posición

# En[9]:


pos_embed = model.pos_embed
print(pos_embed.shape)


# En[10]:


# Visualice similitudes de incrustación de posiciones.
# Una celda muestra porque la similitud entre una incrustación y todas las demás incrustaciones.
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


# #### **Hacer entrada de transformador**
# Un token de clase aprendible se antepone a los vectores de incrustación de parches como el vector 0.
# 197 (1 + 14 x 14) vectores de incrustación de posición aprendibles se agregan a los vectores de incrustación de parches.

# En[11]:


transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
print("Transformer input: ", transformer_input.shape)


#### **3. Codificador de transformador**
# <img src='https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/transformer_encoder.png?raw=true'>
#
# Figura 2. Esquema detallado de Transformer Encoder.
# - N (=197) vectores integrados se alimentan a los codificadores de la serie L (=12).
# - Los vectores se dividen en consulta, clave y valor después de expandirse por una capa fc.
# - q, k y v se dividen en H (=12) y se envían a los cabezales de atención paralelos.
# - Las salidas de los cabezales de atención se concatenan para formar los vectores cuya forma es la misma que la entrada del codificador.
# - Los vectores pasan por una fc, una norma de capa y un bloque MLP que tiene dos capas fc.
#
# Vision Transformer emplea el codificador de transformador que se propuso en el [atención es todo lo que necesita] (https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
#
# Referencia de implementación:
#
# - [implementación de tensorflow](https://github.com/google-research/vision_transformer/blob/502746cb287a107f9911c061f9d9c2c0159c81cc/vit_jax/models.py#L62-L146)
# - [implementación de pytorch (timm)](https://github.com/rwightman/pytorch-image-models/blob/198f6ea0f3dae13f041f3ea5880dd79089b60d61/timm/models/vision_transformer.py#L79-L143)
#

# ### **Codificadores de transformadores en serie**

# En[12]:


print("Input tensor to Transformer (z0): ", transformer_input.shape)
x = transformer_input.clone()
for i, blk in enumerate(model.blocks):
    print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)
x = model.norm(x)
transformer_output = x[:, 0]
print("Output vector from Transformer (z12-0):", transformer_output.shape)


# ### **Cómo funciona la atención**
#
# En esta parte, vamos a ver cómo es la atención real.

# En[13]:


print("Transformer Multi-head Attention block:")
attention = model.blocks[0].attn
print(attention)
print("input of the transformer encoder:", transformer_input.shape)


# En[14]:


# fc capa para expandir la dimensión
transformer_input_expanded = attention.qkv(transformer_input)[0]
print("expanded to: ", transformer_input_expanded.shape)


# En[15]:


# Divida mkv en múltiples vectores q, k y v para la atención de múltiples cabezas
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (H=197, (chkv), G=12, D/G=shch)
print("split qkv : ", qkv.shape)
q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
print("transposed ks: ", kT.shape)


# En[16]:


# Matriz de atención
attention_matrix = q @ kT
print("attention matrix: ", attention_matrix.shape)
plt.imshow(attention_matrix[3].detach().cpu().numpy())


# En[17]:


# Visualizar matriz de atención
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
fig.add_axes()
img = np.asarray(img)
ax = fig.add_subplot(2, 4, 1)
ax.imshow(img)
for i in range(7):  # visualizar las filas 100 de matrices de atención en las cabezas 0-7
    attn_heatmap = attention_matrix[i, 100, 1:].reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(2, 4, i+2)
    ax.imshow(attn_heatmap)


### **4. MLP (Clasificación) Jefe**
# El vector de salida 0-th de los vectores de salida del transformador (correspondiente a la entrada del token de clase) se alimenta al cabezal MLP.
# El resultado de la clasificación de 1000 dimensiones es el resultado de toda la canalización.

# En[18]:


print("Classification head: ", model.head)
result = model.head(transformer_output)
result_label_id = int(torch.argmax(result))
plt.plot(result.detach().cpu().numpy()[0])
plt.title("Classification result")
plt.xlabel("class id")
print("Inference result : id = {}, label name = {}".format(
    result_label_id, imagenet_labels[result_label_id]))


# En[ ]:




