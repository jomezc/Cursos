#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Segmentación Semántica - U-Net y SegNet**
#
# ---
# Vamos a usar image-segmentation-keras para cargar modelos preentrenados, entrenarlos a través del aprendizaje de transferencia y ejecutar inferencias en imágenes.

# ## **Instalar el paquete**

# debe modificar la línea 77 en image-segmentation-keras/keras_segmentation/models/vgg16.py reemplazando:
# VGG_Weights_path = keras.utils.get_file( por VGG_Weights_path = tf.keras.utils.get_file(
# no se olvidó de agregar import tensorflow como tf y reinstalar la biblioteca image-segmentation-keras

# En 1]:


#!git clon https://github.com/divamgupta/image-segmentation-keras
get_ipython().system('git clone https://github.com/rajeevratan84/image-segmentation-keras.git')


# En 2]:


get_ipython().run_line_magic('cd', 'image-segmentation-keras')
get_ipython().system('python setup.py install')


# ### **Descargar el conjunto de datos**

# En 3]:


get_ipython().system('wget https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip && unzip -q dataset1.zip')


# ### **Inicializa el modelo**

# En[4]:


from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=50 ,  input_height=320, input_width=640)


# ### **Entrenar al modelo**

# En[5]:


model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5  )


# En[6]:


out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png")


# En[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt


# En[8]:


plt.imshow(out)


# En[9]:


from IPython.display import Image
Image('/tmp/out.png')


# ## **Pantalla con Leyenda**

# En[10]:


o = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    class_names = [ "Sky", "Building", "Pole","Road","Pavement","Tree","SignSymbol", "Fence", "Car","Pedestrian", "Bicyclist"])


# En[11]:


from IPython.display import Image
Image('/tmp/out.png')


# ## **Ahora vamos a cargar y entrenar un modelo SegNet**

# En[ ]:


from keras_segmentation.models.segnet import segnet

model = segnet(n_classes=50 ,  input_height=320, input_width=640)

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5)


# En[ ]:


from IPython.display import Image

out = model.predict_segmentation(
    inp = "dataset1/images_prepped_test/0016E5_07965.png",
    out_fname = "out.png")

Image('out.png')


# En[ ]:




