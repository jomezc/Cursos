#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **ArcaneGAN**
# 
# An inference notebook for [ArcaneGAN v0.2](https://github.com/Sxela/ArcaneGAN/releases/tag/v0.1).
# 
# Made by [Alex Spirin](https://twitter.com/devdef)
# 
# If you like what I'm doing you can tip me [here](https://donationalerts.com/r/derplearning) or follow on [Patreon](https://www.patreon.com/sxela)
# 
# ![visitors](https://visitor-badge.glitch.me/badge?page_id=sxela_arcanegan)

# In[ ]:


#@title This colab is distributed under the MIT license
"""MIT License

Copyright (c) 2021 Alex Spirin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


# In[ ]:


#@title Install and download. Run once.
#release v0.2
get_ipython().system('wget https://github.com/Sxela/ArcaneGAN/releases/download/v0.1/ArcaneGANv0.1.jit')
get_ipython().system('wget https://github.com/Sxela/ArcaneGAN/releases/download/v0.2/ArcaneGANv0.2.jit')
get_ipython().system('pip -qq install facenet_pytorch')


# In[ ]:


#@title Define functions
#@markdown Select model version and run.
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch, PIL

from tqdm.notebook import tqdm

mtcnn = MTCNN(image_size=256, margin=80)

# simplest ye olde trustworthy MTCNN for face detection with landmarks
def detect(img):
 
        # Detect faces
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        # Select faces
        if not mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
            )
 
        return batch_boxes, batch_points

# my version of isOdd, should make a separate repo for it :D
def makeEven(_x):
  return _x if (_x % 2 == 0) else _x+1

# the actual scaler function
def scale(boxes, _img, max_res=1_500_000, target_face=256, fixed_ratio=0, max_upscale=2, VERBOSE=False):
 
    x, y = _img.size
 
    ratio = 2 #initial ratio
 
    #scale to desired face size
    if (boxes is not None):
      if len(boxes)>0:
        ratio = target_face/max(boxes[0][2:]-boxes[0][:2]); 
        ratio = min(ratio, max_upscale)
        if VERBOSE: print('up by', ratio)

    if fixed_ratio>0:
      if VERBOSE: print('fixed ratio')
      ratio = fixed_ratio
 
    x*=ratio
    y*=ratio
 
    #downscale to fit into max res 
    res = x*y
    if res > max_res:
      ratio = pow(res/max_res,1/2); 
      if VERBOSE: print(ratio)
      x=int(x/ratio)
      y=int(y/ratio)
 
    #make dimensions even, because usually NNs fail on uneven dimensions due skip connection size mismatch
    x = makeEven(int(x))
    y = makeEven(int(y))
    
    size = (x, y)

    return _img.resize(size)

""" 
    A useful scaler algorithm, based on face detection.
    Takes PIL.Image, returns a uniformly scaled PIL.Image
    boxes: a list of detected bboxes
    _img: PIL.Image
    max_res: maximum pixel area to fit into. Use to stay below the VRAM limits of your GPU.
    target_face: desired face size. Upscale or downscale the whole image to fit the detected face into that dimension.
    fixed_ratio: fixed scale. Ignores the face size, but doesn't ignore the max_res limit.
    max_upscale: maximum upscale ratio. Prevents from scaling images with tiny faces to a blurry mess.
"""

def scale_by_face_size(_img, max_res=1_500_000, target_face=256, fix_ratio=0, max_upscale=2, VERBOSE=False):
    boxes = None
    boxes, _ = detect(_img)
    if VERBOSE: print('boxes',boxes)
    img_resized = scale(boxes, _img, max_res, target_face, fix_ratio, max_upscale, VERBOSE)
    return img_resized


size = 256

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

t_stds = torch.tensor(stds).cuda().half()[:,None,None]
t_means = torch.tensor(means).cuda().half()[:,None,None]

def makeEven(_x):
  return int(_x) if (_x % 2 == 0) else int(_x+1)

img_transforms = transforms.Compose([                        
            transforms.ToTensor(),
            transforms.Normalize(means,stds)])
 
def tensor2im(var):
     return var.mul(t_stds).add(t_means).mul(255.).clamp(0,255).permute(1,2,0)

def proc_pil_img(input_image, model):
    transformed_image = img_transforms(input_image)[None,...].cuda().half()
            
    with torch.no_grad():
        result_image = model(transformed_image)[0]; print(result_image.shape)
        output_image = tensor2im(result_image)
        output_image = output_image.detach().cpu().numpy().astype('uint8')
        output_image = PIL.Image.fromarray(output_image)
    return output_image

#load model

version = '0.2' #@param ['0.1','0.2']

model_path = f'/content/ArcaneGANv{version}.jit' 
in_dir = '/content/in'
out_dir = f"/content/{model_path.split('/')[-1][:-4]}_out"

model = torch.jit.load(model_path).eval().cuda().half()

#setup colab interface

from google.colab import files
import ipywidgets as widgets
from IPython.display import clear_output 
from IPython.display import display
import os
from glob import glob

def reset(p):
  with output_reset:
    clear_output()
  clear_output()
  process()
 
button_reset = widgets.Button(description="Upload")
output_reset = widgets.Output()
button_reset.on_click(reset)

def fit(img,maxsize=512):
  maxdim = max(*img.size)
  if maxdim>maxsize:
    ratio = maxsize/maxdim
    x,y = img.size
    size = (int(x*ratio),int(y*ratio)) 
    img = img.resize(size)
  return img
 
def show_img(f, size=1024):
  display(fit(PIL.Image.open(f),size))

def process(upload=True):
  os.makedirs(in_dir, exist_ok=True)
  get_ipython().run_line_magic('cd', '{in_dir}/')
  get_ipython().system('rm -rf {out_dir}/*')
  os.makedirs(out_dir, exist_ok=True)
  in_files = sorted(glob(f'{in_dir}/*'))
  if (len(in_files)==0) | (upload):
    get_ipython().system('rm -rf {in_dir}/*')
    uploaded = files.upload()
    if len(uploaded.keys())<=0: 
      print('\nNo files were uploaded. Try again..\n')
      return

  

  print('\nPress the button and pick some photos to upload\n')
  
  in_files = sorted(glob(f'{in_dir}/*'))
  for img in tqdm(in_files):
    out = f"{out_dir}/{img.split('/')[-1].split('.')[0]}.jpg"
    im = PIL.Image.open(img)
    im = scale_by_face_size(im, target_face=300, max_res=1_500_000, max_upscale=2)
    res = proc_pil_img(im, model)
    res.save(out)

  out_zip = f"{out_dir}.zip"
  get_ipython().system('zip {out_zip} {out_dir}*')
    
  processed = sorted(glob(f'{out_dir}/*'))[:3]
  for f in processed: 
    show_img(f, 256)


# In[ ]:


#@title Click to upload files and run inference. Results will be saved and zipped.
process()


# In[ ]:




