#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **ArcanoGAN**
#
# Un cuaderno de inferencias para [ArcaneGAN v0.2](https://github.com/Sxela/ArcaneGAN/releases/tag/v0.1).
#
# Realizado por [Alex Spirin](https://twitter.com/devdef)
#
# Si te gusta lo que estoy haciendo, puedes darme una propina [aquí](https://donationalerts.com/r/derplearning) o seguirme en [Patreon](https://www.patreon.com/sxela)
#
# ![visitantes](https://visitor-badge.glitch.me/badge?page_id=sxela_arcanegan)

# En[ ]:


#@title Esta colaboración se distribuye bajo la licencia MIT
"""MI licencia
Copyright (c) 2021 Alex Spirin

Por la presente se concede permiso, sin cargo, a cualquier persona que obtenga una copia
de este software y los archivos de documentación asociados (el "Software"), para tratar
en el Software sin restricciones, incluidos, entre otros, los derechos
usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender
copias del Software, y para permitir a las personas a quienes se les
provisto para hacerlo, sujeto a las siguientes condiciones:

El aviso de derechos de autor anterior y este aviso de permiso se incluirán en todos
copias o partes sustanciales del Software.

EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O
IMPLÍCITO, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD,
IDONEIDAD PARA UN PROPÓSITO PARTICULAR Y NO VIOLACIÓN. EN NINGÚN CASO LA
LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES DE CUALQUIER RECLAMACIÓN, DAÑOS U OTROS
RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRA FORMA, DERIVADA DE,
FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL
SOFTWARE."""


# En[ ]:


#@title Instalar y descargar. Corre una vez.
#lanzamiento v0.2
get_ipython().system('wget https://github.com/Sxela/ArcaneGAN/releases/download/v0.1/ArcaneGANv0.1.jit')
get_ipython().system('wget https://github.com/Sxela/ArcaneGAN/releases/download/v0.2/ArcaneGANv0.2.jit')
get_ipython().system('pip -qq install facenet_pytorch')


# En[ ]:


#@title Definir funciones
#@markdown Seleccione la versión del modelo y ejecute.
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch, PIL

from tqdm.notebook import tqdm

mtcnn = MTCNN(image_size=256, margin=80)

# MTCNN más simple y confiable para la detección de rostros con puntos de referencia
def detect(img):
 
        # Detectar rostros
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        # Seleccionar caras
        if not mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
            )
 
        return batch_boxes, batch_points

# mi versión de isOdd, debería hacer un repositorio separado para ella :D
def makeEven(_x):
  return _x if (_x % 2 == 0) else _x+1

# la función de escalador real
def scale(boxes, _img, max_res=1_500_000, target_face=256, fixed_ratio=0, max_upscale=2, VERBOSE=False):
 
    x, y = _img.size
 
    ratio = 2 # relación inicial
 
    # escala al tamaño de cara deseado
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
 
    # reducción de escala para ajustarse a la máxima resolución
    res = x*y
    if res > max_res:
      ratio = pow(res/max_res,1/2); 
      if VERBOSE: print(ratio)
      x=int(x/ratio)
      y=int(y/ratio)
 
    # igualar las dimensiones, porque generalmente los NN fallan en dimensiones desiguales debido a la falta de coincidencia del tamaño de la conexión de omisión
    x = makeEven(int(x))
    y = makeEven(int(y))
    
    size = (x, y)

    return _img.resize(size)

"""Un algoritmo escalador útil, basado en la detección de rostros.
Toma PIL.Image, devuelve un PIL.Image uniformemente escalado
boxes: una lista de bboxes detectados
_img: PIL.Imagen
max_res: área máxima de píxeles para encajar. Utilícelo para mantenerse por debajo de los límites de VRAM de su GPU.
target_face: tamaño de cara deseado. Aumente o reduzca la escala de toda la imagen para ajustar la cara detectada en esa dimensión.
fixed_ratio: escala fija. Ignora el tamaño de la cara, pero no ignora el límite max_res.
max_upscale: ratio máximo de upscale. Evita escalar imágenes con caras diminutas a un desorden borroso.
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

#cargar modelo

version = '0.2' # @parametro ['0.1','0.2']

model_path = f'/content/ArcaneGANv{version}.jit' 
in_dir = '/content/in'
out_dir = f"/content/{model_path.split('/')[-1][:-4]}_out"

model = torch.jit.load(model_path).eval().cuda().half()

#configurar interfaz de colaboración

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


# En[ ]:


#@title Haga clic para cargar archivos y ejecutar la inferencia. Los resultados se guardarán y comprimirán.
process()


# En[ ]:




