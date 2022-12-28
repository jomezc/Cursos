# Imports
import cv2  # pip install opencv-python es el módulo de open cv
import numpy as np  #
import matplotlib.pyplot as plt
# %matplotlib inline # en cuadernos jupiter para poder mostrar directamente las imágenes del cuaderno
from IPython.display import Image  # nos permitirá mostrar y renderizar imágenes directamente en el cuaderno.

# ******************************************
# ***** Operaciones aritméticas con imágenes
# ******************************************
'''
Las técnicas de procesamiento de imágenes aprovechan las operaciones matemáticas para lograr diferentes resultados. 
La mayoría de las veces llegamos a una versión mejorada de la imagen usando algunas operaciones básicas. Echaremos un 
vistazo a algunas de las operaciones fundamentales que se usan a menudo en las canalizaciones de visión por computadora.
En este cuaderno cubriremos operaciones aritméticas como la suma y la multiplicación.
'''
img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)  # cargar imagen a color
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cambiar el color a RGB (cv2 por defecto BGR)

#  Adición o Brillo
'''
La primera operación que analizamos es la simple adición de imágenes. Esto da como resultado aumentar o disminuir el 
brillo de la imagen ya que eventualmente estamos aumentando o disminuyendo los valores de intensidad de cada píxel en 
la misma cantidad. Entonces, esto resultará en un aumento/disminución global del brillo.
'''
matrix = np.ones(img_rgb.shape, dtype = "uint8") * 50
img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker   = cv2.subtract(img_rgb, matrix)
plt.figure(figsize=[18 , 5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Brighter");
plt.show()