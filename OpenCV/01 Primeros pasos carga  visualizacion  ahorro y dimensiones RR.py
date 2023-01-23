########################################################################
### 01 Primeros pasos carga  visualizacion  ahorro y dimensiones RR ####
########################################################################

#!/usr/bin/env python
# coding: utf-8

'''
primera lección de OpenCV. Aquí aprenderemos a:
1. Importar el modelo OpenCV en Python
2. Cargar imágenes
3. Mostrar imágenes
4. Guardar imágenes
5. Obtención de las dimensiones de la imagen
'''

# Así es como importamos OpenCV, no podemos usar las funciones de OpenCV sin antes hacer esto
import cv2


# Veamos qué versión estamos ejecutando
print(cv2.__version__)  # 4.7.0


# ### **Descargando imagenes**
''' comandos ipynb
# Download and unzip our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('unzip -qq images.zip')'''


# ### **Cargamos las imagenes**
# Cargue una imagen usando 'imread' especificando la ruta a la imagen
image = cv2.imread('./images/castara.jpeg')


# ### **Mostramos las imagenes**
from matplotlib import pyplot as plt

# Mostramos la imagen con matpoit matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
'''usamos una funcion de open cv para cambiar el color porque Open CV en su dimensión de colores utiliza el orden Blue 
Green red, BGR y matploit usa red green blue, RGB. necesitamos esos espacios de color porque necesitamos esos 3 colores 
primarios para crear cualquier color que queramos'''
plt.show()


# Vamos a crear una función simple para hacer que mostrar nuestras imágenes sea más simple y fácil.
def imshow(title="", image = None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # cambio de color
    plt.title(title)  # le damos un título a la imagen
    plt.show()  # mostramos la imagen


# Vamos a probarlo
imshow("Displaying Our First Image", image)


# ### **Salvamos la imagen**
# Simplemente use 'imwrite' especificando el nombre del archivo y la imagen que se guardará
cv2.imwrite('output.jpg', image)

# O guárdelo como PNG (gráficos de red portátiles), que es un formato de imagen de mapa de bits sin pérdida
cv2.imwrite('output.png', image)


# ### **mostramos las dimensiones de la imagen**
# Recuerda las imágenes son arrays::
# ![](https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/array.png?token=ADLZD2HNEL33JAKTYRM3B5C7WMIV4)


# Necesitamos usar numpy para realizar esta operación. No te preocupes, numpy se convertirá en uno de tus mejores amigos
# si estás aprendiendo ciencia de datos y visión artificial.

# Import numpy, librería numérica de arrays
import numpy as np

print(image.shape)  # (1280, 960, 3), es una estructura tridimensional, de ancho, alto y color

# # Para acceder a una dimensión, simplemente indícela usando 0, 1 o 2.
image.shape[0]


# Puedes ver que la primera dimensión es la altura y tiene 960 píxeles
# La segunda dimensión es el ancho, que es de 1280 píxeles.
# Podemos imprimirlos muy bien así:
print('Height of Image: {} pixels'.format(int(image.shape[0])))  # Height of Image: 1280 pixels
print('Width of Image: {} pixels'.format(int(image.shape[1])))  # Width of Image: 960 pixels
print('Depth of Image: {} colors components'.format(int(image.shape[2])))  # Depth of Image: 3 colors components
