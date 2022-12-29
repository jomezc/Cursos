import cv2
import glob
import matplotlib.pyplot as plt
import math
# ***********************************************
# ***** Unión de imágenes y creación de panoramas
# ***********************************************
# Características de la imagen y alineación de la imagen

# Creando panoramas usando OpenCV
'''
1. Encuentra puntos clave en todas las imágenes
2. Encuentra correspondencias por pares
3. Estimar homografías por pares
4. Refinar homografías
5. Puntada con mezcla

podemos realizar todos estos pasos con la clase stitcher, es muy similar a los pasos que se explican en  
Características de la imagen y alineación de la imagen. stitcher es una clase que nos permite crear panoramas 
simplemente pasando una lista de imágenes.

las imágenes utilizadas para crear panoramas deben tomarse desde el mismo punto de vista Y también es importante tomar 
las fotos aproximadamente al mismo tiempo para minimizar la iluminación.
'''

# Leemos las imágenes,
'''glob incluye funciones para buscar en una ruta todos los nombres de archivos y/o directorios que coincidan con un 
determinado patrón 
glob.glob() devuelve una lista con las entradas que coincidan con el patrón especificado en pathname.
glob.glob(pathname, recursive=False)
La búsqueda se puede hacer también recursiva con el argumento recursive=True y las rutas pueden ser absolutas 
y relativas.'''
imagefiles = glob.glob("boat/*")
imagefiles.sort()  # ordenamos la lista obtenida
# ['boat\\boat1.jpg', 'boat\\boat2.jpg', 'boat\\boat3.jpg', 'boat\\boat4.jpg', 'boat\\boat5.jpg', 'boat\\boat6.jpg']

images = []
# recorremos la lista de imagenes y para cada imagen la leemos en color añadiendo los objeto a una lista de imágenes
for filename in imagefiles:
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  images.append(img)

num_images = len(images)

# mostramos las imágenes
plt.figure(figsize=[30,10])
num_cols = 3
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
  plt.subplot(num_rows, num_cols, i+1)
  plt.axis('off')
  plt.imshow(images[i])
plt.show()

# Stitch Images
'''
Creamos un objeto Stitcher desde la clase Stitcher_create(). Usamos ese objeto para llamar al método de stitch y 
simplemente pasamos una lista de imágenes. el resultado que obtenemos es la imagen panorámica.
El panorama de retorno incluye estas regiones negras. aquí, que son el resultado de la deformación que se requirió para 
 unir las imágenes.'''
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)
if status == 0:
  plt.figure(figsize=[30,10])
  plt.imshow(result)
plt.show()
