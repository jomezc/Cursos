#############################################################
# 35 **Algoritmo GrabCut para la eliminación del fondo**#####
#############################################################
# Es un algoritmo de segmentación.
# - En esta lección vamos a utilizar el algoritmo GrabCut para la eliminación de fondo

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import dlib
import sys
import numpy as np
from matplotlib import pyplot as plt


# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



### **¿Cómo funciona Grab Cut?**
#
# - **El usuario introduce el rectángulo**. Todo lo que esté fuera de este rectángulo se tomará como fondo. Todo dentro
#   del rectángulo es desconocido.
# - El algoritmo etiqueta los píxeles de primer plano y de fondo (o los etiqueta).
#   A continuación, se utiliza un modelo de mezcla gaussiana (GMM) para modelar el primer plano y el fondo.
# - Dependiendo de los datos que hemos dado, GMM aprende y crea una nueva distribución de píxeles. Es decir, los
#   **píxeles desconocidos se etiquetan como probable primer plano o probable fondo** dependiendo de su relación con los
#   otros píxeles etiquetados en términos de estadísticas de color (es como la agrupación).

# A partir de esta distribución de píxeles se construye un gráfico. Los nodos del gráfico son píxeles. Se añaden dos
# nodos adicionales, Source node y Sink node. Cada píxel en primer plano está conectado al nodo Fuente y cada píxel en
# segundo plano está conectado al nodo Sumidero.
# Los pesos de las aristas que conectan los píxeles al nodo origen/nodo final se definen por la probabilidad de que un
# píxel esté en primer plano/fondo. Los pesos entre los píxeles se definen por la información de borde o similitud de
# píxeles. Si hay una gran diferencia en el color de los píxeles, el borde entre ellos tendrá un peso bajo.

# A continuación, se utiliza un algoritmo de corte mínimo para segmentar el gráfico. Corta el gráfico en dos separando
# el nodo de origen y el nodo de destino con la función de coste mínimo. La función de coste es la suma de todos
# los pesos de las aristas que se cortan. Tras el corte, todos los píxeles conectados al nodo origen pasan a primer
# plano y los conectados al nodo sumidero pasan a segundo plano.
# El proceso continúa hasta que la clasificación converge.
#
# ![](https://docs.opencv.org/3.4/grabcut_scheme.jpg)
# Paper - http://dl.acm.org/citation.cfm?id=1015720
# Más información - https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html


# Cargar nuestra imagen
image = cv2.imread('images/woman.jpeg')
copy = image.copy()

# Crear una máscara (de ceros uint8 tipo de datos) que es el mismo tamaño (ancho, alto) como nuestra imagen original
mask = np.zeros(image.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Necesita ser establecido manualmente o seleccionado con cv2.selectROI() ( seleccionarlo vamos)

x1, y1, x2, y2 = 190, 70, 350, 310
start = (x1, y1)
end = (x2, y2)

# El formato es X,Y,W,H
rect = (x1, y1, x2-x1, y2-y1)

# MODIFICADO
rect = cv2.selectROI(copy)  # PARA SELECCIONARLO NOSOTROS

# Mostrar rectángulo
cv2.rectangle(copy, start, end, (0,0,255), 3)
imshow("Input Image", copy) # se muestra el por defecto pero el algoritmo coge el "nuevo"



# #### **Argumentos de recorte**
#
# - **img** - Imagen de entrada
# - **mask** - Es una imagen máscara donde especificamos qué áreas son fondo, primer plano o probable fondo/primer plano
#              , etc. Se hace mediante las siguientes banderas, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, o
#              simplemente pasando 0,1,2,3 a la imagen.
# - **rec**t - Son las coordenadas de un rectángulo que incluye el objeto en primer plano en el formato (x,y,w,h)
# - **bdgModel, fgdModel** - Son matrices utilizadas por el algoritmo internamente. Basta con crear dos arrays de tipo
#                            np.float64 de tamaño cero (1,65).
# - **iterCount** - Número de iteraciones que debe ejecutar el algoritmo.
# - **mode** - Debe ser cv.GC_INIT_WITH_RECT o cv.GC_INIT_WITH_MASK o combinado que decide si estamos dibujando
#              rectángulo o trazos de retoque final.

# Deja que el algoritmo se ejecute durante 5 iteraciones. El modo debe ser cv.GC_INIT_WITH_RECT ya que estamos usando
# rectángulo.
# Grabcut modifica la imagen de máscara.
# En la nueva imagen de máscara, los píxeles serán marcados con cuatro banderas que denotan fondo/primer plano como
# se especificó anteriormente.

cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Así que modificamos la máscara de tal manera que todos los 0-píxeles y 2-píxeles se ponen a 0 (es decir, de fondo)
# y todos los 1-píxeles y 3-píxeles se ponen a 1 (es decir, los píxeles de primer plano).
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Ahora nuestra máscara final está lista. Sólo hay que multiplicarla con la imagen de entrada para obtener la imagen
# segmentada.
image = image * mask2[:,:,np.newaxis]

imshow("Mask", mask * 80)
imshow("Mask2", mask2 * 255)
imshow("Image", image)

