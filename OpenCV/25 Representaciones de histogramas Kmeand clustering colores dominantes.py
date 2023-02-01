#################################################################################
# 25 Representaciones de histogramas Kmeand clustering colores dominantes ######
#################################################################################
# 1. Visualizar las representaciones del histograma RGB de las imágenes
# 2. Utilizar K-Means Clustering para obtener los colores dominantes y sus proporciones en las imágenes.

# k-means -> agrupamiento para la causa dominante de una imagen.

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes

# Un histograma es básicamente un gráfico, un diagrama de barras o un grafico de líneas, Y un histograma nos da
# básicamente la distribución de algo. Entonces, en el caso de una imagen que vamos a dar, vamos a pasar por una
# distribución de los colores.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
def imshow(title = "Image", image = None, size = 8):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# In[ ]:


image = cv2.imread('images/input.jpg')
imshow("Input", image)

# histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

# Trazamos un histograma, la función ravel() aplana nuestra matriz de imágenes, Así que lo tenemos como una gran matriz
# unidimensional
''' 
ORIGINAL
[[ 3  6 11]
  [ 3  6 11]
  [ 2  5 10]
  ...
  [18 23 38]
  [18 23 38]
  [19 24 39]]]
  
  tras ravel() 
  [12 18 31 ... 19 24 39]'''
print(image)
print(image.ravel())

plt.hist(image.ravel(), 256, [0, 256])  # imagen aplanada, cantidad de contenedores que queremos, el rango

plt.show()  # el básico aplanado muestra el brillo de una imagen en el que el eje vertical es el número de píxeles y
# el horizontal el rango de brillo si se ve un pico al principio significa que hay muchos píxeles oscuros y al final
# claros

# **cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])**
# Usamos la función para calcular el histograma de una imagen
# - **images** : es la imagen de origen de tipo uint8 o float32. Debe darse entre corchetes, es decir, "[img]".
# - **channels** : también se indica entre corchetes. Es el índice del canal para el que calculamos el histograma.
#                  Por ejemplo, si la entrada es una imagen en escala de grises, su valor es [0]. Para una imagen en
#                  color, puede pasar [0], [1] o [2] para calcular el histograma del canal azul, verde o rojo
#                  respectivamente.
# - **mask** : imagen de máscara. Para encontrar el histograma de la imagen completa, se le da como "Ninguno". Pero si
#              desea encontrar el histograma de una región particular de la imagen, tiene que crear una imagen de
#              máscara para eso y darle como máscara. (Mostraré un ejemplo más adelante).
# - **histSize** : esto representa nuestro recuento BIN. Necesita ser dado entre corchetes. Para escala completa,
#                   pasamos [256].
# - **ranges** : este es nuestro RANGO. Normalmente es [0,256].


# Visualización de canales de color separados, las etiquetas de cada color
color = ('b', 'g', 'r')

# Ahora separamos los colores y trazamos cada uno en el Histograma para ver la distribución en el histograma por color
for i, col in enumerate(color):
    # para cada canal ( que contienen el color) de la imagen calculamos su histograma
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)  #  color = col para especificar las etiquetas ver cada color de su color
    plt.xlim([0,256]) # establecemos los límites
    
plt.show()


# hacemos lo mismo con otra imagen para anailizar su distribución de colores
image = cv2.imread('images/tobago.jpg')
imshow("Input", image)

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Trazamos un histograma, ravel() aplana nuestra matriz de imágenes
plt.hist(image.ravel(), 256, [0, 256]); plt.show()

# Visualización de canales de color separados
color = ('b', 'g', 'r')

# Ahora separamos los colores y trazamos cada uno en el Histograma
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])
    
plt.show()


# ## **K-Means Clustering para obtener los colores dominantes en una imagen**
# k-means es básicamente un algoritmo de agrupamiento que agrupa píxeles de valor similar.

def centroidHistogram(clt):
    # Crea un histograma para los clusters basado en los píxeles de cada cluster.
    # Obtener las etiquetas de cada cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Crear nuestro histograma
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # Normalizar el histograma, para que sume uno
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plotColors(hist, centroids): # nos da la distribución de los colores

    # Crear nuestro gráfico de barras en blanco
    bar = np.zeros((100, 500, 3), dtype = "uint8")

    x_start = 0
    # iterar sobre el porcentaje y el color dominante de cada cluster
    for (percent, color) in zip(hist, centroids):
        # trazar el porcentaje relativo de cada cluster
        end = x_start + (percent * 500)
        cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),color.astype("uint8").tolist(), -1)
        x_start = end
    return bar


from sklearn.cluster import KMeans

image = cv2.imread('images/tobago.jpg')
imshow("Input", image)

# Transformamos nuestra imagen en una lista de píxeles RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)  # (1194, 1936, 3)

# remodelamos la imagen porque necesitamos que tenga un formato específico para el CEO de la empresa
image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)  # (2311584, 3) hemos pasado de una imagen tridimensional a una imagen bidimensional

# vamos a crear 5 grupos
number_of_clusters = 5

# ejecutamos el modelo de agrupamiento K
clt = KMeans(number_of_clusters)

# Así que simplemente hacemos el ajuste de puntos de K mientras creamos un sello,
# El objeto clt, que es una clave, significa objeto de agrupación, agrupa píxeles de valor similar
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist, clt.cluster_centers_)

# mostrar nuestro color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


# ### **Probemos con otra imagen**

from sklearn.cluster import KMeans

image = cv2.imread('images/Volleyball.jpeg')
imshow("Input", image)

# Transformamos nuestra imagen en una lista de píxeles RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))

number_of_clusters = 3
clt = KMeans(number_of_clusters)
clt.fit(image)

hist = centroidHistogram(clt)
bar = plotColors(hist, clt.cluster_centers_)

# muestra nuestro color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()




