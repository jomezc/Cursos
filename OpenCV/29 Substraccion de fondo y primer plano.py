################################################
# 29 Substracción de fondo y primer plano ######
################################################
# 1. 1. Sustracción de fondo con algoritmo de segmentación de fondo/primer plano basado en mezcla gaussiana.
# 2. Modelo de mezcla gaussiana adaptativo mejorado para sustracción de fondo

## La sustracción de fondo (BS) es una técnica común y ampliamente utilizada para generar una máscara de primer plano
# (es decir, una imagen binaria que contiene los píxeles pertenecientes a los objetos en movimiento de la escena)
# mediante el uso de cámaras estáticas.
#
# Como su nombre indica, la BS calcula la máscara de primer plano realizando una sustracción entre el fotograma actual
# y un modelo de fondo, que contiene la parte estática de la escena o, más en general, todo lo que puede considerarse
#  como fondo dadas las características de la escena observada.
#
# ![](https://docs.opencv.org/3.4/Background_Subtraction_Tutorial_Scheme.png)
#
# El modelado del fondo consta de dos pasos principales:
# 1. 1. Inicialización del fondo;
# 2. Actualización del fondo.
#
# En el primer paso se calcula un modelo inicial del fondo, mientras que en el segundo se actualiza dicho modelo para
# adaptarse a posibles cambios en la escena.

# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# **¿Qué es la sustracción de fondo?**

# La sustracción de fondo es una técnica de visión por ordenador en la que buscamos aislar el fondo del primer plano
# 'en movimiento'. Consideremos los vehículos que atraviesan una carretera o las personas que caminan por una acera.
#
# Suena sencillo en teoría (es decir, basta con mantener los píxeles fijos y eliminar los que cambian). Sin embargo,
# cosas como cambios en las condiciones de iluminación, sombras, etc. pueden complicar las cosas.
#
# Se han introducido varios algoritmos para este propósito. A continuación veremos dos algoritmos del módulo **bgsegm**.


# *** Algoritmo de segmentación de fondo/primer plano basado en mezclas gaussianas *****
#
# En este trabajo, proponemos un método de sustracción de fondo (BGS) basado en los modelos de mezcla gaussiana
# utilizando información de color y profundidad. Para combinar la información de color y profundidad, utilizamos el
# modelo probabilístico basado en la distribución gaussiana. En particular, nos centramos en resolver el problema del
# camuflaje de color y la eliminación de ruido en profundidad. Para evaluar nuestro método, hemos creado un nuevo
# conjunto de datos que contiene situaciones normales, de camuflaje de color y de camuflaje de profundidad. Los
# archivos del conjunto de datos constan de secuencias de imágenes en color, en profundidad y de la verdad sobre el
# terreno. Con estos archivos, comparamos el algoritmo propuesto con las técnicas convencionales de BGS basadas en el
# color en términos de precisión, recuperación y medida F. El resultado fue que nuestro método demostró ser más preciso
# que los algoritmos convencionales. Como resultado, nuestro método mostró el mejor rendimiento. Así pues, esta técnica
# ayudará a detectar de forma robusta regiones de interés como preprocesamiento en etapas de procesamiento de imágenes
# de alto nivel.
#
# Enlace al artículo -
# https://www.researchgate.net/publication/283026260_Background_subtraction_based_on_Gaussian_mixture_models_using_color_and_depth_information


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y la anchura del fotograma (se requiere que sea un interger)
w = int(cap.get(3)) 
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en un archivo '*.avi'.
out = cv2.VideoWriter('videos/walking_output_GM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()

# Bucle una vez que el vídeo se ha cargado correctamente
while True:

    ret, frame = cap.read()

    if ret:
        #  Aplicar el sustractor de fondo para obtener nuestra máscara de primer plano
        foreground_mask = foreground_background.apply(frame)
        out.write(foreground_mask)
        imshow("Foreground Mask", foreground_mask)
    else:
        break

cap.release()
out.release()


# ### **Probemos el modelo de mezcla gausiano adaptativo mejorado para la sustracción de fondo**
#
# La sustracción de fondo es una tarea común de visión por ordenador. Analizamos el enfoque habitual a nivel de píxel.
# Desarrollamos un algoritmo adaptativo eficiente utilizando la densidad de probabilidad de la mezcla gaussiana.
# Se utilizan ecuaciones recursivas para actualizar constantemente los parámetros y también para seleccionar
# simultáneamente el número apropiado de componentes para cada píxel.
# https://www.researchgate.net/publication/4090386_Improved_Adaptive_Gaussian_Mixture_Model_for_Background_Subtraction


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output_AGMM.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

# Bucle una vez que el vídeo se ha cargado correctamente
while True:

    ret, frame = cap.read()
    if ret:
        # Aplicar el sustractor de fondo para obtener nuestra máscara de primer plano
        foreground_mask = foreground_background.apply(frame)
        out.write(foreground_mask)
        imshow("Foreground Mask", foreground_mask)
    else:
      break

cap.release()
out.release()


# ## **Substracción de primer plano**


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))
ret, frame = cap.read()

# Crear un array numpy float con los valores de los fotogramas
average = np.float32(frame)

while True:
    # Obtener frame
    ret, frame = cap.read()

    if ret:
        # accumulateWeighted nos permite básicamente, almacenar valores del frame pasado.
        # 0.01 es el peso de la imagen, juega para ver como cambia
        cv2.accumulateWeighted(frame, average, 0.01)
        # Posteriomente con esos valores almacenados podemos obtener con convertScaleAbs el promedio, que es lo que
        # especificamos aquí, obteniendo el valor promedio del marco Es una forma de hacer un seguimiento de lo que es
        # el fondo.
        # Escala, calcula valores absolutos, y convierte el resultado a 8-bit, obtenemos así matemáticamente el fondo
        background = cv2.convertScaleAbs(average)

        imshow('Input', frame)
        imshow('Disapearing Background', background)
        out.write(background)
        # No es tan evidente en estas imágenes. Sin embargo, se acumula con el tiempo, por lo que cuanto más tiempo lo
        # dejemos, más se acumulará (no es el mejor método).

    else:
      break

cap.release()
out.release()




cv2.imshow(background)


### **Background Substraction KKN** ( el mejor de este documento)
#
# Los parámetros si desea desviarse de la configuración predeterminada:
#
# - **history** es el número de fotogramas utilizados para construir el modelo estadístico del fondo. Cuanto menor sea
#               el valor, más rápido serán tenidos en cuenta por el modelo los cambios en el fondo y, por tanto, serán
#               considerados como fondo. Y viceversa.
# - **dist2Threshold** es un umbral para definir si un píxel es diferente del fondo o no. Cuanto menor sea el valor,
#                      más sensible será la detección del movimiento. Y viceversa.
# ** detectShadows **: Si se establece en true, las sombras se mostrarán en gris en la máscara generada.(Ejemplo abajo)
#
# https://docs.opencv.org/master/de/de1/group__video__motion.html#gac9be925771f805b6fdb614ec2292006d


cap = cv2.VideoCapture('videos/walking_short_clip.mp4')

# Obtener la altura y anchura del fotograma (necesario para ser una interferencia)
w = int(cap.get(3))
h = int(cap.get(4))

# Definir el codec y crear el objeto VideoWriter. La salida se almacena en el archivo 'outpy.avi'.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

# Obtenemos la estructura del kernel o o matriz de árboles con getStructuringElement usando MORPH_ELLIPSE
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# inicializamos el substractor de fondo
fgbg = cv2.createBackgroundSubtractorKNN()

while (1):
    ret, frame = cap.read()

    if ret:

        # aplicamos el algoritmo al frame mediante el método apply, obteniendo el 1º plano
        fgmask = fgbg.apply(frame)

        # luego debemos aplicar el primer plano la morfología x, que es, usar la función con el kernel que definimos y
        # obtener la salida
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        imshow('frame', fgmask)
    else:
      break

cap.release()
out.release()




