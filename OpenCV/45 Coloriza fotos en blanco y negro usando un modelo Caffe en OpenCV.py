# ****************************************************************************
# ***** 45 Coloriza fotos en blanco y negro usando un modelo Caffe en OpenCV
# ****************************************************************************
# En esta lección aprenderemos a usar modelos pre-entrenados para colorear automáticamente una foto en blanco y negro
# (escala de grises)
#

# ### **Colorizar imágenes en blanco y negro es una técnica increíblemente útil e increíble lograda por el aprendizaje
# profundo.**
#
# [Colorización de imágenes en blanco y negro ](http://arxiv.org/pdf/1603.08511.pdf)
#
# - Los autores abrazan la incertidumbre subyacente del problema (conversión de blanco y negro a color) planteándolo
#   como una tarea de clasificación y utilizan el reequilibrio de clases en tiempo de entrenamiento para aumentar la
#   diversidad de colores en el resultado.
# - El sistema se implementa como un paso feed-forward en una CNN en tiempo de prueba y se entrena con más de un millón
#   de imágenes en color.
# Evalúan nuestro algoritmo mediante una "prueba de Turing de coloración", en la que se pide a los participantes humanos
# que elijan entre una imagen en color generada y otra real.
# Su método consigue engañar a los humanos en el 32% de las pruebas, un porcentaje significativamente superior al de
# métodos anteriores.
#
# ![](http://richzhang.github.io/colorization/resources/images/teaser3.jpg)
#
# por Richard Zhang, Phillip Isola, Alexei A. Efros. En ECCV, 2016.
#
# Utilizaremos los siguientes archivos de modelo Caffe que descargaremos en la siguiente celda de abajo. Estos serán
# luego cargados en OpenCV:
#
# 1. colorization_deploy_v2.prototext
# 2. colorization_release_v2.caffe
# 3. pts_in_hull.npy


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# El script está basado en https://github.com/richzhang/colorization/blob/master/colorize.py
# Para descargar el caffemodel y el prototxt, y npy véase: https://github.com/richzhang/colorization/tree/caffe

# Inicia el programa principal
file_path = "images/color/"
blackandwhite_imgs = [f for f in listdir(file_path) if isfile(join(file_path, f))]
kernel = 'modelos/color/pts_in_hull.npy'

# Selecciona el modelo deseado
if __name__ == '__main__':

    # cargar el modelo y los pesos
    net = cv2.dnn.readNetFromCaffe("modelos/color/colorization_deploy_v2.prototxt",
                               "modelos/color/colorization_release_v2.caffemodel")

    # cargar centros de cluster del fichero .npy ( array de 2D )
    pts_in_hull = np.load(kernel)
    '''[[ -90   50]
        [ -90   60]...'''
    # rellenar los centros de cluster como kernel de convolución 1x1
    # transpose, realiza una transposición de filas a columnas y a eso se le añaden dimensiones (a 1) con reshape
    # que devuelve una array con los mismos valores pero cambio en las dimensiones
    # pasa a ser un array 4D
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    print(pts_in_hull)
    '''[[[[ -90]]
       [[ -90]]
       [[ -90]]
    '''
    # pasa ese kernel como etiqueta de la red, para poder usarlo posteriormente
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    # para cada imagen
    for image in blackandwhite_imgs:

        # carga la imagen
        img = cv2.imread(file_path+image)

        # cambia el orden de los colores y lo pasa a flotante / 255
        img_rgb = (img[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
        # Pasa de BGR a ese fomrato de laboratorio
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
        
        # sacar canal L
        img_l = img_lab[:,:,0]
        
        # obtener el tamaño original de la imagen
        (H_orig,W_orig) = img_rgb.shape[:2] 

        # redimensiona la imagen al tamaño de entrada de la red
        img_rs = cv2.resize(img_rgb, (224, 224)) 
        
        # redimensiona la imagen al tamaño de entrada de la red
        img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:,:,0]
        
        # restar 50 para centrado medio
        img_l_rs -= 50 

        # realiza la transformación de la imagen a blob 4D
        net.setInput(cv2.dnn.blobFromImage(img_l_rs))
        
        # este es nuestro resultado
        # normalmente en net.forward() que realmente realiza el paso de la red con el blob, no introducimos parámetros
        # Sin embargo, en este caso usa la etiqueta antes añadida para pasar el kernel a la red
        ab_dec = net.forward('class8_ab')[0,:,:,:].transpose((1,2,0)) 

        # Saca el ancho y el alto
        (H_out,W_out) = ab_dec.shape[:2]


        ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) 
        
        # concatenar con imagen original L
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

        # mostrar imagen original
        imshow('Original', img)
        # Redimensionar la imagen corlizada a sus dimensiones originales
        img_bgr_out = cv2.resize(img_bgr_out, (W_orig, H_orig), interpolation = cv2.INTER_AREA)
        imshow('Colorized', img_bgr_out)




