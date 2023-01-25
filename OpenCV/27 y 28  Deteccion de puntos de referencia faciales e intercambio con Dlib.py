#!/usr/bin/env python
##########################################################################################
# 27 y 28 Detección de puntos de referencia faciales con Dlib e intercambio de caras######
##########################################################################################
# LIBRERIAS
# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
import cv2
import dlib  # librería de machine learning
import numpy as np
from matplotlib import pyplot as plt

# CLASES UTILIZADAS
class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


# FUNCIONES UTILIZADAS EXPLICADAS DESDE 27 y 28 y leyendolas para entender todo el proceso correctamente
def imshow(title = "Image", image = None, size = 10): # Mostrar por pantalla la imagen
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
def annotate_landmarks(im, landmarks):  # Dibuja las marcas de línea que tenemos en la cara.
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,

                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
def get_landmarks(im): # Toma una imagen.
    """
    La función get_landmarks()toma una imagen en forma de matriz numpy y devuelve una matriz de elementos de 68x2, cada
    una de las cuales se corresponde con las coordenadas x, y de un punto de característica particular en la imagen de
    entrada.

    El extractor de características (predictor) requiere un cuadro delimitador aproximado como entrada para el algoritmo
    Esto lo proporciona un detector de rostros tradicional (detector) que devuelve una lista de rectángulos, cada uno de
     los cuales corresponde a un rostro en la imagen.
    """

    rects = detector(im, 1)  # Lo pasa por el detector.

    # resuelve los cuadros delimitadores aqui, pues solo queremos 1
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    # Es donde realmente llamamos a predictor (codemos la imagen , una x en particular , siendo el 1º el único que
    # queremos) lo ejecutamos a través del predictor
    # mediante la lista de comprensión obtenemos las predicciones históricas que obtenemos del predictor. vamos metiendo
    # las coordenadas X e Y de todas esas predicciones históricas.
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


# Es una transformación desde dos puntos. Devuelve la transformación afin para que los puntos se alineen y la
# perspectiva se ajuste correctament
def transformation_from_points(points1, points2):
    """
    ##2. Alineación de caras con un análisis de procrustes
    Así que en este punto tenemos nuestras dos matrices de puntos de referencia, cada fila tiene las coordenadas de un
    rasgo facial en particular (por ejemplo, la fila 30 da las coordenadas de la punta de la nariz). Ahora vamos
    a averiguar cómo rotar, trasladar y escalar los puntos del primer vector para que se ajusten lo más posible a los
    puntos del segundo vector, la idea es que la misma transformación se puede usar para superponer la segunda imagen
    sobre la primera.
    """
    # Resolver el problema procrustes restando centroides, escalando por la
    # desviación estándar, y luego usando el SVD para calcular la rotación. Ver
    # lo siguiente para más detalles:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # 1. Convierte las matrices de entrada en flotantes. Esto es necesario para las operaciones que van a seguir.
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # 2. Resta el centroide de cada uno de los conjuntos de puntos. Una vez que se ha encontrado una escala y una
    # rotación óptimas para los conjuntos de puntos resultantes, los centroides c1 y c2 se pueden usar para encontrar
    # la solución completa.
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    # 3. Del mismo modo, divida cada punto establecido por su desviación estándar. Esto elimina el componente de escala
    # del problema.
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    # 4. Calcule la porción de rotación utilizando la Descomposición de valores singulares . Consulte el artículo de
    # wikipedia https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # sobre el problema de Procrustes ortogonal para obtener detalles sobre cómo funciona.
    U, S, Vt = np.linalg.svd(points1.T * points2)

    # La R que buscamos es en realidad la transpuesta de la dada por U * Vt. Esto
    # es porque la formulación anterior asume que la matriz va a la derecha
    # (con vectores fila) mientras que nuestra solución requiere que la matriz vaya a la
    # izquierda (con vectores columna).
    R = (U * Vt).T

    # Devuelve la transformación completa como una matriz de transformación afín
    """Devuelve una transformación afín [s * R | T] tal que:
        suma ||s*R*p1,i + T - p2,i||^2
    se minimiza."""
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])

#Esta es una función de imagen distorsionada donde simplemente eliminamos la imagen, la matriz y la forma, y
# simplemente emita la imagen final en función de esos parámetros.
def warp_im(im, M, dshape): #  asigna la segunda imagen a la primera
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


# Esta es la corrección de color.
def correct_colours(im1, im2, landmarks1):
    """
    El problema es que las diferencias en el tono de la piel y la iluminación entre las dos imágenes provocan una
    discontinuidad alrededor de los bordes de la región superpuesta Tratamos de corregir eso

    Esta función intenta cambiar el color de im2para que coincida con el de im1. Lo hace dividiendo im2por un desenfoque
    gaussiano de im2y luego multiplicando por un desenfoque gaussiano de im1. La idea aquí es la de una corrección de
    color de escala RGB , pero en lugar de un factor de escala constante en toda la imagen, cada píxel tiene su propio
    factor de escala localizado.

    Con este enfoque, las diferencias de iluminación entre las dos imágenes pueden explicarse, hasta cierto punto. Por
    ejemplo, si la imagen 1 está iluminada desde un lado pero la imagen 2 tiene una iluminación uniforme, entonces la i
    magen 2 con el color corregido aparecerá más oscura en el lado no iluminado también.

    Dicho esto, esta es una solución bastante cruda al problema y un kernel gaussiano de tamaño apropiado es clave.
    Demasiado pequeño y los rasgos faciales de la primera imagen aparecerán en la segunda. Demasiado grande y el kernel
    se desvía fuera del área de la cara para que los píxeles se superpongan y se produce una decoloración. Aquí se
    utiliza un núcleo de 0,6 * la distancia pupilar.
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(  # desenfoque gaussiano para asegurarse de que se vea bien.
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Evitar errores de división por cero.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# draw_convex_hull es un casco convexo de dibujo, que nos permite mapear los puntos correctamente en tres interfaces.
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# get_face_mask para obtener la primera masa para que podamos extraer la cara de la imagen para ponerla en la primera
# imagen.
def get_face_mask(im, landmarks):
    """
    Se define una rutina para generar una máscara para una imagen y una matriz de puntos de referencia. Dibuja dos
    polígonos convexos en blanco: uno que rodea el área de los ojos y otro que rodea el área de la nariz y la boca.
    Luego, desvanece el borde de la máscara hacia afuera en 11 píxeles. El calado ayuda a ocultar las discontinuidades
    remanentes

    Dicha máscara facial se genera para ambas imágenes. La máscara de la segunda se transforma en el espacio de
    coordenadas de la imagen 1, usando la misma transformación que en el paso 2.
    """
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

# Obtienes la función simple de puntos de referencia.
def read_im_and_landmarks(image):
    im = image
    im = cv2.resize(im,None,fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


def swappy(image1, image2):
    # 1. Detección de puntos de referencia faciales : get_landmarks
    # 2. Rotar, escalar y traducir la segunda imagen para que se ajuste a la primera: transformation_from_points y warp_im
    #    Luego, el resultado se puede conectar a la cv2.warpAffinefunción de OpenCV para asignar la segunda imagen a la
    #    primera:
    # 3. Ajuste del balance de color en la segunda imagen para que coincida con el de la primera: correct_colours
    # 4 .Fusión de características de la segunda imagen encima de la primera: *_POINTS, draw_convex_hull, get_face_mask
    im1, landmarks1 = read_im_and_landmarks(image1)
    im2, landmarks2 = read_im_and_landmarks(image2)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    """
    Dicha máscara facial se genera para ambas imágenes. La máscara de la segunda se transforma en el espacio de 
    coordenadas de la imagen 1, usando la misma transformación que en el paso 2.
    """
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    # Luego, las máscaras se combinan en una tomando un máximo de elementos. La combinación de ambas máscaras asegura
    # que las características de la imagen 1 estén cubiertas y que las características de la imagen 2 se vean.
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    return image


# 27. Aplicar la detección de puntos de referencia faciales
'''# Descarga y descomprime nuestras imágenes y el modelo Facial landmark
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip')
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/shape_predictor_68_face_landmarks.zip')
get_ipython().system('unzip -qq images.zip')
get_ipython().system('unzip -qq shape_predictor_68_face_landmarks.zip')'''


# ## **Detección de puntos de referencia faciales**
PREDICTOR_PATH = "modelos/shape_predictor_68_face_landmarks.dat" # poniendo la parte del modelo en esta variable de aquí
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # cargando el predictor que es un objeto predictor de dylib.
# entra lo que ella predice y solo señalamos la parte del modelo.
detector = dlib.get_frontal_face_detector()  # creamos el detector

# usamos las funciones declaradas y explicadas en este fichero
image = cv2.imread('images/Trump.jpg')
imshow('Original', image)
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
imshow('Result', image_with_landmarks)


# Otra imagen
image = cv2.imread('images/Hillary.jpg')
imshow('Original', image)
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
imshow('Result', image_with_landmarks)


# ##
# ## ** 28 Intercambio de caras**
# ## http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
# El proceso se divide en cuatro pasos, organizado en la funcion swappy:


import sys

PREDICTOR_PATH = "modelos/shape_predictor_68_face_landmarks.dat"
# En primer lugar, declaramos algunas variables, que es un camino para el efecto de escala del modelo de predicción.

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11 # y la cantidad, que es básicamente cuánto estamos haciendo las capas de las caras.

'''Tenemos de cero a 68 puntos, cada uno de esos puntos Cada uno de esos puntos tiene algunos rangos, que corresponden 
a las partes de la cara. Son los puntos que necesitamos engranar y alinear para que podamos alinearnos en la cara.
'''
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Puntos utilizados para alinear las imágenes.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Puntos de la segunda imagen a superponer sobre la primera. Se superpondrá el casco convexo de cada
# elemento se superpondrá.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Cantidad de desenfoque a utilizar durante la corrección de color, como fracción de la
# distancia pupilar, puede hacer que el intercambio de caras se vea un poco más realista.
COLOUR_CORRECT_BLUR_FRAC = 0.6  # parámetro de factor de corrección de color.

#  Detectar y predecir dos objetos.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

## Introduzca aquí las rutas a sus imágenes de entrada
image1 = cv2.imread('images/Hillary.jpg')
image2 = cv2.imread('images/Trump.jpg')

swapped = swappy(image1, image2)
imshow('Face Swap 1', swapped)

swapped = swappy(image2, image1)
imshow('Face Swap 2', swapped)



# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:
    ./faceswap.py <head image> <face image>
If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.
"""