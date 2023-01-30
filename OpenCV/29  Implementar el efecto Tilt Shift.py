#!/usr/bin/env python
# coding: utf-8

##############################################
# 29 **Implement the Tilt Shift Effect**######
##############################################
# Tilt Shift  es un efecto que toma nuestra imagen estándar normal, como el paisaje de una ciudad o de arriba hacia
# abajo, algo que es bonito y lo hace parecer parece que es un modelo miniaturizado, enfocándose en ciertas áreas y
# luego difuminando otras.

# - En esta lección veremos algo de código que genera nuestro efecto Titl Shift en nuestras imágenes de ejemplo
# Fuente - https://github.com/powersj/tilt-shift


# Nuestra configuración, importar librerías, crear nuestra función Imshow y descargar nuestras imágenes
from matplotlib import pyplot as plt
import cv2
import math
import os
import numpy as np
import scipy.signal
import shutil

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

'''# Download our images
get_ipython().system('wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/images_tilt.zip')
get_ipython().system('unzip -qq images_tilt.zip')
get_ipython().system('find . -name ".DS_Store" -delete')
get_ipython().system('find . -name ".ipynb_checkpoints" -delete')'''


# #### **Nuestras funciones para implementar Tilt Shift**
"""Script para mezclar dos imágenes"""

def get_images(sourcefolder):
    """Carga las fotos de cada carpeta y las devuelve"""
    filenames = os.listdir(sourcefolder)

    # Según el nombre del fichero carga la foto
    for photo in filenames:
        black_img = cv2.imread('images/images_tilt/original/' + photo)
        white_img = cv2.imread('images/images_tilt/blur/' + photo)
        mask_img = cv2.imread('images/images_tilt/mask/' + photo)

        # comprueba la carga de las imágenes
        if mask_img is None:
            print('Oops! There is no mask of image: ', photo)
            continue
        if white_img is None:
            print('Oops! There is no blurred version of image: ', photo)
            continue

        # Comprueba que el tamaño de la imagen, su imagen difuminada y su máscara sean iguales
        assert black_img.shape == white_img.shape, \
            "Error - los tamaños de orignal y blur no son iguales"

        assert black_img.shape == mask_img.shape, \
            "Error - los tamaños del original y la máscara no son iguales"

        print(photo)
        yield photo, white_img, black_img, mask_img

def run_blend(black_image, white_image, mask):
    """ Esta función administra la mezcla de las dos imágenes según la máscara. Asume que todas las imágenes son de
    tipo float, y devuelve un tipo float. Recordar que la función ha recibido un solo canal de cada imagen
    """
    # Calcula automáticamente el tamaño
    min_size = min(black_image.shape)
    # calcula la profundidad, al menos 16x16 en el nivel más alto.
    depth = int(math.floor(math.log(min_size, 2))) - 4

    # llama a gauss_pyramid, para construir una pirámide a partir de la imagen
    gauss_pyr_mask = gauss_pyramid(mask, depth)
    gauss_pyr_black = gauss_pyramid(black_image, depth)
    gauss_pyr_white = gauss_pyramid(white_image, depth)

    # Construye una pirámide laplaciana, reduce la imagen de una mayor a otra menor perdiendo poca calidad
    lapl_pyr_black = lapl_pyramid(gauss_pyr_black)
    lapl_pyr_white = lapl_pyramid(gauss_pyr_white)

    # Mezcla las dos pirámides laplacianas ponderándolas según la máscara gaussiana
    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    # Colapsa una pirámide de entrada, ( une las imágenes de la pirámide en una expandiendo y añadiendo)
    outimg = collapse(outpyr)

    # la mezcla a veces resulta en números ligeramente fuera de límites.
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)

    # *La imagen es de 1 canal*, devuelve el resultado
    return outimg


def gauss_pyramid(image, levels):
    """ Construye una pirámide a partir de la imagen reduciéndola por el número de
    niveles introducidos en la entrada.

    Nota: Necesitas usar tu función reduce en esta función para generar la salida.
    salida.
    Args:
      image (numpy.ndarray): Una imagen en escala de grises de dimensión (r,c) y dtype
      float.
      levels (uint8): Un entero positivo que especifica el número de
                    reducciones que debe hacer. Así, si levels = 0, debe
                    devolver una lista que contenga sólo la imagen de entrada. Si
                    levels = 1, debe hacer una reducción.
                    len(salida) = niveles + 1
    Devuelve:
      output (lista): Una lista de matrices de dtype np.float. El primer elemento de
                la lista (output[0]) es la capa 0 de la pirámide (la imagen
                output[1] es la capa 1 de la pirámide (la imagen reducida
                una vez), etc. Ya hemos incluido la imagen original en
                la matriz de salida. Las matrices son de tipo numpy.ndarray.
    """
    output = [image]
    for level in range(levels):
        output.append(reduce_img(output[level]))
    return output


def lapl_pyramid(gauss_pyr):
    """ Construye una pirámide laplaciana a partir de la pirámide gaussiana, de altura
    niveles. Reduce la imagen de una mayor a otra menor perdiendo poca calidad ( el algoritmo de wats up se basa en
    este vamos.

    Nota: Debes usar tu función expand en esta función para generar la
    salida. La pirámide gaussiana que se pasa es la salida de su función
    gauss_pyramid.

    Args:
      gauss_pyr (lista): Una pirámide gaussiana devuelta por la función gauss_pyramid
                     gauss_pyramid. Es una lista de elementos numpy.ndarray.

    Devuelve:
      output (list): Una pirámide Laplaciana del mismo tamaño que gauss_pyr. Esta pirámide
                   pirámide debe ser representada de la misma manera que guassPyr,
                   como una lista de matrices. Cada elemento de la lista
                   corresponde a una capa de la pirámide de Laplaciano, que contiene
                   la diferencia entre dos capas de la pirámide de Gauss.

           output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])

           Nota: El último elemento de la salida debe ser idéntico a la última
           capa de la pirámide de entrada ya que no se puede restar más.

    Nota: A veces, el tamaño de la imagen expandida será mayor que la capa
    capa dada. Debe recortar la imagen expandida para que coincida en forma con
    con la capa dada.

    Por ejemplo, si mi capa es de tamaño 5x7, reduciendo y expandiendo resultará
    una imagen de tamaño 6x8. En este caso, recorte la capa expandida a 5x7.
    """
    output = []
    # revisa las listas, pero ignora el último elemento ya que no puede ser
    # restado
    for image1, image2 in zip(gauss_pyr[:-1], gauss_pyr[1:]):
        # añadir la diferencia restada
        # expandir y enlazar la segunda imagen en función de las dimensiones de la primera
        output.append(
            image1 - expand(image2)[:image1.shape[0], :image1.shape[1]])

    # añade ahora el último elemento
    plt.imshow(gauss_pyr[-1])
    output.append(gauss_pyr[-1])

    return output


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    """ Mezcla las dos pirámides laplacianas ponderándolas según la máscara gaussiana. Máscara gaussiana.
    Args:
      lapl_pyr_white (lista): Una pirámide laplaciana de una imagen, construida por su función lapl_pyramid.
      lapl_pyr_black (lista): Una pirámide de Laplaciano de otra imagen, construida por la función lapl_pyramid.
                        construida por su función lapl_pyramid.
      gauss_pyr_mask (lista): Una pirámide gaussiana de la máscara. Cada valor está en el rango de [0, 1].

    Las pirámides tendrán el mismo número de niveles. Además, se garantiza que cada capa
    tiene garantizada la misma forma que los niveles anteriores.

    Debe devolver una pirámide Laplaciana que tenga las mismas dimensiones que las
    pirámides de entrada. Cada capa debe ser una mezcla alfa de las correspondientes
    capas de las pirámides de entrada, ponderadas por la máscara gaussiana. Esto significa que
    siguiente cálculo para cada capa de la pirámide:
      salida[i, j] = máscara_actual[i, j] * imagen_blanca[i, j] +
                   (1 - máscara_actual[i, j]) * imagen_negra[i, j]
    Por lo tanto:
      Los píxeles en los que máscara_actual == 1 deben tomarse completamente de la imagen blanca.
      blanca.
      Los píxeles en los que máscara_actual == 0 deben tomarse completamente de la imagen negra.
      negra.

    Nota: máscara_actual, imagen_blanca e imagen_negra son variables que se refieren a la imagen de la capa actual que
    estamos viendo a la imagen de la capa actual que estamos viendo. Este
    cálculo para cada capa de la pirámide.
    """

    blended_pyr = []
    for lapl_white, lapl_black, gauss_mask in \
            zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
        blended_pyr.append(gauss_mask * lapl_white +
                           (1 - gauss_mask) * lapl_black)

    plt.imshow(blended_pyr[0])
    plt.show()
    return blended_pyr


def collapse(pyramid):
    """ Colapsa una pirámide de entrada.

    Args:
      pyramid (list): Una lista de imágenes numpy.ndarray. Se puede asumir que la entrada
            se toma de blend() o lapl_pyramid().

    Devuelve:
      output(numpy.ndarray): Una imagen de la misma forma que la capa base de
            la pirámide y dtype float.

    Plantea este problema de la siguiente manera, empieza por la capa más pequeña de la
    pirámide. Expande la capa más pequeña y añádela a la segunda capa más pequeña.
    más pequeña. Luego, expanda la segunda a la capa más pequeña, y continúe el proceso
    hasta llegar a la imagen más grande. Este es el resultado.

        Nota: a veces expandir devolverá una imagen más grande que la siguiente
    siguiente. En este caso, debe recortar la imagen expandida hasta el tamaño de la siguiente capa.
    la siguiente capa. Mira en numpy slicing / lee nuestro README para hacer esto
    fácilmente.

    Por ejemplo, expandir una capa de tamaño 3x4 resultará en una imagen de tamaño
    6x8. Si la siguiente capa es de tamaño 5x7, recorta la imagen expandida a tamaño 5x7.
    """
    output = pyramid[-1]
    for image in reversed(pyramid[:-1]):
        output = image + expand(output)[:image.shape[0], :image.shape[1]]

    return output

def generating_kernel(parameter):
    """ Devuelve un kernel generador 5x5 basado en un parámetro de entrada.
     Nota: Esta función se proporciona para ti, no la cambies.
     Args:
         parameter (float): Rango de valor: [0, 1].
     output:
        numpy.ndarray: Un núcleo de 5x5.
     """
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                       0.25, 0.25 - parameter / 2.0])
    return np.outer(kernel, kernel)


def reduce_img(image):
    """ Convoluciona la imagen de entrada con un kernel generador de parámetro de 0.4
        y luego reducir su anchura y altura por dos.
        Puedes utilizar cualquiera / todas las funciones para convolucionar y reducir la imagen, aunque
        las conferencias han recomendado métodos que aconsejamos ya que hay un montón de
        de piezas en esta tarea que necesitan trabajar 'justo a la derecha'.
        Args:
        image (numpy.ndarray): una imagen en escala de grises de forma (r, c)
        Devuelve:
        output (numpy.ndarray): una imagen de la forma (ceil(r/2), ceil(c/2))
          Por ejemplo, si la entrada es 5x7, la salida será 3x4.

    """
    # según las instrucciones, utilice 0.4 para la generación del kernel
    kernel = generating_kernel(0.4)

    # usa convolve2d con la imagen y el kernel enviados
    output = scipy.signal.convolve2d(image, kernel, 'same')

    # devuelve cada dos líneas y filas
    return output[:output.shape[0]:2, :output.shape[1]:2]


def expand(image):
    """ Expandir la imagen al doble de tamaño y luego convolucionarla con un
    kernel generador con un parámetro de 0.4.

    Deberías aumentar el tamaño de la imagen y luego convolucionarla con un kernel generador
    kernel generador de a = 0,4.

    Por último, multiplique la imagen de salida por un factor de 4 para volver a escalarla.
    escala. Si no hace esto (y le recomiendo que lo pruebe sin
    esto) verá que sus imágenes se oscurecen al aplicar la convolución.
    Por favor, explica por qué ocurre esto en tu PDF de presentación.

    Por favor, consulte las conferencias y readme para una discusión más a fondo de
    cómo abordar la función de expansión.

    Puede utilizar cualquier / todas las funciones para convolucionar y reducir la imagen, aunque
    las conferencias han recomendado métodos que aconsejamos ya que hay un montón de
    piezas de esta tarea que tienen que funcionar "a la perfección".

    Args:
    image (numpy.ndarray): una imagen en escala de grises de forma (r, c)

    Devuelve:
    output (numpy.ndarray): una imagen de la forma (2*r, 2*c)
    """
    # según las instrucciones, usa 0.4 para la generación del kernel
    kernel = generating_kernel(0.4)

    # hacer un nuevo array del doble de tamaño, asignar valores iniciales
    output = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    output[:output.shape[0]:2, :output.shape[1]:2] = image

    # usa convolve2d para rellenar el resto
    # multiplicar por 4 por instrucciones para volver a escalar
    output = scipy.signal.convolve2d(output, kernel, 'same') * 4
    return output



def main():
    """Dadas las dos imágenes, mézclalas según la máscara"""

    sourcefolder = 'images/images_tilt/original'
    outfolder = 'images/images_tilt/output'

    # si no encuentra el directorio de salida, lo crea
    if os.path.isdir(outfolder):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)

    # Mediante el uso de la función get_images, usándola como un Iterador gracias a Yield,
    # va mos a cargar de cada imagen original su máscara y su blur, cargándola en las 3 variables
    for photo, white_img, black_img, mask_img in get_images(sourcefolder):
        imshow("Original Image", black_img)
        print("...applying blending")
        black_img = black_img.astype(float)
        white_img = white_img.astype(float)
        mask_img = mask_img.astype(float) / 255

        # inicializa la salida
        out_layers = []

        # para cada canal de color (RGB) llama a run_blend (mezcla las imágenes según la máscara)
        for channel in range(3):
            outimg = run_blend(black_img[:, :, channel],
                               white_img[:, :, channel],
                               mask_img[:, :, channel])
            out_layers.append(outimg)

        # la salida es la fusión de cada canal ya tratado
        outimg = cv2.merge(out_layers)

        # escribe en la carpeta de salida la imagen ya calculada
        cv2.imwrite(os.path.join(outfolder, photo), outimg)
        imshow("Tilt Shift Effect", outimg)
        print('...[DONE]')

if __name__ == "__main__":
    main()

