import cv2
import numpy as np
import matplotlib.pyplot as plt
# ************************************************************
# ***** Características de la imagen y alineación de la imagen
# ************************************************************
'''
Demostraremos los pasos a través de un ejemplo en el que alinearemos una foto de un formulario tomado con un teléfono 
móvil con una plantilla del formulario. La técnica que usaremos a menudo se denomina alineación de imágenes "basada en 
funciones" porque en esta técnica se detecta un conjunto escaso de funciones en una imagen y se compara con las 
funciones en la otra imagen. Luego se calcula una transformación basada en estas características combinadas que deforma 
una imagen sobre la otra.

La alineación de imágenes (también conocida como registro de imágenes) es la técnica de deformar una imagen (o, a veces,
ambas imágenes) para que las características de las dos imágenes se alineen perfectamente.
'''

# **** Paso 1: Lea la plantilla y la imagen escaneada
# Leemos la imagen de referencia
refFilename = "form.jpg"
print("Reading reference image : ", refFilename)
im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

# leemos la imagen que queremos alinear
imFilename = "scanned-form.jpg"
print("Reading image to align : ", imFilename)
im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Mostramos las imágenes cargadas
plt.figure(figsize=[20,10]);
plt.subplot(121); plt.axis('off'); plt.imshow(im1); plt.title("Original Form")
plt.subplot(122); plt.axis('off'); plt.imshow(im2); plt.title("Scanned Form")
plt.show()


# ****** Paso 2: encuentra puntos clave en ambas imágenes
'''
objetivo  es tratar de extraer información significativa que esté contextualmente relacionada con la imagen en sí.
Por lo general, buscamos bordes, esquinas y texturas en las imágenes, las función orb() es una forma de hacerlo, 
vamos a crear este objeto orbe, y luego vamos a usar ese objeto para detectar y calcular puntos clave y descriptores 
para cada una de las imágenes.

Entonces, los puntos clave son características interesantes en cada imagen que generalmente se asocian con algunos 
puntos nítidos. borde o esquina, y están descritos por un conjunto de coordenadas de píxeles que describen la ubicación
del punto clave. El tamaño del punto clave. En otras palabras, la escala del punto clave y luego también la orientación 
del punto clave. luego hay una lista asociada de descriptores para cada punto clave, y cada descriptor es en realidad un
vector de alguna información que describe la región alrededor del punto clave, que actúa efectivamente como una firma 
para ese punto clave. Es una representación vectorial de la información de píxeles alrededor del punto clave. Y la idea 
aquí es que si estamos buscando el mismo punto clave en ambas imágenes, podemos intentar usar los descriptores para 
emparejarlos.'''

# Convertimos las imágenes a escala de grises
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detecta características de ORB y calcula descriptores.

MAX_NUM_FEATURES = 500
'''El algoritmo utilizado para la detección de características de la imagen dada junto con la orientación y los 
descriptores de la imagen se denomina algoritmo ORB y es una combinación del detector de punto clave FAST y el 
descriptor BRIEF.

- Localizador : identifica puntos en la imagen que son estables bajo transformaciones de imagen como traslación 
  (desplazamiento), escala (aumento/disminución de tamaño) y rotación. El localizador encuentra las coordenadas x, y de 
  dichos puntos. El localizador que utiliza el detector ORB se llama FAST .
- Descriptor : El localizador del paso anterior solo nos dice dónde están los puntos interesantes. La segunda parte del 
  detector de características es el descriptor que codifica la apariencia del punto para que podamos distinguir un punto
  característico de otro. El descriptor evaluado en un punto característico es simplemente una matriz de números. 
  Idealmente, el mismo punto físico en dos imágenes debería tener el mismo descriptor. ORB usa una versión modificada 
  del descriptor de características llamado BRISK .
  
sintaxis 
ORB_object = cv.ORB_create()
keypoints = ORB_object.detect(input_image)
keypoints, descriptors = ORB_object.compute(input_image, keypoints)

- El algoritmo ORB se puede implementar usando una función llamada función ORB().
- La implementación del algoritmo ORB funciona creando un objeto de la función ORB().
- Luego hacemos uso de una función llamada función ORB_object.detect() para detectar los puntos clave de una imagen dada
- Luego hacemos uso de una función llamada función ORB_object.compute() para calcular los descriptores de una imagen 
  determinada.
- Luego, la imagen con los puntos clave calculados dibujados en la imagen se devuelve como salida
https://www.educba.com/opencv-orb/
https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/


'''
orb = cv2.ORB_create(MAX_NUM_FEATURES)

# detectAndCompute aúna las dos explicadas anteriormente
keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

'''Estamos dibujando los puntos clave detectados en la imagen usando la función drawKeypoints()
Sintaxis de la función drawKeypoints():
dibujar puntos clave (imagen_de_entrada, puntos_clave, imagen_de_salida, color, bandera)
parámetros:
- input_image: la imagen que se convierte en escala de grises y luego los puntos clave se extraen utilizando los 
                algoritmos SURF o SIFT se denomina imagen de entrada.
- key_points: los puntos clave obtenidos de la imagen de entrada después de usar los algoritmos se denominan puntos 
              clave.
- output_image :   imagen sobre la que se dibujan los puntos clave.
- color : el color de los puntos clave.
- bandera: las características del dibujo están representadas por la bandera.
https://www.geeksforgeeks.org/python-opencv-drawkeypoints-fuction/
'''
im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255, 0, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255, 0, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

'''
Hemos calculado los puntos clave calculados en los descriptores de cada imagen. Y aquí, en estas cifras, se muestran 
solo los puntos clave.  todos estos círculos rojos son puntos clave. El centro del círculo es la ubicación del punto 
clave. El tamaño del círculo representa la escala del punto clave y luego la línea que conecta el centro del círculo al 
exterior del círculo representa la orientación del punto clave. Hay algunos puntos clave en ambas imágenes que tal vez 
sean los mismos, y esos son los que vamos a tratar de encontrar para que podamos calcular el gráfico de Hamas entre 
estas dos representaciones de imágenes.'''

plt.figure(figsize=[20,10]);
plt.subplot(121); plt.axis('off'); plt.imshow(im1_display); plt.title("Original Form")
plt.subplot(122); plt.axis('off'); plt.imshow(im2_display); plt.title("Scanned Form")
plt.show()

# **** Paso 3: haga coincidir los puntos clave en las dos imágenes
'''
El primer paso en este proceso de coincidencia es crear una coincidencia u objeto llamando a DescriptorMatcher_create.
le pasamos a esa función DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, una medida de distancia (los descriptores de o cadena 
binaria requieren una métrica de hamming para ese objetivo). lo que hace es  Toma el descriptor de una característica 
en el primer conjunto y se compara con todas las demás características en el segundo conjunto utilizando algún cálculo 
de distancia. Y se devuelve el más cercano.

luego usamos esa coincidencia u objeto para llamar a la función de match, que luego intenta proporcionar una lista de 
las mejores coincidencias asociadas con esa lista de descriptores. tenemos una estructura de datos  que contiene la 
lista de coincidencias de los puntos clave que determinamos arriba.

Y luego, una vez que obtengamos esa lista, ordenaremos la lista en función de la distancia entre los distintos, tras lo 
que vamos a limitar al 10 por ciento superior de las coincidencias devueltas.

https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''
# Coincidir las características encontradas en ambas imágenes.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# match ()para obtener las mejores coincidencias en dos imágenes.
matches = matcher.match(descriptors1, descriptors2, None)
'''
el resutlado de la línea línea 162 es una lista de objetos DMatch. Este objeto DMatch tiene los siguientes atributos:
DMatch.distance - Distancia entre descriptores. Cuanto más bajo, mejor.
DMatch.trainIdx - Índice del descriptor en descriptores de train
DMatch.queryIdx - índice del descriptor en los descriptores de consulta
DMatch.imgIdx - Índice de la imagen de train.
'''
# ordenar las coincidencias por resultado ascendentemente
matches = sorted(matches, key=lambda x: x.distance, reverse=False)  # al ser una tupla sort no.


# Eliminar las coincidencias menos favorables, quedándonos solo con el 10%
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

'''
Y vamos a usar DrewMatches para dibujar las coincidencias en este código, puedes ver que varios puntos clave en una 
imagen coinciden los puntos clave de la otra imagen'''
# Dibujar las mejores coincidencias aportando las dos imágenes, sus puntos y las coincidencias
im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

plt.figure(figsize=[40,10])
plt.imshow(im_matches); plt.axis('off'); plt.title("Original Form");
plt.show()

# **** Paso 4: Encuentra la homografía
'''
¿Qué es la Homografía?
Considere dos imágenes de un plano con un libro en diferentes posiciones y distancia.  Si el libro tiene un cuadro con 
una imagen, un punto en la esquina del cuadro representa el mismo punto en las dos imágenes. En la jerga de la visión 
artificial, llamamos a estos puntos correspondientes. Una homografía es una transformación (una matriz de 3×3) que 
asigna los puntos de una imagen a los puntos correspondientes de la otra imagen.

Si conociéramos la homografía, podríamos aplicarla a todos los píxeles de una imagen para obtener una imagen 
deformada que esté alineada con la segunda imagen, es decir , puede aplicar la homografía a la primera imagen y el libro
de la primera imagen se alineará con el libro de la segunda imagen. Si conocemos 4 o más puntos correspondientes en las
dos imágenes, podemos usar la función de OpenCV findHomography para encontrar la homografía

h, status = cv2.findHomography(points1, points2)
donde, puntos1 y puntos2 son vectores/matrices de puntos correspondientes, y h es la matriz homográfica.'''

# Extraer ubicación de las buenas coincidencias
'''Crea y devuelve una referencia a un array con las dimensiones especificadas en la tupla dimensiones cuyos elementos 
son todos ceros básicamente está creando un array de arrays con los puntos inicializados a 0, con el número de puntos 
por la longitud que tiene el objeto matches
'''
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

'''Recorre matches desde la primera posición va introducendo el valor de los puntos de los descriptores de match de 
entrenamiento y consulta'''
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Encuentra la homografía
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)


# ***** Paso 5: deformar la imagen
# Usar homografía para deformar la imagen
height, width, channels = im1.shape  # desmpaquetamos la dimensión de la imagen de referencia

''' la transformación de perspectva está asociada con el cambio de punto de vista. Este tipo de transformación no
conserva el paralelismo, la longitud y el ángulo pero conserva la colinealidad y la incidencia, lo que significa que 
las líneas rectas permanecerán rectas despues de la transformación. 

para ello seleccionamos 4 puntos de la imagen de entrada y asignamos esos 4 puntos a las ubicaciones deseadas en la 
imagen de salida, realizando

dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]] )
# src: imagen de entrada
# M: Matriz de transformación, en este caso usamos la homografía como esa matriz
# dsize: tamaño de la imagen de salida (ancho, alto)
# flags: método de interpolación a utilizar
https://theailearner.com/tag/cv2-warpperspective/'''

im2_reg = cv2.warpPerspective(im2, h, (width, height))
# Display results
plt.figure(figsize=[20,10]);
plt.subplot(121); plt.imshow(im1); plt.axis('off'); plt.title("Original Form");
plt.subplot(122); plt.imshow(im2_reg); plt.axis('off'); plt.title("Scanned Form");
plt.show()