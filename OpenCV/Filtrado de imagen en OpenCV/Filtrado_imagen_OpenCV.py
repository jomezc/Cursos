import cv2
import sys
import numpy

# **********************************
# ***** Filtrado de imagen en OpenCV
# **********************************

PREVIEW  = 0   # Vista previa
BLUR     = 1   # filtro de desenfoque
FEATURES = 2   # Detector de características de corner
CANNY    = 3   # Detector de borde astuto

# Estamos definiendo un pequeño diccionario de configuración de parámetros para el detector de características de corner
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)

'''Estamos configurando el índice del dispositivo para la cámara (linea22), creando una ventana de salida para los 
resultados transmitidos (30)y luego crea un objeto de captura de video (33) para que podamos procesar la transmisión de 
video en el bucle (36)'''
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()  # leemos el frame de vídeo
    if not has_frame:
        break

    frame = cv2.flip(frame,1)  # mediante flip giramos el video horizontalmente

    if image_filter == PREVIEW:  # según la configuración de ejecución del script (línea 27)
        result = frame  # solo cogemos el frame y lo mostramos
    elif image_filter == CANNY:
        '''
        Detector de bordes Canny con OpenCV
        La función Canny() en OpenCV se utiliza para detectar los bordes de una imagen
        canny = cv2.Canny(imagen, umbral_minimo, umbral_maximo)
        Donde:
        - canny: es la imagen resultante. Aparecerán los bordes detectados tras el proceso.
        - imagen: es la imagen original.
        - umbral_minimo: es el umbral mínimo en la umbralización por histéresis
        - umbral_maximo: es el umbral máximo en la umbralización por histéresis
        hay mas parámetros: 
        - opening_size: Tamaño de apertura del filtro Sobel.
        - L2Gradient: Parámetro booleano utilizado para mayor precisión en el cálculo de Edge Gradient.
        el umbral mínimo y el máximo dependerá de cada situación.
        Docu 
        https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html'''
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        '''
        método cv2.blur()
        El método se utiliza para desenfocar una imagen utilizando el filtro de cuadro normalizado. La función suaviza 
        una imagen.
        Sintaxis: cv2.blur(src, ksize[, dst[, ancla[, borderType]]])
        Parámetros:
        - src: Es la imagen de la que se desea difuminar.
        - ksize: una tupla que representa el tamaño del kernel de desenfoque, es decir  son las dimensiones del núcleo 
            de la caja. En este ejemplo sería un kernel de caja de 13 por 13 que estaría involucrado con la imagen para 
            dar como resultado una imagen borrosa. si el tamaño del kernel es más pequeño que el desenfoque, se reduce, 
            si el tamaño del kernel es más grande se obtiene un desenfoque más sustancial.
        - dst: Es la imagen de salida del mismo tamaño y tipo que src.
        - ancla: es una variable de tipo entero que representa el punto de anclaje y su valor predeterminado es (-1, -1)
          ,lo que significa que el ancla está en el centro del kernel.
        - borderType: representa qué tipo de borde se agregará. Está definido por indicadores como cv2.BORDER_CONSTANT 
          , cv2.BORDER_REFLECT , etc.
        - Valor devuelto: Devuelve una imagen.
        '''
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertimos la imagen a escala de grises
        '''La función goodFeaturesToTrack encuentra N esquinas más fuertes 
         cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, [,mask[,blockSize[,useHarrisDetector[,k]]]])

        - imagen: entrada de imagen de un solo canal de 8 bits o punto flotante de 32 bits
        - maxCorners - Número máximo de esquinas a devolver. Si hay más esquinas de las que se encuentran, se devuelve 
          la más fuerte de ellas. si <= 0 implica que no se establece ningún límite en el máximo y se devuelven todas 
          las esquinas detectadas.
        - qualityLevel - Parámetro que caracteriza la calidad mínima aceptada de las esquinas de la imagen. Consulte el 
          párrafo anterior para obtener una explicación.
        - minDistance - Distancia euclidiana mínima posible entre las esquinas devueltas
        - máscara - Región de interés opcional. Si la imagen no está vacía, especifica la región en la que se detectan 
          las esquinas.
        - blockSize - Tamaño de un bloque promedio para calcular una matriz de covariación derivada sobre cada 
          vecindario de píxeles
        - useHarrisDetector - ya sea para usar Shi-Tomasi o Harris Corner
        -k - Parámetro libre del detector de Harris
        Documentación: 
        https://theailearner.com/tag/cv2-goodfeaturestotrack/
        https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html'''
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:  # Devuelve una lista de esquinas encontradas en la imagen
            ''' Y si detectamos una o más esquinas, simplemente anotaremos el resultado con pequeños
             círculos verdes para indicar las ubicaciones de esas características ojo con los parámetros
             al ser capturados las  posiciones x,y PASARLO a entero'''
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    cv2.imshow(win_name, result)  # Enviamos el resultado a la salida

    # para poder cambiar el tratamiento de la imagen dependiendo de la tecla introducida
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)
