
# ******************************************
# ***** 16 Usando la camara en OpenCV
# ******************************************
import cv2
import sys

# especificamos un índice de dispositivo de cámara predeterminado de cero.
s = 0
print(sys.argv)  # contiene los argumentos de la librería sys, por ejemplo 0 es la ruta
# ['C:\\Users\\jgomcano\\PycharmProjects\\guiapython\\OpenCV\\Usando la camara en openCV\\16 Usando_camara_OpenCV.py']
# y simplemente estamos verificando si hubo una especificación de línea de comando para anular ese valor predeterminado.
if len(sys.argv) > 1:
    s = sys.argv[1]
print(s)  # 0
source = cv2.VideoCapture(s)  # llamamos a la clase de captura de video para crear un objeto de captura de video,
#  Con el índice 0 accederá a la cámara predeterminada en su sistema, si no hay que indicarlo
win_name = 'Vista de camara'
# estamos creando una ventana con nombre, que eventualmente vamos a enviar la salida transmitida
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

'''ciclo while nos permitirá transmitir continuamente video desde la cámara y enviarlo a la salida a menos que el 
usuario pulse la tecla de escape.'''
while cv2.waitKey(1) != 27:  # Escape
    '''usa esa fuente de objeto de captura de vídeo  de captura de video para llamar al método read, que  devolverá un 
    solo cuadro de la transmisión de video, así como una variable lógica has_frame.
    Entonces, si hay algún tipo de problema con la lectura de la transmisión de video o el acceso a la cámara, entonces 
    has_frame sería falso y saldríamos del bucle.
    De lo contrario, continuaríamos y llamaríamos a la función de visualización de mensajes instantáneos y abriríamos
     kbps para enviar el video (frame) a la ventana de salida'''
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)

### generar un boceto

# Nuestra función generadora de bocetos
def sketch(image):
    # Convierte la imagen a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Limpia la imagen usando Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Extraer bordes
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # Invertir y binarizar la imagen
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


# Inicializar webcam, cap es el objeto proporcionado por VideoCapture
cap = cv2.VideoCapture(0)

while True:
    # Contiene un booleano indicando si tuvo éxito (ret)
    # También contiene las imágenes recogidas de la webcam (frame)
    ret, frame = cap.read()
    # Pasamos nuestro frame a nuestra función sketch directamente dentro de cv2.imshow()
    cv2.imshow('Nuestro dibujante en vivo', sketch(frame))
    if cv2.waitKey(1) == 13:  # 13 es la tecla Enter
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()