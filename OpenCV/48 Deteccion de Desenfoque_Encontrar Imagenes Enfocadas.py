# *************************************************************
# ***** 48 Detección de Desenfoque_Encontrar Imágenes Enfocadas
# *************************************************************


# ### **Para Detectar Desenfoque, simplemente Convolvemos con el kernel Laplaciano.**
#
# Tomamos la escala de grises de una imagen y la convolucionamos con el kernel Laplaciano (kernel 3 x 3):
#
# Para cuantificar el desenfoque, entonces tomamos la varianza de la salida de respuesta.
#
# El Laplaciano es la 2ª derivada de una imagen y, por tanto, resalta las áreas de una imagen que contienen cambios
# rápidos de intensidad. De ahí su uso en la detección de bordes. Una varianza alta debería, en teoría, indicar la
# presencia tanto de bordes como de no bordes (de ahí el amplio rango de valores que resulta en una varianza alta),
# lo que es típico de una imagen normal enfocada.
#
# Una varianza baja, por lo tanto, podría significar que hay muy pocos bordes en la imagen, lo que significa que podría
# estar borrosa, ya que cuanto más borrosa esté, menos bordes habrá. #



import cv2
from matplotlib import pyplot as plt

# Define nuestra función imshow
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()



# #### **producir some Blurred Images**

# Cargar nuestra imagen de entrada
image = cv2.imread('./images/liberty.jpeg')
imshow("Original Image", image)

blur_1 = cv2.GaussianBlur(image, (5,5), 0)
imshow('Blurred Image 1', blur_1) 

blur_2 = cv2.GaussianBlur(image, (9,9), 0)
imshow('Blurred Image 2', blur_2) 

blur_3 = cv2.GaussianBlur(image, (13,13), 0)
imshow('Blurred Image 3', blur_3) 


def getBlurScore(image):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cv2.Laplacian(image, cv2.CV_64F).var()


# #### **Mostrar nuestras puntuaciones, ¡recuerda que más alto significa menos desenfoque!**

print("Blur Score = {}".format(getBlurScore(image)))
print("Blur Score = {}".format(getBlurScore(blur_1)))
print("Blur Score = {}".format(getBlurScore(blur_2)))
print("Blur Score = {}".format(getBlurScore(blur_3)))





