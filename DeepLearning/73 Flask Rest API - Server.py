#!/usr/bin/env python
# codificación: utf-8

#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Servidor Flask RestAPI**

# En 1]:


get_ipython().system('git clone https://github.com/rajeevratan84/pytorch-flask-api.git')


# ## **Instalar ngrok**
#
# ngrok es una aplicación multiplataforma que expone los puertos del servidor local a Internet.

# En[6]:


get_ipython().system('pip install flask-ngrok')


# En 2]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz')


# En[10]:


get_ipython().system('curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok')


# Enlace de registro: https://dashboard.ngrok.com/signup?

# En[11]:


get_ipython().system('ngrok authtoken YOUR_TOKEN')


# En[4]:


get_ipython().run_line_magic('cd', 'pytorch-flask-api/')


# En[ ]:


import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)

imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()


# ## **Abra una nueva ventana de Colab y ejecute los siguientes bloques**
#
# 1. Obtenga nuestra imagen de prueba.
# ```
# !wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/bird.jpg
# ```
#
# 2. Envíe nuestras solicitudes de publicación de API. Reemplace la URL a continuación con la URL de NGROK anterior.
# ```
# solicitudes de importación
# resp = solicitudes.post("http://xxxxxxxxxxxx.ngrok.io/predict",
# archivos={"archivo": abierto('pájaro.jpg','rb')})
# imprimir(resp.json())
# ```
#
#
#
