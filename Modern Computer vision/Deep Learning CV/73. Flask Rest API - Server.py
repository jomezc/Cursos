#!/usr/bin/env python
# coding: utf-8

# 
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Flask RestAPI Server**

# In[1]:


get_ipython().system('git clone https://github.com/rajeevratan84/pytorch-flask-api.git')


# ## **Install ngrok**
# 
# ngrok is cross-platform application that exposes local server ports to the Internet.

# In[6]:


get_ipython().system('pip install flask-ngrok')


# In[2]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz')


# In[10]:


get_ipython().system('curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok')


# Sign up link - https://dashboard.ngrok.com/signup?

# In[11]:


get_ipython().system('ngrok authtoken YOUR_TOKEN')


# In[4]:


get_ipython().run_line_magic('cd', 'pytorch-flask-api/')


# In[ ]:


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


# ## **Open new Colab Window and run the following blocks**
# 
# 1. Get our test image.
# ```
# !wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/bird.jpg
# ```
# 
# 2. Send our API Post Requests. Replace the URL below with the NGROK URL above.
# ```
# import requests
# resp = requests.post("http://xxxxxxxxxxxx.ngrok.io/predict",
#                      files={"file": open('bird.jpg','rb')})
# print(resp.json())
# ```
# 
# 
# 
