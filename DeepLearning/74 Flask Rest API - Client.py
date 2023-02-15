#!/usr/bin/env python
# codificación: utf-8

#
# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Servidor Flask RestAPI**

# En 3]:


get_ipython().system('wget https://raw.githubusercontent.com/rajeevratan84/ModernComputerVision/main/bird.jpg')


# En[4]:


import requests

resp = requests.post("http://f6eb-34-71-59-16.ngrok.io/predict",
                     files={"file": open('bird.jpg','rb')})

print(resp.json())

