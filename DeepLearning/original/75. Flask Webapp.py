#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
# 
# # **Flask WebApp**

# In[1]:


get_ipython().system('git clone https://github.com/rajeevratan84/pytorch-flask-api-heroku.git')


# In[2]:


get_ipython().system('pip install flask-ngrok')


# In[3]:


get_ipython().run_line_magic('cd', 'pytorch-flask-api-heroku/')


# In[4]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz')


# In[5]:


get_ipython().system('curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok')


# ## **Sign up for NGROK**
# 
# 

# In[6]:


get_ipython().system('ngrok authtoken 22HGRYrSB3NseGQDhfMETArVWFi_4DE4QrBHdkFPUQnu8rDEn')


# In[ ]:


import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction
from commons import format_class_name
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()


# In[ ]:




