#!/usr/bin/env python
# codificación: utf-8

# ![](https://github.com/rajeevratan84/ModernComputerVision/raw/main/logo_MCV_W.png)
#
# # **Redes adversarias generativas (GAN) en PyTorch: GAN convolucionales profundas o DCGAN con MNIST**
#cle
# ---
#
# En esta lección, aprenderemos a usar PyTorch para crear un DCGAN simple usando el conjunto de datos MNIST.

### **1. Configuración de nuestros datos y módulos**

# En 1]:


import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(42)


### **2. Obtenga nuestro conjunto de datos MNIST usando torchvision y cree nuestras transformaciones y el cargador de datos**

# En 2]:


batch_size = 32

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                               
train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


# ### **Ahora representemos algunos de nuestros datos originales (reales)**

# En 3]:


samples, labels = next(iter(train_loader))

for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap="gray")
    plt.xticks([])
    plt.yticks([])


### **4. Definir nuestro Modelo Discriminador**
#
# Nuestro Descriminador recibe la imagen de 28x28 generada por nuestro Generador y genera una puntuación de probabilidad de que pertenezca o no al conjunto de datos original.



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # No usamos capas Conv aquí pero vectorizamos nuestras entradas
            nn.Linear(784, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

# Instanciamos nuestro modo y lo enviamos a la GPU
discriminator = Discriminator().to(device=device)


### **5. Definir nuestro Modelo de Generador**


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),  # Usamos la función de activación Tanh() para que nuestras salidas estén entre -1 y 1
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output

# Instanciamos nuestro modo y lo enviamos a la GPU
generator = Generator().to(device=device)


# ## **Establecer nuestros parámetros de entrenamiento**



# Establecer nuestra tasa de aprendizaje, épocas
lr = 0.0001
epochs = 25
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)




for epoch in range(epochs):
    for n, (samples, labels) in enumerate(train_loader):
        # Obtener datos para entrenar al discriminador
        real_samples = samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)  # los reales son un 1 , un ok que es real
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)  # generamos vectores aleatorios
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)  # los generados son 0 son falsas
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))  # los real samples son 1 queremos generarlo todo verdadero

        # Entrenando al discriminador
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Datos para entrenar el generador
        # el vector necesita ese vector 1x100 para generar esa muestra , por lo que creamos esos vectores aleatorios de
        # ese tamaño
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

        # Entrenando al generador
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Mostrar pérdida
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Descrimianted Loss: {loss_discriminator}")
            print(f"Epoch: {epoch} Generator Loss: {loss_generator}")


### **5. Ahora inspeccionemos nuestras muestras generadas**



latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)




generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    # También tenemos que remodelarlo (reshape), porque el generador produce un vector 784, así que tenemos
    # que traerlo de vuelta a la Forma de veintiocho por 28 píxeles.
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.show()




