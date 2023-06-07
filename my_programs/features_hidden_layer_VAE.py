# we are going to create a dataset that contains the representation 
# in a latent space of the VAE 
# of the images from the test set of the celebA dataset 

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image



folder = 'C:\\Users\\Roberto\\Desktop\\celebA'
# folder of images
test_folder = folder + '\\Img\\img_align_celeba\\img_align_celeba\\'

# the test set are the images from 182638.jpg to 202599.jpg

# load the weights of the VAE


# Load the pretrained DCGAN-VAE model
model = models.dcgan_vae(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

# Load and preprocess the input image
image = Image.open("path/to/your/image.jpg")
input_tensor = transform(image).unsqueeze(0)

# Encode the input image into a latent representation
with torch.no_grad():
    latent_representation, _, _ = model.encode(input_tensor)

# Print the latent representation
print(latent_representation)
