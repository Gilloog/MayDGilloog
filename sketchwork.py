#piecing out the code so that this just gets the sketch images working, then we can throw this back to the model.py file to train it and whatnot
#all this code is in model.py im using this file to see if i can get the images turned into sketches properly (im not getting it yet)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import numpy as np
import os
torch.manual_seed(42)

epochs = 50
batch_size = 64
learning_rate = 0.0002
image_size = 64
desired_width = 128
desired_height = 128


root_path = "/home/gilloog@campus.sunyit.edu/FinalProj/MayDGilloog/CelebA"
image_path = "/home/gilloog@campus.sunyit.edu/FinalProj/MayDGilloog/CelebA/img_align_celeba"
sketch_path = "/home/gilloog@campus.sunyit.edu/FinalProj/MayDGilloog/Sketches"

transform = transforms.Compose([
    transforms.Resize((desired_width, desired_height)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=root_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
def preprocess_celeba_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sketch_images = []

    for i, image in enumerate(images):
        image_np = np.array(image.permute(1, 2, 0).cpu().numpy())
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        blurred_image = cv2.GaussianBlur(gray_image, (0, 0), 3)
        sketch_image = cv2.subtract(gray_image, blurred_image)
        sketch_image = sketch_image / 255.0
        sketch_image = np.expand_dims(sketch_image, axis=2)
        sketch_image = np.concatenate([sketch_image] * 3, axis=2)
        sketch_image = torch.from_numpy(sketch_image).permute(2, 0, 1).float()
        save_path = os.path.join(output_folder, f"sketch_{i}.png")
        save_image(sketch_image, save_path)
        sketch_images.append(sketch_image)

    return sketch_images

def generate_and_save_sketches(dataset, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (real_images, _) in enumerate(dataset):
        preprocess_celeba_images(real_images, output_folder)

generate_and_save_sketches(dataloader, sketch_path)
            