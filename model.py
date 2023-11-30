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

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

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
            
generator = Generator(in_channels=3, out_channels=3)
discriminator = Discriminator(in_channels=6)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

for epoch in range(epochs):
    for i, (real_images, _) in tqdm(enumerate(dataloader)):
        real_images = real_images.cuda()
        target_sketches = preprocess_celeba_images(image_path)

        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).cuda()
        real_pair = torch.cat((real_images, target_sketches), 1)
        real_outputs = discriminator(real_pair)
        d_loss_real = criterion_GAN(real_outputs, real_labels)

        fake_images = generator(target_sketches)
        fake_labels = torch.zeros(batch_size, 1).cuda()
        fake_pair = torch.cat((fake_images.detach(), target_sketches), 1)
        fake_outputs = discriminator(fake_pair)
        d_loss_fake = criterion_GAN(fake_outputs, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        validity = discriminator(torch.cat((fake_images, target_sketches), 1))
        real_labels = torch.ones(batch_size, 1).cuda()
        g_loss_GAN = criterion_GAN(validity, real_labels)

        g_loss_L1 = criterion_L1(fake_images, real_images)
        lambda_L1 = 0.01
        g_loss = g_loss_GAN + lambda_L1 * g_loss_L1
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss GAN: {g_loss_GAN.item()}] [G loss L1: {g_loss_L1.item()}]")

    save_image(fake_images, f"output_images/celebA_generated_{epoch}.png")
