import torch
import torch.nn as nn
import torch.optim as optim
from models  import Generator
from models  import Discriminator
from utils import BitmojiDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import yaml 

f = open("./config/config.yaml", 'r',encoding="utf-8" )
config_dict = yaml.load(f.read(),Loader=yaml.SafeLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

data = BitmojiDataset(f"data/bitmojis", config_dict["image_size"])
dataloader = DataLoader(data, config_dict["batch_size"], shuffle=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

real_label = 1
fake_label = 0

G_losses, D_losses = [], []

for epoch in range(epochs):
    for batch in tqdm.tqdm(dataloader):
        real_images = batch.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        output_real = discriminator(real_images).view(-1, 1)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_images).view(-1, 1)
        loss_G = criterion(output_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")


checkpoint = {
    'generator' : generator.state_dict(),
    'discriminator' : discriminator.state_dict(),
    'optimizerG' : optimizer_G.state_dict(),
    'optimizerD' : optimizer_D.state_dict(),
} 

torch.save(checkpoint , "results/model.pth")