import torch
import matplotlib.pyplot as plt
from models.generator import Generator
import subprocess

generator = Generator()
generator.load_state_dict(torch.load("results/model.pth")["generator"])
generator.eval()

noise = torch.randn(100, 100, 1, 1)
fake_images = generator(noise).detach().numpy().transpose(0, 2, 3, 1)

fig, axes = plt.subplots(10, 10, figsize=(10,10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow((fake_images[i] + 1) / 2)
    ax.axis("off")

plt.savefig("results/generated_grid.png")


#selecting 1000 random images
image_paths = np.array(glob('data/bitmojis/*'))
image_paths = image_paths[np.random.randint(0, len(image_paths), 1000)]
for i in image_paths:
    image = transform.resize(io.imread(i), (64,64,3))
    io.imsave("results/FID/real/"+i.split('/')[1])

#generating 1000 images
noise = torch.FloatTensor(np.random.randn(1000,100, 1, 1))
generated_img = generator(noise).numpy().reshape(1000,64,64,3)
for i in range(len(generated_img)):
    io.imsave('results/FID/generated/'+str(i)+'.png')


subprocess.run(["python", "-m", "pytorch_fid", "FID/real", "FID/generated"], check=True)