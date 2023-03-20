import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BitmojiDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.jpg') or file.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

