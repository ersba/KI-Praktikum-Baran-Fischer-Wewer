from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class WildfireDataset(Dataset):
    def __init__(self, root_wildfire, root_nowildfire, transform=None):
        self.root_wildfire = root_wildfire
        self.root_nowildfire = root_nowildfire
        self.transform = transform

        self.wildfire_images = os.listdir(root_wildfire)
        self.nowildfire_images = os.listdir(root_nowildfire)
        self.length_dataset = max(len(self.wildfire_images), len(self.nowildfire_images)) # 1000, 1500
        self.wildfire_len = len(self.wildfire_images)
        self.nowildfire_len = len(self.nowildfire_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        wildfire_img = self.wildfire_images[index % self.wildfire_len]
        nowildfire_img = self.nowildfire_images[index % self.nowildfire_len]

        wildfire_path = os.path.join(self.root_wildfire, wildfire_img)
        nowildfire_path = os.path.join(self.root_nowildfire, nowildfire_img)

        wildfire_img = np.array(Image.open(wildfire_path).convert("RGB"))
        nowildfire_img = np.array(Image.open(nowildfire_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=wildfire_img, image0=nowildfire_img)
            wildfire_img = augmentations["image"]
            nowildfire_img = augmentations["image0"]

        return wildfire_img, nowildfire_img
