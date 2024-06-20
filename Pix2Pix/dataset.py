import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :350, :]
        target_image = image[:, 350:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

def save_normalized_image(img, path):
    # Normierung in den Bereich [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    # Konvertierung zu float
    img = img.float()
    save_image(img, path)


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    dataset = MapDataset(os.path.join(current_folder, "data"))
    print(current_folder)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
