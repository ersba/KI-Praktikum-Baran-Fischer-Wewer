import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.wildfire_dir = os.path.join(root_dir, "wildfire")
        self.nowildfire_dir = os.path.join(root_dir, "nowildfire")
        self.list_files = os.listdir(self.wildfire_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        wildfire_path = os.path.join(self.wildfire_dir, img_file)
        nowildfire_path = os.path.join(self.nowildfire_dir, img_file)

        wildfire_image = np.array(Image.open(wildfire_path).convert("RGB"))
        nowildfire_image = np.array(Image.open(nowildfire_path).convert("RGB"))

        augmentations = config.both_transform(image=nowildfire_image, image0=wildfire_image)
        nowildfire_image = augmentations["image"]
        wildfire_image = augmentations["image0"]

        nowildfire_image = config.transform_only_input(image=nowildfire_image)["image"]
        wildfire_image = config.transform_only_mask(image=wildfire_image)["image"]

        return nowildfire_image, wildfire_image

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    dataset = MapDataset(os.path.join(current_folder, "..", "actual_dataset", "train"))
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
