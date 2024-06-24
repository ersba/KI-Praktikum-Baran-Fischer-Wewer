import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import config  # Assuming your config file contains necessary parameters
import torch.nn.functional as F

# Define Block class
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

# Define Generator class (same as in the training script)
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        print(f"d1: {d1.shape}, d2: {d2.shape}, d3: {d3.shape}, d4: {d4.shape}, d5: {d5.shape}, d6: {d6.shape}, d7: {d7.shape}, bottleneck: {bottleneck.shape}")

        up1 = self.up1(bottleneck)
        print(f"up1: {up1.shape}")
        up2 = self.up2(torch.cat([up1, F.interpolate(d7, up1.shape[2:])], 1))
        print(f"up2: {up2.shape}")
        up3 = self.up3(torch.cat([up2, F.interpolate(d6, up2.shape[2:])], 1))
        print(f"up3: {up3.shape}")
        up4 = self.up4(torch.cat([up3, F.interpolate(d5, up3.shape[2:])], 1))
        print(f"up4: {up4.shape}")
        up5 = self.up5(torch.cat([up4, F.interpolate(d4, up4.shape[2:])], 1))
        print(f"up5: {up5.shape}")
        up6 = self.up6(torch.cat([up5, F.interpolate(d3, up5.shape[2:])], 1))
        print(f"up6: {up6.shape}")
        up7 = self.up7(torch.cat([up6, F.interpolate(d2, up6.shape[2:])], 1))
        print(f"up7: {up7.shape}")
        return self.final_up(torch.cat([up7, F.interpolate(d1, up7.shape[2:])], 1))

# Parameters (same as in the training script)
image_size = config.IMAGE_SIZE  # Update with your image size
# Define the path to the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the images directory within the Test_Model directory
images_dir = os.path.join(current_dir, 'images', 'original')

# Get the list of files in the images directory
image_files = os.listdir(images_dir)

# Filter the list to only include files with .jpg extension
image_files = [file for file in image_files if file.endswith('.jpg')]

# Sort the files (optional, depending on how you want to define "first")
image_files.sort()

# Define the relative path to the gen.pth file within the Test_Model directory
checkpoint_path = os.path.join(current_dir, 'gen.pth')

print(f"Images directory: {images_dir}")
print(f"Checkpoint path: {checkpoint_path}")

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the generator
netG = Generator(in_channels=config.CHANNELS_IMG, features=64).to(device)

# Load the weights
checkpoint = torch.load(checkpoint_path, map_location=device)
netG.load_state_dict(checkpoint['state_dict'])

# Set the generator to evaluation mode
netG.eval()

# Define the transformations (same as in the training script)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
generated_images_dir = os.path.join(current_dir, 'images', 'generated')

# Create the generated images directory if it doesn't exist
os.makedirs(generated_images_dir, exist_ok=True)
# Iterate over all images in the folder
for image_file in image_files:
    test_image_path = os.path.join(images_dir, image_file)

    # Open the image
    image = Image.open(test_image_path).convert('RGB')

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate the fake image
    with torch.no_grad():
        fake_image = netG(image_tensor).detach().cpu()

    # Plot the original and generated images
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(np.transpose(image_tensor.squeeze().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)  # De-normalize

    # Generated image
    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(np.transpose(fake_image.squeeze().numpy(), (1, 2, 0)) * 0.5 + 0.5)  # De-normalize

    # Save the plot instead of showing it
    save_path = os.path.join(generated_images_dir, f'generated_vs_original_{os.path.splitext(image_file)[0]}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved generated image comparison to {save_path}")
          
