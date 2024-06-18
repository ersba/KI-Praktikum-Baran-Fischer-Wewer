
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# Define Generator class (same as in the training script)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Parameters (same as in the training script)
nz = 100  # Size of z latent vector (same as in the training script)
ngf = 64  # Size of feature maps in generator (same as in the training script)
nc = 3    # Number of channels in the training images. For RGB this is 3
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
image_size = 350

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the generator
netG = Generator(ngpu).to(device)

# Load the weights
netG.load_state_dict(torch.load("generator.pth"))

# Set the generator to evaluation mode
netG.eval()

# Load a test image
test_image_path = "D:/neu_studium/Master_Semester_01/KI/Actual_dataset/test/nowildfire/-61.33042,51.79317.jpg"  # Update with your test image path

# Define the transformations (same as in the training script)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Open the image
image = Image.open(test_image_path).convert('RGB')

# Apply transformations
image_tensor = transform(image).unsqueeze(0).to(device)

# Generate random noise
noise = torch.randn(1, nz, 1, 1, device=device)

# Generate the fake image
with torch.no_grad():
    fake_image = netG(noise).detach().cpu()

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
plt.savefig('generated_vs_original.png')
