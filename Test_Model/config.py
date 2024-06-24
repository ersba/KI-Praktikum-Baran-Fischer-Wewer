import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'actual_dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'actual_dataset', 'val')
TEST_DIR = os.path.join(BASE_DIR, 'actual_dataset', 'test')
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth"
CHECKPOINT_GEN = "gen.pth"

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)
