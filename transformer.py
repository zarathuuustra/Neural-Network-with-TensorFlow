import torch
import torch.nn as nn
import torch.nn.functional as F # layers, loss functions, so
import torch.optim as optim
from torch.utils.data import DataLoader # batchfile
from torchvision import datasets, transforms
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)

## 2. Setup Device-Agnostic Code

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## 3. Set the seed

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

## 3. Setting the hyperparameters

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE = 4 # like tokens
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3 # 3 color channels
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1

# 5. Define Image Transformations