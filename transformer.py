"""
Building a Vision Transformer from Scratch:
1. Reading the paper "Attention is all you need"
2. Building a Vision Transformer with a YouTube-Video
3. Apply the concepts of the video on the framework used in the seminar
4. Write the paper for the seminar "Build Your Own Neural Network" by the end of May

Sources:
Building a Vision Transformer Model from Scratch with PyTorch
Video on YouTube: https://www.youtube.com/watch?v=7o1jpvapaT0
GitHub page is: https://github.com/MOHAMMEDFAHD/Pytorch-Collections/tree/main/vision-transformer
"""



import torch
import torch.nn as nn
import torch.nn.functional as F # layers, loss functions, so
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
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

transform = transforms.Compose([
    transforms.ToTensor(), # we need to convert our image to tensors
    transforms.Normalize((0.5,), (0.5,))
    # 1. Helps the model to converge faster
    # 2. Helps to make he numerical computations stable
])

# 6. Getting a dataset

train_dataset = datasets.CIFAR10(root='data',
                                 train=True,
                                 transform=transform,
                                 download=True)

# print(train_dataset)

test_dataset = datasets.CIFAR10(root='data',
                                train=False,
                                download=True,
                                transform=transform)

# print(test_dataset)

print(len(train_dataset))
print(len(test_dataset))

# 7.Converting our datasets into dataloaders
# right now our data is in the form of PyTorch Datasets
# Dataloader turns our data into batcher or mini-batches

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

print(f"DataLoader: {train_loader, test_loader}")
print(f"Length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}...")
print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}...")

# 8. Building Vision Transformer Model from Scratch

class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim ):
        super().__init__()
        self.patch_size = patch_size # # Convolution 2D layer;
        # kernel= feature detector; stride = steps, how many pixels you want to move over
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_dim))
        # special vector we want to train in the training phase
        self.pos_embed = nn.Parameter(torch.randn(1,1 + num_patches), embed_dim)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # to make sure the shapes align -> (B, N, E)
        cls_token = self.cls_token( B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=in_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x): # gelu: activation function
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class TransformerEncoder (nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first = True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward (self, x):
        x = x + self.attn(self.norm1(x), self.norm2(x), self.norm1(x)[0])
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Sequential -> when the data through the Sequential class it will go through it Layer by Layer
        self.encoder = nn.Sequential([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # act as a classifier

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

