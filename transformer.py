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

## 4. Setting the hyperparameters

BATCH_SIZE = 16 # CPU optimiert 32 # instead of 128
EPOCHS = 10
LEARNING_RATE = 2e-3 # CPU optimiert 1e-3 # instead of 3e-4
PATCH_SIZE = 4 # like tokens
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3 # 3 color channels
EMBED_DIM = 64 # Cpu optimiert 128 # instead of 256
NUM_HEADS = 2 # CPU optimiert 4 # instead of 8
DEPTH = 2 # CPU optimiert 3 # instead of 6
MLP_DIM = 128 # CPU optimiert 256 # instead of 512
DROP_RATE = 0.2 # instead of 0.1

# 5. Define Image Transformations

transform = transforms.Compose([
    transforms.ToTensor(), # we need to convert our image to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the values of the tensors to process them more efficiently
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
                 embed_dim):

        super().__init__()

        # super() macht hier: - Parameter speichern; Gradienten berechnen
        # .to(device) verwenden; Optimizer benutzen

        self.patch_size = patch_size  # # Convolution 2D layer;
        # kernel= feature detector; a small, learnable matrix or tensor used to extract features from input data

        # stride = controls the step size of the convolution window (kernel) as it slides across the input tensor.
        # It determines how many pixels the filter shifts horizontally and vertically after each operation
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # special vector we want to train in the training phase
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # to make sure the shapes align -> (B, N, E)
        cls_token = self.cls_token.expand(B, -1, -1)
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

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first = True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward (self, x):
        # Self-Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Sequential -> when the data goes through the Sequential class it will go through it Layer by Layer
        self.encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_dim, drop_rate)
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

# Instantiate model

# specified parameters; in order;
model = VisionTransformer(
    IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES,
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE
).to(device) # move to the target device
print(model)

## 9. Defining a Loss function and optimizer

criterion = nn.CrossEntropyLoss() # Measure how wrong our model is
optimizer = torch.optim.AdamW(params=model.parameters(), # update our models paraments
                             lr = LEARNING_RATE)


## 10. Defining a Training Loop function

def train(model, loader, optimizer, criterion):
    # Set the mode of the model into training
    model.train()

    total_loss, correct = 0, 0
    # x = batch of images/photos; y = batch of labels or targets
    for x,y in loader:
        # Moving (sending) our data to target device
        x, y = x.to(device), y.to(device) # load to cuda device
        optimizer.zero_grad()
        # 1. Forward pass (model outputs raw logits)
        out = model(x) # batch of pictures
        # 2. Calculate the loss (per batch)
        loss = criterion(out, y)
        # 3. Perform backpropagation
        loss.backward()
        # 4. Perform Gradient Descent
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct = correct + (out.argmax(1) == y).sum().item() # Fehlerquelle?

    # You have to scale the loss (Normalization step to make the loss general across all batches
    return total_loss / len(loader.dataset), correct / len(loader.dataset )

def evaluate(model, loader):
    model.eval() # Set the mode of the model into evaluation
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)  # move to target device
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
        return correct / len(loader.dataset)

## Training
# from tqdm.auto import tqdm
train_accuracies , test_accuracies = [], []

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_acc)
    print(f"Epoch: {epoch+1}/{EPOCHS}, train loss {train_loss:.3f}, "
          f"Train accuracy {train_accuracy:.4f}%, Test accuracy: {test_acc:.4f} ")