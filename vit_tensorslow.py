import TensorSlow

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from common.nlp_util import preprocess
from TensorSlow.core import Function
from TensorSlow.core import Variable, Parameter
import TensorSlow.transforms
import TensorSlow.functions as F
import matplotlib.pyplot as plt
from TensorSlow.utils import sum_to
import TensorSlow.layers as L  # import as L
from TensorSlow import Layer
from TensorSlow import optimizers
from TensorSlow.models import MLP
from TensorSlow import DataLoader
from TensorSlow.datasets import Spiral
from common.nlp_util import (preprocess, create_contexts_target,convert_one_hot, MatMul, SoftmaxWithLoss,
                             create_contexts_target, convert_one_hot, Trainer, Adam)
import math

# ------------------------

# Training a Vision Transformer with the CIFAR10 dataset

## Setting the hyperparameters
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

# Define Image Transformations

transform = TensorSlow.transforms.Compose([
    TensorSlow.transforms.ToArray(),
    TensorSlow.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Getting a dataset
train_set = TensorSlow.datasets.CIFAR10(train=True)
test_set = TensorSlow.datasets.CIFAR10(train=False)

print(len(train_set))
print(len(test_set))

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)

print(f"DataLoader: {train_loader, test_loader}")
print(f"Length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}...")
print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}...")


class PatchEmbedding(Layer):

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size

        self.proj = L.Linear(
            out_size=embed_dim,
            in_size=patch_dim
        )

    def forward(self, x):

        patches = F.im2col(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad=0,
            to_matrix=True
        )

        embeddings = self.proj(patches)

        return embeddings

#TODO: (Question) Do I need to move the Vision Transformer to the CPU? How does it work here?

# # model = MLP((hidden_size, 10))
# model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
# optimizer = optimizers.SGD().setup(model)  # default lr=0.01
#
#
# for epoch in range(max_epoch):
#     sum_loss, sum_acc = 0, 0
#
#     for x, t in train_loader:
#         y = model(x)
#         loss = F.softmax_cross_entropy(y, t)
#         acc = F.accuracy(y, t)
#         model.cleargrads()
#         loss.backward()
#         optimizer.update()
#
#         sum_loss += float(loss.data) * len(t)
#         sum_acc += float(acc.data) * len(t)
#
#     print('epoch: {}'.format(epoch + 1))
#     print('train loss: {:.4f}, accuracy: {:.2f}'.format(
#         sum_loss / len(train_set), sum_acc / len(train_set)))
#
#     sum_loss, sum_acc = 0, 0
#     with TensorSlow.no_grad():
#         for x, t in test_loader:
#             y = model(x)
#             loss = F.softmax_cross_entropy(y, t)
#             acc = F.accuracy(y, t)
#
#             sum_loss += float(loss.data) * len(t)
#             sum_acc += float(acc.data) * len(t)
#
#     print('test loss: {:.4f}, accuracy: {:.2f}'.format(
#         sum_loss / len(test_set), sum_acc / len(test_set)))