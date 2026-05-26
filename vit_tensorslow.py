import TensorSlow

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
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
from TensorSlow.optimizers import Adam as Ada
from TensorSlow.models import MLP, SelfAttention
from TensorSlow import DataLoader
from TensorSlow.dataloaders import SeqDataLoader
from common.nlp_util import (preprocess, create_contexts_target,convert_one_hot, MatMul, SoftmaxWithLoss,
                             create_contexts_target, convert_one_hot, Trainer, Adam)
import math

# ------------------------

# Training a Vision Transformer with the CIFAR10 dataset

## Setting the hyperparameters
BATCH_SIZE = 16 # CPU optimiert 32 # instead of 128
EPOCHS = 10
LEARNING_RATE = 1e-3 # Beim Bauen genutzt: 2e-3 # Im Video: 3e-4
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

x, y = train_set[0]

print("X-Typ:", x.dtype)
print("X min:", x.min())
print("X max:", x.max())

labels = []

for i in range(100):
    _, y = test_set[i]
    labels.append(y)

labels = np.array(labels)

print(labels.min())
print(labels.max())
print(np.unique(labels))

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
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)

        self.pos_embed = Parameter(np.random.randn(1, 1 + self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        patches = F.im2col(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad=0,
            to_matrix=True
        )

        embeddings = self.proj(patches)
        b = x.shape[0]
        embeddings = F.reshape(
            embeddings,
            (b, self.num_patches, self.embed_dim)
        )

        cls_tokens = F.broadcast_to(
            self.cls_token,
            (b, 1, self.embed_dim)
        )

        x = F.concat((cls_tokens, embeddings), axis=1)

        x = x + self.pos_embed # .data hier (falls es nicht funktioniert)
        # langfristig problematisch, da ich dadurch Gradienten und Autograd verliere

        return x


class TransformerMLP(Layer):
    def __init__(self,
                 in_features,
                 hidden_features):

        super().__init__()

        self.fc1 = L.Linear(
            out_size=hidden_features,
            in_size=in_features
        )

        self.fc2 = L.Linear(
            out_size=in_features,
            in_size=hidden_features
        )

    def forward(self, x):

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class TransformerEncoder(Layer):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 drop_rate):

        super().__init__()
        self.norm1 = L.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim)
        self.norm2 = L.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(
            in_features=embed_dim,
            hidden_features=mlp_dim
        )

    def forward(self, x):
        # print("TransformerEncoder, vor attn_out, Wert:", self.attn)
        # print("TransformerEncoder, vor attn_out, Typ:", type(self.attn))
        attn_out = self.attn(self.norm1(x))
        # print("TransformerEncoder, nach attn_out, Wert:", attn_out.creator)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x

class VisionTransformer(Layer):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Sequential -> when the data goes through the Sequential class it will go through it Layer by Layer
        self.encoders = []
        for i in range (depth):
            encoder = TransformerEncoder(
                embed_dim,
                num_heads,
                mlp_dim,
                drop_rate
            )
            setattr(self, f"encoder_{i}", encoder)
            self.encoders.append(encoder)
        self.norm = L.LayerNorm(embed_dim)
        self.head = L.Linear(
            out_size=num_classes,
            in_size=embed_dim
        )  # act as a classifier

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

model = VisionTransformer(
    IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES,
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE
)# move to the target device

print("Das hier ist das Modell", model)
# for p in model.params():
#     print(p.shape)

## 9. Defining a Loss function and optimizer

# Measure how wrong our model is
criterion = F.softmax_cross_entropy

# update our models paraments
optimizer = Ada().setup(model)
# Da TensorSlow aktuell kein AdamW-Optimizer hat wird derzeit der Adam-Optimizer benutzt

## 10. Defining a Training Loop function

train_accuracies = []
test_accuracies = []
losses = []

for epoch in range(EPOCHS):

    sum_loss = 0
    sum_acc = 0

    # TRAINING
    for x, t in train_loader:

        # Forward pass
        y = model(x)

        # Loss berechnen
        loss = F.softmax_cross_entropy(y, t)

        # Accuracy berechnen
        acc = F.accuracy(y, t)

        # Gradienten zurücksetzen
        model.cleargrads()

        # Backpropagation
        loss.backward()

        # print("Der Erschaffer von W ist:", model.encoders[0].attn.query.W.creator)
        # print("Der Gradient von W (patch_embed) ist:", model.patch_embed.proj.W.grad)
        # print("Der Gradient von W (attn, query) ist:", model.encoders[0].attn.query.W.grad)
        # print("Der Gradient von W (attn, key) ist:", model.encoders[0].attn.key.W.grad)
        # print("Der Gradient von W (attn, value) ist:", model.encoders[0].attn.value.W.grad)
        # print("Der Gradient W (model, head) ist:", model.head.W.grad)
        # print("Der Gradient des CLS_Token ist:", model.patch_embed.cls_token.grad) # for testing purposes

        # Parameter updaten
        # for param in model.params():
        #     if param.grad is not None:
        #         print("PARAM:", param)
        #         print("DTYPE:", param.grad.data.dtype)
        #         print()
        #
        # params_dict = {}
        # model._flatten_params(params_dict)

        # for name, param in params_dict.items():
        #     if param.grad is not None:
        #         print(name, param.grad.data.dtype)

        optimizer.update()

        # Statistik sammeln
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss = sum_loss / len(train_set)
    train_acc = sum_acc / len(train_set)

    train_accuracies.append(train_acc)
    losses.append(train_loss)

    # TEST / EVALUATION
    sum_test_acc = 0

    with TensorSlow.no_grad():

        for x, t in test_loader:
            y = model(x)
            acc = F.accuracy(y, t)
            sum_test_acc += float(acc.data) * len(t)

    test_acc = sum_test_acc / len(test_set)
    test_accuracies.append(test_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, "
        f"Test Acc: {test_acc:.4f}"
    )

epochs = range(1, len(losses) + 1)

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(epochs, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# TODO: Es fehlen: MultiHeadAttention & Dropout
