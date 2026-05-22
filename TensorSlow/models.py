import TensorSlow.functions as F
import TensorSlow.layers as L
from TensorSlow.layers import Layer
from TensorSlow import utils
import numpy as np


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(Model):
    """Multi-Layer Perceptron."""
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class SelfAttention(Layer):

    def __init__(self, embed_dim):

        super().__init__()

        self.embed_dim = embed_dim

        self.query = L.Linear(
            out_size=embed_dim,
            in_size=embed_dim
        )

        self.key = L.Linear(
            out_size=embed_dim,
            in_size=embed_dim
        )

        self.value = L.Linear(
            out_size=embed_dim,
            in_size=embed_dim
        )

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        K_t = F.transpose(K, (0, 2, 1))
        scores = F.batch_matmul(Q, K_t)
        scores = scores / np.sqrt(self.embed_dim)
        attention = F.softmax(scores, axis=-1)
        output = F.batch_matmul(attention, V)
        return output
