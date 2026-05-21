from TensorSlow.core import Parameter, Variable
from TensorSlow import cuda
from TensorSlow.utils import pair
import TensorSlow.functions as F
import weakref
import numpy as np
import os



# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


#########################################################################
# class LSTM(Layer):

#    def __init__(self, hidden_size, in_size=None):
#        super().__init__()
#    # 1. define initialized variables (hidden_size, in_size)
#    # 2. define weights in different gates (recall formulas): (x2f, x2i, x2o, x2u; h2f, h2i, h2o, h2u)


#        self.reset_state()

#    def reset_state(self):
#    # 3.reset h and c


#    def forward(self, x):
#    # 4.define how f, i, o, u are calculated
#        if self.h is None:

#        else:

#    # 5.define how c_new is calculated
#        if self.c is None:

#        else:
#
#    # 6. define how h_new is calculated

#    # 7. update h, c
#
#        return h_new
#################################################################################

class LSTM(Layer):
    """LSTM-Layer \n
    x = linear transformation of the input depending on the gate \n
    h = linear transformation of the hidden_states depending on the gate; carries information for the current timestep t \n
    -> short term memory\n
    b = bias depending on the gate, added to linear transformations \n
    c = cell state; saves information over multiple time steps \n
    -> long term memory \n
    hidden_size = amount of neurons in the hidden state (h) \n
    in_size = input size -> how big matrix of the input is (number of features in the input matrix)
    """

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size

        # Define weights for each gate: forget (f), input (i), output (o), and candidate (u).
        self.x2f = Linear(hidden_size, in_size=in_size, nobias=True)  # Forget gate weights for the input (x).
        self.h2f = Linear(hidden_size, nobias=True)  # Forget gate weights for the hidden state (h)
        self.b_f = Parameter(np.zeros((1, hidden_size)))  # Forget gate bias

        self.x2i = Linear(hidden_size, in_size=in_size, nobias=True)  # Input gate weights for the input (x).
        self.h2i = Linear(hidden_size, nobias=True)  # Input gate weights for the hidden state (h)
        self.b_i = Parameter(np.zeros((1, hidden_size)))  # Input gate bias

        self.x2o = Linear(hidden_size, in_size=in_size, nobias=True)  # Output gate weight for the input (x)
        self.h2o = Linear(hidden_size, nobias=True)  # Output gate weight for the hidden state (h)
        self.b_o = Parameter(np.zeros((1, hidden_size)))  # Output gate bias

        self.x2u = Linear(hidden_size, in_size=in_size, nobias=True)  # candidate gate weight for the input (x)
        self.h2u = Linear(hidden_size, nobias=True)  # candidate gate weight for the hidden state (h)
        self.b_u = Parameter(np.zeros((1, hidden_size)))  # candidate gate bias

        self.reset_state()

    def reset_state(self):
        """Reset the hidden (h) and cell (c) states of the LSTM. This ensures that the layer starts with a clean state, \n
        with no memory of previous sequences.
        """
        self.h = None  # Short term memory ("working memory")
        self.c = None  # Long term memory (internal)

    def forward(self, x):
        if self.h is None:
            # Initialize hidden and cell states with zeros for the current batch.
            batch_size = x.shape[0]
            self.h = np.zeros((batch_size, self.hidden_size), dtype=x.dtype)
            self.c = np.zeros((batch_size, self.hidden_size), dtype=x.dtype)

        # Forget gate: Determines how much of the past cell state (c) to retain.
        f = F.sigmoid(self.x2f(x) + self.h2f(self.h) + self.b_f)
        # Input gate: Decides how much new information to add to the cell state.
        i = F.sigmoid(self.x2i(x) + self.h2i(self.h) + self.b_i)
        # Output gate: Controls the amount of information from the cell state to expose.
        o = F.sigmoid(self.x2o(x) + self.h2o(self.h) + self.b_o)
        # Candidate gate: Creates new candidate values to potentially update the cell state.
        u = F.tanh(self.x2u(x) + self.h2u(self.h) + self.b_u)

        # Update cell state: Combine retained memory (f * c) with new input (i * u).
        self.c = f * self.c + i * u
        # Update hidden state: Controlled by the output gate (o) and cell state (c)
        self.h = o * F.tanh(self.c)

        return self.h


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)

class LayerNorm(Layer):

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(embed_dim))
        self.beta = Parameter(np.zeros(embed_dim))

    def forward(self, x):

        mean = F.mean(
            x,
            axis=-1,
            keepdims=True
        )

        var = F.mean(
            (x - mean) ** 2,
            axis=-1,
            keepdims=True
        )

        x_hat = (x - mean) / F.sqrt(var + self.eps)
        y = self.gamma * x_hat + self.beta

        return y