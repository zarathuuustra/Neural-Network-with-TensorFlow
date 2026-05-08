import numpy as np
from TensorSlow.core import Function, as_variable

###### Sinus ################
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x) # might be problematic
        return gx

def sin(x):
    return Sin()(x)


######## Cosinus ##############
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

############ Tanh - hyperbolic tangent ###################
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)

########## Tensor operations ##################

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    """
    To make sure that the shape of the variable data is consistent with the shape of the gradient
    :param x: input variable
    :param shape: shape of the variable
    :return:
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)