import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # add the corresponding gradient value


class Function:
    def __call__(self, input):
        """
        Retrieves data from the Variable and saving the calculation results to the Variable.
        :param input:
        :return:
        """
        x = input.data
        y = self.forward(x)  # concrete calculation is implemented in forward method
        self.input = input  # save input variables
        output = Variable(y)
        return output

    def forward(self, x):
        """
        forward propagation
        :param x:
        :return:
        """
        raise NotImplementedError()

    def backward(self, gy):  # added
        raise NotImplementedError()

class Square(Function):
    """
    Inherits from the Function class AND squares the input of the values
    """

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    """
    Inherits from the Function class AND exponentiates the input of the values
    """

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    """
    Numerical Differentiation; derivative that represents the rate of change, which is defined as the amount of change
    in a very short period of time.
    :param f: Function
    :param x: Variable
    :param eps: Very small value to calculate the value without causing an error
    :return:
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)