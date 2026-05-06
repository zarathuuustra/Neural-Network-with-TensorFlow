import numpy as np
import unittest

class Variable:
    """
    Variable class that only supports data from `ndarray` instances
    """
    def __init__(self, data):
        if data is not None:   # added
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None # add the corresponding gradient value
        self.creator = None  # added

    def set_creator(self, func):  # added the creator of the variable (function or non-function)
        self.creator = func

    def backward(self):
        # for the user to omit y.grad = np.array(1.0)
        if self.grad is None:  # added
            self.grad = np.ones_like(self.data) # creates a derivative-> =1

        funcs = [self.creator]  # use list to record functions
        while funcs:
            f = funcs.pop()  # 1. Get a function
            x, y = f.input, f.output  # 2. Get the function's input/output
            x.grad = f.backward(y.grad)  # 3. Call the function's backward

            if x.creator is not None:
                funcs.append(x.creator) # 4. add previous functions to the list


class Function:
    def __call__(self, input):
        """
        Retrieves data from the Variable and saving the calculation results to the Variable.
        :param input:
        :return:
        """
        x = input.data
        y = self.forward(x)  # concrete calculation is implemented in forward method
        output = Variable(as_array(y))
        output.set_creator(self)  # Set parent(function); let the output variable save its creator
        self.input = input  # save input variables
        self.output = output  # Set output
        return output

    def forward(self, x):
        """
        forward propagation of the base class
        :param x:
        :return:
        """
        raise NotImplementedError()

    def backward(self, gy):  # added
        """
        backward propagation of the base class
        :param gy:
        :return:
        """
        raise NotImplementedError()

class Square(Function):
    """
    Inherits from the Function class AND squares the input of the values
    """

    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    """
    Inherits from the Function class AND exponentiates the input of the values
    """

    def forward(self, x):
        y = np.exp(x)
        return y

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

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    """
    If the value of the input is a scalar, convert it to an array.
    Otherwise do nothing.
    :param x: the converted value of the input
    :return:
    """
    if np.isscalar(x):
        return np.array(x)
    return x
