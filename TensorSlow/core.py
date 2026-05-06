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
        self.creator = None

    def set_creator(self, func):  # added the creator of the variable (function or non-function)
        self.creator = func

    def backward(self):
        # for the user to omit y.grad = np.array(1.0)
        if self.grad is None:  # added
            self.grad = np.ones_like(self.data) # creates a derivative-> =1

        funcs = [self.creator]  # use list to record functions
        while funcs:
            f = funcs.pop()  # 1. Get a function
            gys = [output.grad for output in f.outputs] # Summarise the derivatives of the output variables in the list
            gxs = f.backward(*gys)    # call backpropagation and unwrap the list
            if not isinstance(gxs, tuple):  # transform gxs into tuple if it is not
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs): # set the derivate in the backpropagation to the grad variable
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator) # 4. add previous functions to the list


class Function:
    def __call__(self, *inputs): # Asterisk for any number of arguments
        """
        Retrieves data from the Variable and saving the calculation results to the Variable.
        :param inputs:
        :return:
        """
        xs = [x.data for x in inputs] # to support multiple inputs and outputs
        ys = self.forward(*xs)  # concrete calculation is implemented in forward method
        if not isinstance(ys, tuple):  # added
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]   # wrap data

        for output in outputs: # loops for creator records
            output.set_creator(self)  # Set parent(function); let the output variable save its creator
        self.inputs = inputs  # save input variables
        self.outputs = outputs  # Set output
        return outputs if len(outputs) > 1 else outputs[0]

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
        x = self.inputs[0].data
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
    """
    To make square to a python function
    :param x:
    :return:
    """
    f = Square()
    return f(x)

def exp(x):
    """
    to make exponential to a python function
    :param x:
    :return:
    """
    return Exp()(x)

def add(x0, x1):
    """
    to make addition to a python function
    :param x0:
    :param x1:
    :return:
    """
    return Add()(x0, x1)

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


class Add(Function):
    """
    Performs the addition of two variables
    """
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)