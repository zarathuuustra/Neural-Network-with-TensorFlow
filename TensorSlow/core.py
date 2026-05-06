import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # add the corresponding gradient value
        self.creator = None  # added

    def set_creator(self, func):  # added the creator of the variable (function or non-function)
        self.creator = func

    def backward(self):
        f = self.creator  # 1. get the function
        if f is not None:
            x = f.input  # 2. get the function's input
            x.grad = f.backward(self.grad)  # 3. call the function's backward method
            x.backward()  # call the previous Variable's backward method (recursively)


class Function:
    def __call__(self, input):
        """
        Retrieves data from the Variable and saving the calculation results to the Variable.
        :param input:
        :return:
        """
        x = input.data
        y = self.forward(x)  # concrete calculation is implemented in forward method
        output = Variable(y)
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# backpropagation
y.grad = np.array(1.0)
y.backward()
print(x.grad)  # 3.297442541400256; same with previous code