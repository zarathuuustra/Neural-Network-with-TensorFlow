import numpy as np

# Capitalizing the first letter of class names is documented in Python's PEP8 coding style.
class Variable:
    def __init__(self, data):
        self.data = data

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
        return output

    def forward(self, x):

        """
        forward propagation
        :param x:
        :return:
        """
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)