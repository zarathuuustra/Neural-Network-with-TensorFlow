import numpy as np

# Capitalizing the first letter of class names is documented in Python's PEP8 coding style.
class Variable:
    def __init__(self, data):
        self.data = data

# This example uses NumPy's multidimensional arrays to store data.
# Machine learning systems use multidimensional arrays as the basic data structure.
# x is an instance of Variable. The value is assigned to x.
data = np.array(1.0)
x = Variable(data)
print(x.data)

class Function:
    def __call__(self, input):
        x = input.data  # read data
        y = x ** 2  # calculation
        output = Variable(y)  # return as a Variable
        return output