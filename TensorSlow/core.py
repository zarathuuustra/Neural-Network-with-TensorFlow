import numpy as np
import weakref
import contextlib
import TensorSlow

class Config:  # a way to enable/disable backpropagation mode
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """
    A way to use the context manager with the enable backpropagation mode
    :return:
    """
    return using_config('enable_backprop', False)


class Variable:
    """
    Variable class that only supports data from `ndarray` instances
    """
    __array_priority__ = 200  # added priority (larger than 0 and 10)

    def __init__(self, data, name=None):
        if data is not None:  # added
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None  # add the corresponding gradient value
        self.creator = None
        self.generation = 0  # to get the correct order/priority

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):  # print
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):  # added the creator of the variable (function or non-function)
        self.creator = func
        self.generation = func.generation + 1  # the generaion advances if there is a creator

    def cleargrad(self):  # resets the derivatives stored in the variable.
        self.grad = None

    def reshape(self, *shape):
        """
        Adjusts the shape parameter of the variable
        :param shape:
        :return:
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
           shape = shape[0]
        return TensorSlow.functions.reshape(self, shape)

    def transpose(self, *axes): #might be buggy because of the axes problem
        """
        Transpose method
        :param axes: Number of axes
        :return:
        """
        if len(axes) == 0:
           axes = None
        elif len(axes) == 1:
           if isinstance(axes[0], (tuple, list)) or axes[0] is None:
              axes = axes[0]
        return TensorSlow.functions.transpose(self, axes)

    @property
    def T(self): # Instance variable
       return TensorSlow.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return TensorSlow.functions.sum(self, axis, keepdims)

    def backward(self, retain_grad=False, create_graph=False):
        """

        :param retain_grad: if true = gradients retain derivates;
        if false = derivates of intermediate variables is reset
        :param create_graph: if true = create graph
        :return:
        """
        # for the user to omit y.grad = np.array(1.0)
        if self.grad is None:  # added
            # self.grad = np.ones_like(self.data)  # creates a derivative-> =1
            self.grad = Variable(np.ones_like(self.data)) # reference to a Variable instance

        funcs = []
        seen_set = set()  # purpose = to prevent the same function from being added to the list more than once

        def add_func(f):
            """
            list of functions will be sorted by generation
            :param f:
            :return:
            """
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 1. Get a function

            ###### backpropagation ################
            gys = [output().grad for output in f.outputs]  # Summarise the derivatives of the output variables in the list
            # the output is weakref

            with using_config ('enable_backprop', create_graph):
                gxs = f.backward(*gys)  # call backpropagation and unwrap the list
                if not isinstance(gxs, tuple):  # transform gxs into tuple if it is not
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):  # set the derivate in the backpropagation to the grad variable
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)  # 4. add previous functions to the list

                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None  # y is weakref


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


def as_variable(obj):
    """
    So that variable instances can be used with ndarrays
    :param obj:
    :return:
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):  # Asterisk for any number of arguments
        """
        Retrieves data from the Variable and saving the calculation results to the Variable.
        :param inputs:
        :return:
        """
        inputs = [as_variable(input) for input in inputs]
        ####### forward propagation ##############
        xs = [x.data for x in inputs]  # to support multiple inputs and outputs
        ys = self.forward(*xs)  # concrete calculation is implemented in forward method
        if not isinstance(ys, tuple):  # added
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # wrap data

        if Config.enable_backprop:  # if backpropagation is active
            self.generation = max([x.generation for x in inputs])  # set generations
        #### create links ##############
            for output in outputs:  # loops for creator records
                output.set_creator(self)  # Set parent(function); let the output variable save its creator; reference
            self.inputs = inputs  # save input variables
            self.outputs = [weakref.ref(output) for output in outputs]  # weak reference to exclude a circular reference

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):  # forward function of the base class
        raise NotImplementedError()

    def backward(self, gys):  # backward function of the base class
        raise NotImplementedError()


########## Multiplication ###########
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0


def mul(x0, x1):
    """
    Multiplication in TensorSlow
    :param x0:
    :param x1:
    :return:
    """
    x1 = as_array(x1)
    return Mul()(x0, x1)


############# Division ###################
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)  # swap x1 and  x0


########## Square ###############
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


def square(x):
    """
    To make square to a python function
    :param x:
    :return:
    """
    return Square()(x)


########### Exp ######################
class Exp(Function):
    """
    Applies the natural exponential function to the input.
    """

    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    """
    to make exponential to a python function
    :param x:
    :return:
    """
    return Exp()(x)

########### Power operator ###############
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)


######### Numerical differentiation - in contrast to automatic backpropagation #####
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


############ Addition ######################
class Add(Function):
    """
    Performs the addition of two variables
    """

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    """
    to make addition to a python function
    :param x0:
    :param x1:
    :return:
    """
    x1 = as_array(x1)  # -> ndarray
    return Add()(x0, x1)


########### Subtraction #################
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)  # swap x1 and x0


########### Negative operator ##############
class Neg(Function):
    """
    If something is multiplied by -1 and passed downstream
    """

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


######### Operator overloading #######
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
############### Testing stage #####################

# to test with no backpropogation
# with no_grad():
#     x = Variable(np.array(2.0))
#     y = square(x)
#     print(y.data)