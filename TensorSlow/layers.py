from TensorSlow.core import Parameter, Variable
import TensorSlow.functions as F
import weakref
import numpy as np

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
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
            yield self.__dict__[name]
        def cleargrads(self):
            for param in self.params():
                param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:  # If in_size is not specified, then process later.
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # initialize weights during forward propagation
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

# layer = Layer()
#
# layer.p1 = Parameter(np.array(1))
# layer.p2 = Parameter(np.array(2))
# layer.p3 = Variable(np.array(3))
# layer.p4 = 'test'
#
# print(layer._params)
# print('-------------')
# for name in layer._params:
#     print(name, layer.__dict__[name])