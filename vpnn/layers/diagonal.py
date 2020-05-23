from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

from vpnn.utils import default_diag, build_diagonal


class Diagonal(Layer):
    def __init__(self, n_outputs, M=0.01, **kwargs):
        self.output_dim = n_outputs
        self.func = default_diag # can be changed
        self.params = None
        self.M = M
        self.f_t = None
        self.diag = None
        super(Diagonal, self).__init__()
    
    def build(self, input_shape):
        self.params = self.add_weight(name='t',
                                      initializer='uniform',
                                      shape=(self.output_dim,))
        self.diag = build_diagonal(self.params, self.func, self.M)
        super(Diagonal, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Diagonal, self).__call__(*args, **kwargs)
    
    def call(self, x):
        return x * self.diag


class Hadamard(Layer):
    """
    a linear transformation of a diagonal matrix,
    uses hadamard product
    """
    def __init__(self, n_outputs, **kwargs):
        self.output_dim = n_outputs
        self.vec = None
        super(Hadamard, self).__init__()

    def build(self, input_shape):
        self.vec = self.add_weight(name='t',
                                    initializer='uniform',
                                    shape=(self.output_dim,))
        super(Hadamard, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Hadamard, self).__call__(*args, **kwargs)

    def call(self, x):
        return x * self.vec
