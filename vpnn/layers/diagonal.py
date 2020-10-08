from keras.layers import Layer
from keras.initializers import constant
import keras.backend as K
import tensorflow as tf


def exp_sin(x):
    return K.exp(K.sin(x))


class Diagonal(Layer):
    def __init__(self, n_outputs,
                 M=0.01,
                 func_name='exp_sin',
                 t_initializer='uniform',
                 use_M=True,
                 **kwargs):
        self.output_dim = n_outputs
        self.params = None
        self.M = M
        self.f_t = None
        self.diag = None
        self.func_name = func_name
        if func_name == 'exp_sin':
            self.func = exp_sin
        else:
            self.func = K.sigmoid
        self.use_M = use_M
        self.t_initializer = t_initializer
        super(Diagonal, self).__init__(**kwargs)

    def build(self, input_shape):
        if type(self.t_initializer) is float:
            initializer = constant(self.t_initializer)
        else:
            initializer = self.t_initializer
        self.params = self.add_weight(name='t',
                                      initializer=initializer,
                                      shape=(self.output_dim,))
        if self.use_M:
            f = self.M * self.func(self.params / self.M) + self.M
        else:
            f = self.func(self.params)
        self.diag = f / tf.roll(f, -1, 0)
        super(Diagonal, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Diagonal, self).__call__(*args, **kwargs)

    def call(self, x, **kwargs):
        return x * self.diag

    def get_config(self):
        config = super(Diagonal, self).get_config()
        config.update({'n_outputs': self.output_dim, 'M': self.M,
                       'func_name': self.func_name, 'use_M': self.use_M,
                       't_initializer': self.t_initializer})
        return config


class Hadamard(Layer):
    """
    a linear transformation of a diagonal matrix,
    uses hadamard product
    """

    def __init__(self, n_outputs, **kwargs):
        self.output_dim = n_outputs
        self.vec = None
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vec = self.add_weight(name='vec',
                                   initializer='uniform',
                                   shape=(self.output_dim,))
        super(Hadamard, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Hadamard, self).__call__(*args, **kwargs)

    def call(self, x, **kwargs):
        return x * self.vec
