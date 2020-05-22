from keras.layers import Activation, Layer
import keras.backend as K
import tensorflow as tf
from vpnn.utils import merge_layer
from math import sqrt


PRECISION_EPSILON = 1e-5

def cheby_activate(x, merger, M=2.0):
    """
    returns the standard Chebyshev activation of x.
    :param M:
    :param merger:
    :param x: input tensor
    :return: an output tensor with the same shape as x
    """
    xs = x[..., ::2]
    ys = x[..., 1::2]
    r = K.sqrt(K.square(xs) + K.square(ys)) + PRECISION_EPSILON
    sqrtM = sqrt(M)
    cinv = tf.math.acos(xs / r)
    evens = r / sqrtM * K.cos(M * cinv)
    odds = r / sqrtM * K.sign(ys) * K.sin(M * cinv)
    return merger([evens, odds])


class Chebyshev(Layer):
    def __init__(self, n_inputs, M=2.0, **kwargs):
        self.n_inputs = n_inputs
        self.M = M
        self.merger = None
        super(Chebyshev,self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.merger = merge_layer(self.n_inputs)
        super(Chebyshev,self).build(input_shape)
    
    def __call__(self, *args, **kwargs):
        return super(Chebyshev, self).__call__(*args, **kwargs)
    
    def call(self, x):
        return cheby_activate(x, self.merger, M=self.M)