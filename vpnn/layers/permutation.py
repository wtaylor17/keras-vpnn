import numpy as np
from keras.layers import Layer
import keras.backend as K
from vpnn.utils import build_permutation

class Permutation(Layer):
    def __init__(self, n_outputs, **kwargs):
        if 'trainable' in kwargs:
            print('WARNING: trainable should never be passed to Permutation')
            del kwargs['trainable']
        self.output_dim = n_outputs
        self.kernel, self.q_ = build_permutation(self.output_dim)
        super(Permutation,self).__init__(trainable=False, **kwargs)

    def build(self, input_shape):
        super(Permutation,self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Permutation, self).__call__(*args, **kwargs)

    def call(self,x):
        return K.dot(x, self.kernel)
