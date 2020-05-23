from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from vpnn.utils import build_rotation


class Rotation(Layer):
    def __init__(self, n_outputs, **kwargs):
        self.output_dim = n_outputs
        assert self.output_dim % 2 == 0
        self.thetas, self.c, self.s, self.kernel = None, None, None, None
        super(Rotation,self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.thetas = self.add_weight(name='theta',
                                      initializer='uniform',
                                      shape=(self.output_dim//2,))
        self.c = K.cos(self.thetas)
        self.s = K.sin(self.thetas)
        self.kernel = build_rotation(self.output_dim, self.c, self.s)
        super(Rotation, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Rotation, self).__call__(*args, **kwargs)
    
    def call(self, x):
        return K.dot(x, self.kernel)  # xV^T
