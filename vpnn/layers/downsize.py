from keras import backend as K
from keras.layers import Layer
from numpy.linalg import svd
from numpy.random import normal
import tensorflow as tf


class SVDDownsize(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(SVDDownsize, self).__init__(**kwargs)
        val = normal(size=(input_dim, output_dim))
        z = svd(val, full_matrices=False)[0]
        self.Z = tf.Variable(z, dtype=tf.float32, trainable=False)
    
    def build(self, input_shape):
        super(SVDDownsize, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(SVDDownsize, self).__call__(*args, **kwargs)

    def call(self, x, **kwargs):
        return K.dot(x, self.Z)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = super(SVDDownsize, self).get_config()
        config.update({'input_dim': self.input_dim,
                       'output_dim': self.output_dim})
        return config
