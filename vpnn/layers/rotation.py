from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from vpnn.utils import build_rotation
import numpy as np


class Rotation(Layer):
    def __init__(self, n_outputs, **kwargs):
        self.output_dim = n_outputs
        assert self.output_dim % 2 == 0
        self.thetas, self.c, self.s, self.kernel = None, None, None, None
        self.inp_pairs = tf.Variable(
            np.random.permutation(self.output_dim).reshape(-1, 2),
            trainable=False,
            dtype=tf.int32
        )
        self.outp_inds = tf.Variable(
            np.random.permutation(self.output_dim),
            trainable=False,
            dtype=tf.int32
        )
        super(Rotation,self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.thetas = self.add_weight(name='theta',
                                      initializer='uniform',
                                      shape=(self.output_dim//2,))
        self.c = K.cos(self.thetas)
        self.s = K.sin(self.thetas)
        super(Rotation, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Rotation, self).__call__(*args, **kwargs)
    
    def call(self, x, **kwargs):
        xi = tf.gather(x, self.inp_pairs[:, 0], axis=-1)
        xj = tf.gather(x, self.inp_pairs[:, 1], axis=-1)
        yi = self.c * xi - self.s * xj
        yj = self.c * xj + self.s * xi
        return tf.gather(tf.concat([yi, yj], axis=-1), self.outp_inds, axis=-1)
