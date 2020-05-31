from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np


class Rotation(Layer):
    def __init__(self, n_outputs, **kwargs):
        self.output_dim = n_outputs
        assert self.output_dim % 2 == 0
        self.thetas, self.c, self.s, self.kernel = None, None, None, None
        self.inp_inds = tf.Variable(
            np.random.permutation(self.output_dim),
            trainable=False,
            dtype=tf.int32
        )
        self.outp_inds = tf.Variable(
            np.random.permutation(self.output_dim),
            trainable=False,
            dtype=tf.int32
        )
        super(Rotation, self).__init__(**kwargs)
    
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
        permuted = tf.gather(x, self.inp_inds, axis=-1)
        xi = permuted[:, :self.output_dim // 2]
        xj = permuted[:, self.output_dim // 2:]
        yi = self.c * xi - self.s * xj
        yj = self.c * xj + self.s * xi
        return tf.concat([yi, yj], axis=-1)
