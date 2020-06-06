from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np


class Rotation(Layer):
    def __init__(self, n_outputs, perm=None, **kwargs):
        self.output_dim = n_outputs
        assert self.output_dim % 2 == 0
        self.thetas, self.c, self.s = None, None, None
        self.perm = perm or np.random.permutation(self.output_dim).tolist()
        self.inp_inds = tf.Variable(
            self.perm,
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
        return K.concatenate([yi, yj], axis=-1)

    def get_config(self):
        config = super(Rotation, self).get_config()
        config.update({'n_outputs': self.output_dim,
                       'perm': self.perm})
        return config
