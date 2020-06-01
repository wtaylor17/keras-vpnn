from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from math import sqrt

K.set_epsilon(1e-5)


class Chebyshev(Layer):
    def __init__(self, n_inputs, M=2.0, **kwargs):
        self.n_inputs = n_inputs
        self.M = M
        super(Chebyshev, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Chebyshev, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(Chebyshev, self).__call__(*args, **kwargs)

    def call(self, x, **kwargs):
        xs = x[..., ::2]
        ys = x[..., 1::2]
        r = K.sqrt(K.square(xs) + K.square(ys)) + K.epsilon()
        M_angle = self.M * tf.math.acos(K.clip(xs / r, -1, 1))
        cheby_cos = K.cos(M_angle)
        cheby_sin = K.sin(M_angle)
        evens = r / sqrt(self.M) * cheby_cos
        odds = r / sqrt(self.M) * K.sign(ys) * cheby_sin
        return tf.reshape(
            tf.concat([evens[..., tf.newaxis], odds[..., tf.newaxis]], axis=-1),
            [tf.shape(evens)[0], self.n_inputs]
        )

    def get_config(self):
        config = super(Chebyshev, self).get_config()
        config.update({'n_inputs': self.n_inputs,
                       'M': self.M})
        return config
