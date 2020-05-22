from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf

class SVDDownsize(Layer):
    def __init__(self, input_dim, output_dim, n_rotations=1, **kwargs):
        self.output_dim = output_dim
        super(SVDDownsize, self).__init__(**kwargs)
        val = np.random.normal(size=(input_dim, output_dim))
        u,s,v = np.linalg.svd(val)
        z = u[:,:output_dim]
        self.Z = tf.Variable(z,dtype=tf.float32,trainable=False)
    
    def build(self, input_shape):
        super(SVDDownsize, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        return super(SVDDownsize, self).__call__(*args, **kwargs)

    def call(self, x):
        return K.dot(x,self.Z)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim