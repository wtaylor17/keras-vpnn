"""
vpnn.utils
========================================================================================
General utility functions for the package. Includes many custom ``keras.layers.Lambda``,
and some volume preserving transformation creation functions.
"""

from keras.layers import Lambda, Layer, Activation
from . import layers
import keras.backend as K
import tensorflow as tf
import numpy as np


def get_activation(activation, dim=-1, cheby_M=2):
    """
    Safely tries to get an activation function

    :param cheby_M: M for chebyshev
    :param activation: str, or layer, or callable
    :param dim: the output dim, only needed if ``activation='cheby'``

    :return: a ``keras.layers.Activation`` layer or ``vpnn.layers.cheby.Chebyshev`` layer.
    """
    if isinstance(activation, Layer):
        return activation
    elif activation == 'cheby':
        if type(dim) != int or dim <= 0:
            return None
        return layers.Chebyshev(dim, M=cheby_M)
    else:
        return Activation(activation)


def build_permutation(dim):
    """
    Build a permutation transformation without trainable parameters.

    :param dim: the in/out dimension

    :return: a tuple, with the first element being a ``tf.Variable`` and the second being a numpy array.
    """
    perm = np.random.permutation(dim)
    q = np.zeros((dim, dim))
    for i in range(dim):
        q[i][perm[i]] = 1
    kernel = tf.Variable(q, dtype=tf.float32, trainable=False)
    return kernel, q


def default_diag(x):
    """
    Default diagonal function, currently a sigmoid.

    :param x: a tensor input

    :return: the function applied to x as a new tensor.
    """
    return 1 / (1 + K.exp(-x))


def build_diagonal(params, func, M=0.01):
    """
    Build the diagonal transformation as a vector.

    :param params: the parameters to be used (trainable)
    :param func: function to use
    :param M: M parameter to use for the function

    :return: the diagonal transformation as a tensor.
    """
    f_t = M * func(params / M) + M
    return f_t / tf.roll(f_t, shift=-1, axis=0)


def temporal_slice_layer(j):
    """
    Returns a ``keras.layers.Layer`` which maps a tensor ``X`` to the time step ``X[:,j,:]``.

    :param j: the time step (int)

    :return: a ``keras.layers.Lambda`` instance.
    """
    return Lambda(lambda x: x[:, j, :])


def merge_layer(dim):
    """
    returns a ``keras.layers.Layer`` which attempts to merge a list of tensors into a desired form.
    The main use of this is for Chebyshev stuff.
    If the list of tensors is ``[E,O]`` such that ``E = X[...,::2]`` and ``O = X[...,1::2]`` for some tensor X,
    then the layer ``L`` returned has the property that ``L([E,O]) = X`` provided that ``dim = X.shape[-1]``.

    :param dim: the desired output dimension of the last axis

    :return: a ``keras.layers.Lambda`` instance.
    """
    return Lambda(lambda tensors:
                  tf.reshape(tf.concat([t[..., tf.newaxis] for t in tensors], axis=-1),
                             [tf.shape(tensors[0])[0], dim]))


def addition_layer():
    """
    Layer for adding tensors. A wrapper for ``tf.math.add_n``.

    :return: a Lambda layer taking in a list of tensors as input
    """
    return Lambda(lambda tensors: tf.math.add_n(tensors))


def train_dropout(x, rate=0.4):
    """
    Returns a tensor representing dropout during training.

    :param x: the input tensor to apply dropout to
    :param rate: the dropout rate, a float between 0 and 1

    :return: a tensor representing the dropout of x during training, and simply x otherwise
    """
    return K.in_train_phase(K.dropout(x, rate), x)


def mnist_generator(batch_size=256):
    """
    Creates a function for creating generators on MNIST.

    :param batch_size: the batch size to use

    :return: a function for training batch generation
    """
    (x_train, label_train), (x_test, label_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    y_train = tf.keras.utils.to_categorical(label_train, 10)

    def batch_generator():
        while True:
            end = batch_size
            q = np.random.permutation(x_train.shape[0])
            x, y = x_train[q, :], y_train[q, :]
            while end <= batch_size:
                yield x[end - batch_size:end], y[end - batch_size:end]
                end += batch_size

    return batch_generator


def adding_problem_generator(batch_size=256, time_steps=10):
    """
    A batch generator for the adding problem.
    Code reused from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py

    :param batch_size: batch size to use
    :param time_steps: sequence length

    :return: a function for making generators
    """

    def batch_generator():
        while True:
            """Generate the adding problem dataset"""
            # Build the first sequence
            add_values = np.random.rand(batch_size, time_steps)

            # Build the second sequence with one 1 in each half and 0s otherwise
            add_indices = np.zeros_like(add_values)
            half = int(time_steps / 2)
            for i in range(batch_size):
                first_half = np.random.randint(half)
                second_half = np.random.randint(half, time_steps)
                add_indices[i, [first_half, second_half]] = 1

            # Zip the values and indices in a third dimension:
            # inputs has the shape (batch_size, time_steps, 2)
            inputs = np.dstack((add_values, add_indices))
            targets = np.sum(np.multiply(add_values, add_indices), axis=1)

            # center at zero mean
            inputs -= np.mean(inputs, axis=0, keepdims=True)

            yield inputs, targets

    return batch_generator
