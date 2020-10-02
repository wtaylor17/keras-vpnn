"""
vpnn.utils
========================================================================================
General utility functions for the package
"""
import keras
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
    if activation == 'cheby':
        if type(dim) != int or dim <= 0:
            return None
        return layers.Chebyshev(dim, M=cheby_M)
    else:
        return keras.activations.get(activation)


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


def adding_problem_generator(batch_size=256, time_steps=10, center=False):
    """
    A batch generator for the adding problem.
    Code reused from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py

    :param batch_size: batch size to use
    :param time_steps: sequence length
    :param center: whether or not to give data 0 mean
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
            if center:
                inputs -= np.mean(inputs, axis=0, keepdims=True)

            yield inputs, targets

    return batch_generator
