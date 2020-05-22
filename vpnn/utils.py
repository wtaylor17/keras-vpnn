from keras.layers import Lambda, Input, Layer, Activation
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np

def get_activation(activation, dim=-1, cheby_M=2):
    """
    safely tries to get an activation function
    :param cheby_M: M for chebyshev
    :param activation: str, or layer, or callable
    :param dim: the output dim, only needed if activation=='cheby'
    :return: an activation layer
    """
    if isinstance(activation, Layer):
        return activation
    elif activation == 'cheby':
        if type(dim)!=int or dim<=0:
            return None
        return Chebyshev(dim, M=cheby_M)
    else:
        return Activation(activation)

def build_permutation(dim):
    """
    build a permutation transformation
    :param dim: the in/out dimension
    :return: the permutation kernel
    """
    perm = np.random.permutation(dim)
    q = np.zeros((dim,dim))
    for i in range(dim):
        q[i][perm[i]] = 1
    kernel = tf.Variable(q,dtype=tf.float32,trainable=False)
    return kernel, q

def default_diag(x):
    """
    default diagonal function
    :param x: tensor input
    :return: the function applied to x
    """
    return 1 / (1 + K.exp(-x))

def build_diagonal(params, func, M=0.01):
    """
    build the diagonal transformation as a vector
    :param params: the parameters to be used (trainable)
    :param func: function to use
    :param M: M parameter to use for the function
    :return: the diagonal transformation
    """
    f_t = M * func(params / M) + M
    return f_t / tf.roll(f_t,shift=-1,axis=0)

def build_rotation(output_dim, cos, sin):
    """
    makes a rotational kernel based on direct sums of 2d rotation matrices
    :param output_dim: target output dimension
    :param cos: cos of theta, should have dimension output_dim//2
    :param sin: sin of theta, should have dimension output_dim//2
    :return: the rotational kernel matrix (transposed)
    """
    dim = output_dim
    # index build
    evens = range(0,dim,2)
    ul_indices = [[i,i] for i in evens]
    sparse_out = tf.sparse.SparseTensor(ul_indices,cos,[dim,dim])
    ur_indices = [[i,i+1] for i in evens]
    sparse_out = tf.sparse.add(sparse_out,
                   tf.sparse.SparseTensor(ur_indices,-sin,[dim,dim]))
    ll_indices = [[i+1,i] for i in evens]
    sparse_out = tf.sparse.add(sparse_out,
                   tf.sparse.SparseTensor(ll_indices,sin,[dim,dim]))
    lr_indices = [[i+1,i+1] for i in evens]
    sparse_out = tf.sparse.add(sparse_out,
                   tf.sparse.SparseTensor(lr_indices,cos,[dim,dim]))
    return K.transpose(K.to_dense(sparse_out))

def temporal_slice_layer(j):
    """
    returns a keras.layers.Layer which maps a tensor X to the time step X[:,j,:].
    :param j: the time step (int)
    :return: a keras.layers.Lambda instance.
    """
    return Lambda(lambda x: x[:, j, :])

def merge_layer(dim):
    """
    returns a keras.layers.Layer which attempts to merge a list of tensors into a desired form.
    The main use of this is for Chebyshev stuff.
    If the list of tensors is [E,O] such that E = X[...,::2] and O = X[...,1::2] for some tensor X,
    then the layer L returned has the property that L([E,O]) = X provided that dim = X.shape[-1].
    :param dim: the desired output dimension of the last axis
    :return: a keras.layers.Lambda instance.
    """
    return Lambda(lambda tensors:
                    tf.reshape(tf.concat([t[...,tf.newaxis] for t in tensors], axis=-1),
                                [tf.shape(tensors[0])[0],dim]))

def addition_layer():
    """
    layer for adding tensors
    :return: a Lambda layer taking in a list of tensors as input
    """
    return Lambda(lambda tensors: tf.math.add_n(tensors))

def train_dropout(x, rate=0.4):
    """
    returns a tensor representing dropout during training
    :param x: the input tensor to apply dropout to
    :param rate: the dropout rate, a float between 0 and 1
    :return: a tensor representing the dropout of x during training, and simply x otherwise
    """
    return K.in_train_phase(K.dropout(x, rate), x)

def mnist_generator(batch_size=256):
    """
    creates a function for creating generators on MNIST.
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
            x,y = x_train[q,:], y_train[q,:]
            while end <= batch_size:
                yield x[end-batch_size:end], y[end-batch_size:end]
                end += batch_size
    return batch_generator

def adding_problem_generator(batch_size=256, time_steps=10):
    """
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
