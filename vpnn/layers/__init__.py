from .rotation import Rotation
from .permutation import Permutation
from .diagonal import Diagonal
from .bias import Bias
from .downsize import SVDDownsize
from .cheby import Chebyshev
from . import bias, cheby, diagonal, downsize, permutation, rotation

from keras.models import Model
from keras.layers import Input, Activation, Layer
import keras.backend as K


def vpnn_layer(dim,
               n_rotations=2,
               output_dim=None,
               activation=None,
               bias=True,
               perm=True,
               name=None,
               cheby_M=2.0,
               diagonal_M=0.01):
    """
    creates a single VPNN layer
    :param diagonal_M: parameter for diagonal function
    :param cheby_M:
    :param name: name for the layer
    :param perm: if True, permutation sub-layers are enabled
    :param dim: input dimension of the model
    :param n_rotations: number of rotations in each of the 2 rotational sub layers.
    :param output_dim: if not None, the dimension of a SVDDownsize used as an output sublayer.
    :param activation: if str, interpreted as the name of an activation function to use.
    :param bias: if True, a bias layer is applied before the activation
    :return: a keras.models.Model instance representing the layer
    """
    if not perm:
        print('WARNING: no permutations will be used.')
    _hidden_layers = []
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
        if perm:
            _hidden_layers.append(Permutation(dim))
    _hidden_layers.append(Diagonal(dim, M=diagonal_M))
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
        if perm:
            _hidden_layers.append(Permutation(dim))
    if bias:
        _hidden_layers.append(Bias(dim))
    if activation is not None:
        if activation == 'cheby':
            _hidden_layers.append(Chebyshev(dim, M=cheby_M))
        else:
            _hidden_layers.append(Activation(activation))
    if output_dim:
        _hidden_layers.append(SVDDownsize(dim, output_dim))

    input_layer = Input((dim,))
    outp = input_layer
    for layer in _hidden_layers:
        outp = layer(outp)
    output_layer = outp
    return Model(input_layer, output_layer, name=name)
