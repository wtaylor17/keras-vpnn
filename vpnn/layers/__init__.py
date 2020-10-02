from .rotation import Rotation
from .diagonal import Diagonal
from .bias import Bias
from .downsize import SVDDownsize
from .cheby import Chebyshev
from . import bias, cheby, diagonal, downsize, rotation
from keras.models import Model
from keras.layers import Input
from ..utils import get_activation


def VPNNLayer(dim,
              n_rotations=2,
              output_dim=None,
              activation=None,
              use_bias=True,
              use_diag=True,
              diag_func='exp_sin',
              cheby_M=2.0,
              diagonal_M=0.01,
              **kwargs):
    """
    creates a single VPNN layer
    :param diagonal_M: parameter for diagonal function
    :param cheby_M: parameter for cheby activation
    :param dim: input dimension of the model
    :param n_rotations: number of rotations in each of the 2 rotational sub layers.
    :param output_dim: if not None, the dimension of a SVDDownsize used as an output sublayer.
    :param activation: if str, interpreted as the name of an activation function to use.
    :param use_bias: if True, a bias layer is applied before the activation
    :param use_diag: if True, a diagonal sublayer is used inbetween rotations
    :param diag_func: if == 'exp_sin', then exp(sin(x)) is used instead of sigmoid for diagonals
    :return: a keras.models.Model instance representing the layer
    """
    _hidden_layers = []
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
    if use_diag:
        _hidden_layers.append(Diagonal(dim, M=diagonal_M, func_name=diag_func))
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
    if use_bias:
        _hidden_layers.append(Bias(dim))
    if activation is not None:
        fn = get_activation(activation, dim=dim, cheby_M=cheby_M)
        _hidden_layers.append(fn)
    if output_dim:
        _hidden_layers.append(SVDDownsize(dim, output_dim))

    input_layer = Input((dim,))
    outp = input_layer
    for layer in _hidden_layers:
        outp = layer(outp)
    output_layer = outp
    return Model(input_layer, output_layer, **kwargs)
