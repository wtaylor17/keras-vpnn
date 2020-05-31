from .rotation import Rotation
from .diagonal import Diagonal
from .bias import Bias
from .downsize import SVDDownsize
from .cheby import Chebyshev
from . import bias, cheby, diagonal, downsize, rotation
from keras.models import Model
from keras.layers import Input, Activation, Layer
import keras.backend as K


def vpnn_layer(dim,
               n_rotations=2,
               output_dim=None,
               activation=None,
               use_bias=True,
               cheby_M=2.0,
               diagonal_M=0.01):
    """
    creates a single VPNN layer
    :param diagonal_M: parameter for diagonal function
    :param cheby_M: parameter for cheby activation
    :param dim: input dimension of the model
    :param n_rotations: number of rotations in each of the 2 rotational sub layers.
    :param output_dim: if not None, the dimension of a SVDDownsize used as an output sublayer.
    :param activation: if str, interpreted as the name of an activation function to use.
    :param use_bias: if True, a bias layer is applied before the activation
    :return: a keras.models.Model instance representing the layer
    """
    _hidden_layers = []
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
    _hidden_layers.append(Diagonal(dim, M=diagonal_M))
    for _ in range(n_rotations):
        _hidden_layers.append(Rotation(dim))
    if use_bias:
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
    return Model(input_layer, output_layer)


class VPNNLayer(Layer):
    def __init__(self,
                 dim,
                 n_rotations=2,
                 output_dim=None,
                 activation=None,
                 use_bias=True,
                 cheby_M=2.0,
                 diagonal_M=0.01,
                 **kwargs):
        """
        creates a single VPNN layer
        :param diagonal_M: parameter for diagonal function
        :param cheby_M:
        :param name: name for the layer
        :param dim: input dimension of the model
        :param n_rotations: number of rotations in each of the 2 rotational sub layers.
        :param output_dim: if not None, the dimension of a SVDDownsize used as an output sublayer.
        :param activation: if str, interpreted as the name of an activation function to use.
        :param use_bias: if True, a bias layer is applied before the activation
        :param kwargs: passed to super constructor
        """
        self.vpnn_layer_model = vpnn_layer(dim,
                                           n_rotations=n_rotations,
                                           output_dim=output_dim,
                                           activation=activation,
                                           use_bias=use_bias,
                                           cheby_M=cheby_M,
                                           diagonal_M=diagonal_M)
        super(VPNNLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.vpnn_layer_model.outputs[0])

    def call(self, x, **kwargs):
        return self.vpnn_layer_model(x)
