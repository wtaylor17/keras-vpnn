"""
vpnn.models
==================================================
functions for loading specific model architectures
"""

from .layers import VPNNLayer, Chebyshev, SVDDownsize
from .utils import get_activation
from keras.layers import Input
from keras.models import Model



def vpnn(dim, n_layers=1, out_dim=None, out_ac=None, **kwargs):
    """
    stacks vpnn layers, making a multi layer model.

    :param dim: input dimension of the vpnn
    :param n_layers: number of layers in the model
    :param out_dim: if not None, the dimension of an SVDDownsize output layer
    :param out_ac: str, the output activation of the model
    :param kwargs: passed to vpnn_layer

    :return: a ``keras.models.Model`` instance.
    """
    if 'output_dim' in kwargs:
        del kwargs['output_dim']
    layers = [VPNNLayer(dim, name='vpnn_%d'%_, **kwargs) for _ in range(n_layers-1)]
    if 'activation' in kwargs:
        kwargs['activation'] = None
    layers.append(VPNNLayer(dim, name='vpnn_out', **kwargs))
    input_layer = Input((dim,))
    current_output = input_layer
    for L in layers:
        current_output = L(current_output)
    if out_dim is not None:
        current_output = SVDDownsize(dim, out_dim)(current_output)
    if out_ac:
        if out_ac == 'cheby':
            M = 1.3 if 'cheby_M' not in kwargs else kwargs['cheby_M']
            current_output = Chebyshev(out_dim if out_dim is not None else dim, M=M)
        else:
            current_output = get_activation(out_ac)(current_output)
    return Model(input_layer, current_output)
