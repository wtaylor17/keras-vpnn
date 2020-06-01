from . import layers
from . import models
from . import utils

vpnn = models.vpnn

_custom_objects = {
    layer.__name__: layer for layer in [layers.Rotation,
                                        layers.Diagonal,
                                        layers.Bias,
                                        layers.Chebyshev,
                                        layers.VPNNLayer,
                                        layers.SVDDownsize]
}


def custom_objects():
    global _custom_objects
    return _custom_objects
