# keras-vpnn
Implementation of "Volume Preserving Neural Networks" by MacDonald et al. in keras.

# Examples
## Using layers
```python
from vpnn.layers import VPNNLayer
from keras.layers import Input
from keras.models import Model

in_dim, out_dim = 784, 10 # MNIST
input_layer = Input((in_dim,))
vpnn_layer = VPNNLayer(in_dim,
                       activation='softmax',
                       output_dim=out_dim)
vpnn_model = Model(input_layer, vpnn_layer(input_layer))
```

## Using models
```python
from vpnn import vpnn

in_dim, out_dim = 784, 10 # MNIST
vpnn_model = vpnn(in_dim,
                  n_layers=3,
                  activation='softmax',
                  out_dim=out_dim)
```

# Demos
* MNIST [colab notebook](https://colab.research.google.com/drive/1SDfFRQKZh2us9VQekcQZK0j_gjO-zc9y?usp=sharing)

# Note on Rotation Implementation
In the original VPNN each "rotation-permutation" pair computes `R*Q*x`, i.e., a permutation and then a rotation.
In the fast implementation, this requires splitting `Q*x` into even-odd index pairs and then performing the transformation `R`.
In this implementation, instead of performing even-odd splitting, I simply take the top and bottom halfs of `Q*x` as this is faster.
This is equivalent to adding a fixed permutation `Q'` to the transformation (as we are just shuffling indices) and does not change the actual architecture.
To see this fact, we could replace each `Q` by a permutation `Q * inv(Q')` or something similar to make these extra shuffles cancel out, but in practice this won't have much (if any benefit). It certainly
doesn't give the model any unfair advantage, is the point of this disclaimer.

# Citation of original work:
```
@misc{macdonald2019volumepreserving,
    title={Volume-preserving Neural Networks: A Solution to the Vanishing Gradient Problem},
    author={Gordon MacDonald and Andrew Godbout and Bryn Gillcash and Stephanie Cairns},
    year={2019},
    eprint={1911.09576},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
