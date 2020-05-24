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
* MNIST: https://colab.research.google.com/drive/129DK7ZRVO018a4BxY2LoueIiJxE4aMAm?usp=sharing

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