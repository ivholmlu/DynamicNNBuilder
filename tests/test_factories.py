import src.network.layers as layers
import toml
import torch.nn as nn

config = toml.load("tests/conf_test/test_conf.toml")

"""
[[layer]]
type = 'dense'
dims = [784, 512]
dim_in = 784
dim_out = 512
activation = 'relu'

[[layer]]
type = 'dense'
dims = [512, 256]
dim_in = 512
dim_out = 256
activation = 'relu'

[[layer]]
type = 'dense'
dims = [256, 10]
dim_in = 256
dim_out = 10
activation = 'linear'
"""


def test_creation_layerfactory():
    layerfactory = layers.LayerFactory()
    assert layerfactory.classes == {"dense": layers.Denselayer,
                                    "vanillalowrank": layers.VanillaLowRank,
                                    "lowrank": layers.LowRank}


def test_return_layerfactory():
    layerfactory = layers.LayerFactory()
    layer = layerfactory(config["layer"][0])
    assert isinstance(layer, layers.Denselayer)


def test_return_activationfactory():
    activationfactory = layers.ActivationFactory()
    activation = activationfactory(config["layer"][0]["activation"])
    assert isinstance(activation, nn.ReLU)
