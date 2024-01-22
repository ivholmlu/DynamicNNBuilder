from src.network.layers import Denselayer, VanillaLowRank, LowRank, ActivationFactory, LayerFactory, ActivationFactory, LayerFactory
import toml
import torch.nn as nn

config = toml.load("tests/config_test/test_conf.toml")

def test_creation_layerfactory():
    layerfactory = LayerFactory()
    assert layerfactory.classes == {"dense": Denselayer,
                                    "vanillalowrank": VanillaLowRank,
                                    "lowrank": LowRank}
    
def test_return_layerfactory():
    layerfactory = LayerFactory()
    layer = layerfactory(config["layers"][0])
    assert isinstance(layer, Denselayer)
    
def test_creation_activationfactory():
    activationfactory = ActivationFactory()
    assert activationfactory.activations == {"relu": nn.ReLU(),
                                            "identity": nn.Identity()}
    
def test_return_activationfactory():
    activationfactory = ActivationFactory()
    activation = activationfactory(config["layers"][0]["activation"])
    assert isinstance(activation, nn.ReLU)