import toml
import torch
import torch.nn as nn
from src.network.network import NeuralNetwork

#Several different configurations with dense layers
single_dense_layer = toml.load("tests/conf_test/test_network.toml")
config = toml.load("tests/conf_test/test_conf.toml")
test_network_relu = toml.load("tests/conf_test/test_network_relu.toml")
test_dense = toml.load("tests/conf_test/teste_dense.toml")

@pytest.mark.parametrize("config", [config, test_network_relu, single_dense_layer, test_dense])
def test_creations_b(conf):
    obj = NeuralNetwork(conf)
    for i, _ in enumerate(obj._layers):
        assert obj._layers[i]._b.size() == (conf["layer"][i]["dim_out"],)

@pytest.mark.parametrize("config", [config, test_network_relu, single_dense_layer, test_dense])
def test_creations_dense_W(conf):
    obj = NeuralNetwork(conf)
    for i, _ in enumerate(obj._layers):
        assert obj._layers[i]._W.size() == (conf["layer"][i]["dim_in"], conf["layer"][i]["dim_out"])

def test_forward():
    network = NeuralNetwork(single_dense_layer)
    network._layers[0]._b = nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float32))
    network._layers[0]._W = nn.Parameter(torch.tensor([[1, 2, 3],  
                        [4, 5, 6], 
                        [7, 8, 9]], dtype=torch.float32))

    x = torch.tensor([1, 0, -1], dtype=torch.float32).unsqueeze(0)

    Z = network.forward(x)

    assert torch.equal(Z, torch.tensor([[-5, -5, -5]], dtype=torch.int32))

single_dense_layer_relu = toml.load("tests/conf_test/test_network_relu.toml")
def test_forward_with_relu():
    #Testing 3x3 dense layer with ReLU
    network = NeuralNetwork(single_dense_layer_relu)
    network._layers[0]._W = nn.Parameter(torch.tensor([[1, -1, 0],  
                                                        [0, 1, -1], 
                                                        [-1, 0, 1]], dtype=torch.float32))
    network._layers[0]._b = nn.Parameter(torch.tensor([1, 0, -1], dtype=torch.float32))
    x = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(0)
    Z = network(x)

    # Expected output should have non-negative values due to ReLU
    # Change these values according to the expected result after applying ReLU
    expected_output = torch.tensor([[0, 1, 0]], dtype=torch.int32)  # Update as necessary
    assert torch.equal(Z, expected_output)

several_layer = toml.load("tests/conf_test/several_layer.toml")

def test_entire_network_creation():
    network = NeuralNetwork(several_layer)
    assert len(network._layers) == 3
