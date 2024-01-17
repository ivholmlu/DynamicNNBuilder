import toml
import torch
import torch.nn as nn
from src.network.network import NeuralNetwork

single_dense_layer = toml.load("tests/conf_test/test_network.toml")

def test_forward():
    network = NeuralNetwork(single_dense_layer)
    network._layers[0]._b = nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float32))
    network._layers[0]._W = nn.Parameter(torch.tensor([[1, 2, 3],  
                        [4, 5, 6], 
                        [7, 8, 9]], dtype=torch.float32))

    x = torch.tensor([1, 0, -1], dtype=torch.float32).unsqueeze(0)

    Z = network.forward(x)

    assert torch.equal(Z, torch.tensor([[-5, -5, -5]], dtype=torch.int32))
