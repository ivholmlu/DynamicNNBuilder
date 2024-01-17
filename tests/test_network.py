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
    print(Z)
    expected_output = torch.tensor([[0, 1, 0]], dtype=torch.int32)  # Update as necessary
    assert torch.equal(Z, expected_output)
