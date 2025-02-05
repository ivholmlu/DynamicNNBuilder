"""Tests for the network module"""
import pytest
import toml
import torch
from torch import nn
from src.network.network import NeuralNetwork

# Several different configurations with dense layers
single_dense_layer = toml.load("tests/conf_test/test_network.toml")
config = toml.load("tests/conf_test/test_conf.toml")
test_network_relu = toml.load("tests/conf_test/test_network_relu.toml")
test_dense = toml.load("tests/conf_test/test_dense.toml")


@pytest.mark.parametrize(
        "conf",
        [config, test_network_relu, single_dense_layer, test_dense])
def test_creations_b(conf):
    "Test if the bias vector dimensions is created correctly"
    obj = NeuralNetwork(conf)
    for i, _ in enumerate(obj.layers):
        assert obj.layers[i].b.size() == (conf["layer"][i]["dim_out"],)


@pytest.mark.parametrize(
        "conf", [config, test_network_relu, single_dense_layer, test_dense])
def test_creations_dense_w(conf):
    "Test if the weight matrix dimensions is created correctly"
    obj = NeuralNetwork(conf)
    for i, _ in enumerate(obj.layers):
        layer_dim_in = conf["layer"][i]["dim_in"]
        layer_dim_out = conf["layer"][i]["dim_out"]
        expected_size = (layer_dim_in, layer_dim_out)
        assert obj.layers[i].W.size() == expected_size


def test_forward():
    """Testing 3x3 dense layer with identity activation"""
    network = NeuralNetwork(single_dense_layer)
    network.layers[0].b = nn.Parameter(
        torch.tensor([1, 1, 1], dtype=torch.float32))
    network.layers[0].W = nn.Parameter(torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32))
    x = torch.tensor([1, 0, -1], dtype=torch.float32).unsqueeze(0)
    z = network.forward(x)
    assert torch.equal(z, torch.tensor([[-5, -5, -5]], dtype=torch.int32))


single_dense_layer_relu = toml.load("tests/conf_test/test_network_relu.toml")


def test_forward_with_relu():
    "Testing 3x3 dense layer with ReLU"
    network = NeuralNetwork(single_dense_layer_relu)
    network.layers[0].W = nn.Parameter(torch.tensor(
        [[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=torch.float32))
    network.layers[0].b = nn.Parameter(
        torch.tensor([1, 0, -1], dtype=torch.float32))
    x = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(0)
    z = network(x)

    # Expected output should have non-negative values due to ReLU
    # Change these values according to the expected result after applying ReLU
    expected_output = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    assert torch.equal(z, expected_output)


several_layer = toml.load("tests/conf_test/several_layer.toml")


def test_entire_network_creation():
    """Test if the network is created correctly and without errors"""
    network = NeuralNetwork(several_layer)
    assert len(network.layers) == 4
