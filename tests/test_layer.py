"""Test for the layer module."""
import toml
from src.network.layers import Denselayer

config = toml.load("tests/conf_test/test_conf.toml")
test_network_relu = toml.load("tests/conf_test/test_conf.toml")
test_network = toml.load("tests/conf_test/test_conf.toml")


def test_creation_dense():
    """Test the creation of a dense layer"""
    dense_obj = Denselayer(config["layer"][0], lr=0)
    assert dense_obj.b.size() == (512,)
    assert dense_obj.W.size() == (784, 512)
    assert dense_obj.activation == 5


def test_config():
    """Test the config file"""
    assert config["layer"][0]["type"] == "dense"
    assert config["layer"][0]["dims"] == [784, 512]
    assert config["layer"][0]["dim_in"] == 784
    assert config["layer"][0]["dim_out"] == 512
    assert config["layer"][0]["activation"] == "relu"

    assert config["layer"][1]["type"] == "dense"
    assert config["layer"][1]["dims"] == [512, 256]
    assert config["layer"][1]["dim_in"] == 512
    assert config["layer"][1]["dim_out"] == 256
    assert config["layer"][1]["activation"] == "relu"

    assert config["layer"][2]["type"] == "dense"
    assert config["layer"][2]["dims"] == [256, 10]
    assert config["layer"][2]["dim_in"] == 256
    assert config["layer"][2]["dim_out"] == 10
    assert config["layer"][2]["activation"] == "linear"
