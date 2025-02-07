"""Module for testing the network."""
import torch
from torch import nn
from .layers import LayerFactory


# pylint: disable=C0103
# pylint: disable=C0116
class NeuralNetwork(nn.Module):
    """Class for creating a neural network."""
    def __init__(self, config, create_net=True) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self._layers = nn.Sequential()
        LF = LayerFactory()
        if create_net:
            self._lr = config["settings"]["learning_rate"]
            self._contains_lowrank = False
            for i, layer in enumerate(config["layer"], 1):
                self._layers.add_module(
                    name=f"layer_{i}_{layer['type']}_{layer['activation']}",
                    module=LF(layer, self._lr))
                if layer["type"] == "lowrank":
                    self._contains_lowrank = True

        else:  # From params
            self._layers = nn.Sequential()
            for layer_num in config:
                self._layers.add_module(
                    name=f"layer_{layer_num}_{config[layer_num]['type']}_" +
                    "{config[layer_num]['activation']}",
                    module=LF(config[layer_num], load=True))

    def forward(self, Z) -> torch.Tensor:
        Z = self.flatten(Z)
        for layer in self._layers:
            Z = layer(Z)
        return Z

    def step(self, s=False) -> None:
        for layer in self._layers:
            layer.step(s)

    @property
    def layers(self):
        return self._layers

    @property
    def contains_lowrank(self):
        return self._contains_lowrank
    