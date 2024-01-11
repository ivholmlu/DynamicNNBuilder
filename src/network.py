import torch.nn as nn
from src.layers import Denselayer

class NeuralNetwork(nn.Module):

    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        #Setting up network configs
        self._layers = nn.Sequential()
        self._lr = config["settings"]["learning_rate"]
    
        #Creating layers from config
        for i, layer in enumerate(config["layer"], 1):
            self._layers.add_module(name=f"layer_{i}", module=Denselayer(layer, self._lr))
        
    def forward(self, Z):
        for layer in self._layers:
            Z = layer(Z)
        return Z
    
    def step(self):
        for layer in self._layers:
            layer.step()