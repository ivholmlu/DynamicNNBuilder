import torch
import toml
import torch.nn as nn

from src.network import NeuralNetwork
from src.layers import LayerFactory
from src.loader import Loader


class Trainer:
    """Object for training a neural network.
    """
    def __init__(self, conf_path="config.toml") -> None:

        self._config = toml.load(conf_path)
        self.net = NeuralNetwork(self._config)
        self._iterations = self._config["settings"]["iterations"]
        self._criterion = nn.CrossEntropyLoss() #TODO Create a criterion factory
        loader = Loader()
        self._trainloader, self._testloader = loader.load_dataset(self._config)

    @torch.no_grad
    def test(self):
        total = 0
        correct = 0
        for images, labels in self._testloader:
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    def train(self, show_progress=True):
        for i in range(self._iterations):
            for step, (images, labels) in enumerate(self._trainloader):
                out = self.net(images)
                loss = self._criterion(out, labels)
                ### Update loss for network
                loss.backward()
                self.net.step()

                #Update S in lowrank if network contains.
                if self.net._contains_lowrank:
                    self.net.step(s=True)
                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i+1}/{self._iterations}], Step [{step+1}/{len(self._trainloader)}], Loss: {loss.item():.4f}')

            if show_progress:
                self.test()
            #TODO Isse with this not running as intende. No 
            

            accuracy = correct / total
            print(f'Epoch [{i+1}/{self._iterations}], Validation Accuracy: {100 * accuracy:.2f}%')