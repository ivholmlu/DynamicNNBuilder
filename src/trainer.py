from src.network import NeuralNetwork
from src.layers import LayerFactory
import torch.nn as nn
import toml
from src.loader import Loader
import torch

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
        
    def train(self, show_progress=True):
        for i in range(self._iterations):
            for step, (images, labels) in enumerate(self._trainloader):
                out = self.net(images)
                loss = self._criterion(out, labels)
                loss.backward()
                self.net.step()
                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i+1}/{self._iterations}], Step [{step+1}/{len(self._trainloader)}], Loss: {loss.item():.4f}')

            total = 0
            correct = 0
            #TODO Isse with this not running as intende. No 
            with torch.no_grad():
                for images, labels in self._testloader:
                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Epoch [{i+1}/{self._iterations}], Validation Accuracy: {100 * accuracy:.2f}%')