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
        self.device = torch.device("cpu")

    
    def test(self, epoch):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(self._testloader):
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{self._iterations}], Validation Accuracy: {100 * accuracy:.2f}%')

    def train(self, show_progress=True):
        for epoch in range(self._iterations):
            self.net.train()
            for batch, (images, labels) in enumerate(self._trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                #Forward pass
                out = self.net(images)
                loss = self._criterion(out, labels)

                ### Update loss for network
                loss.backward()

                #Update network(Not s in lowrank)
                self.net.step(s=False)
                
                #Update S in lowrank if network contains lowrank layers.
                if self.net._contains_lowrank:
                    #Forward pass
                    out = self.net(images)
                    loss = self._criterion(out, labels)
                    
                    #Calculate gradients
                    loss.backward()

                    #Update coefficients
                    self.net.step(s=True)
                
                if (batch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self._iterations}], Step [{batch+1}/{len(self._trainloader)}], Loss: {loss.item():.4f}')
                    
            if show_progress:
                self.test(epoch)

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        print(f"Parameters saved to {path}")

    
