from src.network import NeuralNetwork
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import toml

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
config = toml.load("config.toml")
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=config["settings"]["batch_size"], shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=config["settings"]["batch_size"], shuffle=False)

config = toml.load("config.toml")
criterion = nn.CrossEntropyLoss()
flatten = nn.Flatten()
net = NeuralNetwork(config)

num_iterations = config["settings"]["iterations"]

for i in range(num_iterations):
    for step, (images, labels) in enumerate(train_loader):
        images = flatten(images)
        out = net(images)
        loss = criterion(out, labels)
        loss.backward()
        net.step()
        if (step + 1) % 100 == 0:
            print(f'Epoch [{i+1}/{num_iterations}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = flatten(images)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{i+1}/{num_iterations}], Validation Accuracy: {100 * accuracy:.2f}%')
