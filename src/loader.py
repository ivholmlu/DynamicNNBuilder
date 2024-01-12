from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Loader:

    def __init__(self):
        self._root = './data'
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def load_dataset(self, config):

        train_dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config["settings"]["batch_size"], shuffle=True)

        test_dataset = datasets.MNIST(root='./data', train=False, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=config["settings"]["batch_size"], shuffle=False)

        return train_loader, test_loader