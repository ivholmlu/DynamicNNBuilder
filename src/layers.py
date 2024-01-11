import torch
import torch.nn as nn

class ActivationFactory:
    def __init__(self):
        self.types = {"relu": nn.ReLU(), "linear": nn.Identity()}
    def __call__(self, type):
        return self.types[type]

class Denselayer(nn.Module):
    
    def __init__(self, config, lr) -> None:
        super(Denselayer, self).__init__()

        self._W = nn.Parameter(torch.randn(config["dim_in"], config["dim_out"]), requires_grad=True)
        self._b = nn.Parameter(torch.randn(config["dim_out"]), requires_grad=True)
        activation = ActivationFactory()
        self.activation = activation(config["activation"])
        self.lr = lr

    def forward(self, X):
        return self.activation(torch.matmul(X, self._W) + self._b)
    
    def step(self):
        self._W.data = self._W - self.lr*self._W.grad
        self._b.data = self._b - self.lr*self._b.grad
        self._W.grad.zero_()
        self._b.grad.zero_()

class VanillaLowRank(nn.Module):
    
    def __init__(self, config, lr) -> None:
        super(VanillaLowRank, self).__init__()

        self._U = nn.Parameter(torch.randn(config["dim_in"], config["rank"]), requires_grad=True)
        self._S = nn.Parameter(torch.randn(config["rank"], config["rank"]), requires_grad=True)
        self._VT = nn.Parameter(torch.randn(config["rank"], config["dim_out"]), requires_grad=True)

        self._b = nn.Parameter(torch.randn(config["dim_out"]), requires_grad=True)

        activation = ActivationFactory()
        self.activation = activation(config["activation"])
        self.lr = lr

    def forward(self, X):
        return self.activation(torch.matmul(X, self._W) + self._b)
    
    def step(self):
        self._U.data = self._U - self.lr*self._U.grad
        self._S.data = self._S - self.lr*self._S.grad
        self._U.data = self._VT - self.lr*self._VT.grad
        self._b.data = self._b - self.lr*self._b.grad
        self._W.grad.zero_()
        self._b.grad.zero_()