"""Module for layers of the neural network."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActivationFactory:
    """Factory for activation functions."""
    def __init__(self) -> None:
        try:
            self.activations = {"relu": nn.ReLU(), "linear": nn.Identity()}
        except KeyError as e:
            print(f"Activation {e} is not a defined activation")

    def __call__(self, type) -> nn.Module:
        return self.activations[type]


class LayerFactory:
    """Factory for layers of the neural network."""
    def __init__(self) -> None:
        self.classes = {"dense": Denselayer,
                        "vanillalowrank": VanillaLowRank,
                        "lowrank": LowRank}

    def __call__(self, config, lr=0, load=False) -> nn.Module:
        return self.classes[config["type"]](config, lr, load)

class BaseLayer(nn.Module, ABC):

    def __init__(self, config, lr, load=False) -> None:
        super(BaseLayer, self).__init__()
        activation = ActivationFactory()
        if not load:
            self.activation = activation(config["activation"])
            self.lr = lr
            self._b = nn.Parameter(torch.randn(
                config["dim_out"]),
                requires_grad=True)
        else:
            self._b = config["attributes"]["_b"]
            self.activation = activation(config["activation"])

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def step(self, s):
        pass

    @property
    def b(self):
        return self._b

#Needs b implemented, and 

class Denselayer(nn.Module):
    """Dense layer of the neural network."""
    def __init__(self, config, lr, load=False) -> None:
        super().__init__(config, lr, load)
        activation = ActivationFactory()

        if not load:
            self._W = nn.Parameter(torch.randn(
                config["dim_in"],
                config["dim_out"]),
                requires_grad=True)

        else:
            self._b = config["attributes"]['_b']
            self._W = config["attributes"]['_W']

    def forward(self, X):
        return self.activation(torch.matmul(X, self._W) + self._b)

    @torch.no_grad()
    def step(self, s):
        if not s:
            self._W.data = self._W - self.lr * self._W.grad
            self._b.data = self._b - self.lr * self._b.grad
            self._W.grad.zero_()
            self._b.grad.zero_()
    
    @property
    def b(self):
        return self._b

    @property
    def W(self):
        return self._W


class VanillaLowRank(nn.Module):
    """Vanilla low rank layer of the neural network."""
    def __init__(self, config, lr, load=False) -> None:
        super(VanillaLowRank, self).__init__()
        activation = ActivationFactory()
        if not load:
            self._U = nn.Parameter(torch.randn(
                config["dim_in"], config["rank"]), requires_grad=True)
            self._S = nn.Parameter(torch.randn(
                config["rank"], config["rank"]), requires_grad=True)
            self._V = nn.Parameter(torch.randn(
                config["dim_out"], config["rank"]), requires_grad=True)
            self._b = nn.Parameter(torch.randn(
                config["dim_out"]), requires_grad=True)

            U1, _ = torch.linalg.qr(self._U, 'reduced')
            V1, _ = torch.linalg.qr(self._V, 'reduced')
            self._U.data = U1
            self._V.data = V1

            self.activation = activation(config["activation"])
            self.lr = lr

        else:
            self._U = config["attributes"]['_U']
            self._S = config["attributes"]['_S']
            self._V = config["attributes"]["_V"]
            self._b = config["attributes"]["_b"]
            self._r = self._S.size()[0]

            self.activation = activation(config["activation"])

    def forward(self, X) -> torch.Tensor:
        W = torch.matmul(torch.matmul(self._U, self._S), self._V.T)
        return self.activation(torch.matmul(X, W) + self._b)

    @torch.no_grad()
    def step(self, s) -> None:
        if not s:
            self._U.data = self._U - self.lr * self._U.grad
            self._S.data = self._S - self.lr * self._S.grad
            self._V.data = self._V - self.lr * self._V.grad
            self._b.data = self._b - self.lr * self._b.grad

            self._U.grad.zero_()
            self._S.grad.zero_()
            self._V.grad.zero_()
            self._b.grad.zero_()

    @property
    def U(self):
        return self._U
    
    @property
    def S(self):
        return self._S
    
    @property
    def V(self):
        return self._V

    @property
    def b(self):
        return self._b


class LowRank(nn.Module):
    """Low rank layer of the neural network."""
    def __init__(self, config, lr, load=False) -> None:
        super(LowRank, self).__init__()
        activation = ActivationFactory()
        if not load:
            self._r = config["rank"]
            self.lr = lr
            activation = ActivationFactory()
            self.activation = activation(config["activation"])

            self._U = nn.Parameter(torch.randn(config["dim_in"], self._r))
            self._S = nn.Parameter(torch.randn(self._r, self._r))
            self._V = nn.Parameter(torch.randn(config["dim_out"], self._r))
            self._b = nn.Parameter(torch.randn(config["dim_out"]))

            U1, _ = torch.linalg.qr(self._U, 'reduced')
            V1, _ = torch.linalg.qr(self._V, 'reduced')
            self._U.data = U1
            self._V.data = V1

            self._U1 = nn.Parameter(torch.randn(
                config["dim_in"], self._r), requires_grad=False)
            self._V1 = nn.Parameter(torch.randn(
                config["dim_out"], self._r), requires_grad=False)

        else:
            self._U = config["attributes"]["_U"]
            self._S = config["attributes"]["_S"]
            self._V = config["attributes"]["_V"]
            self._b = config["attributes"]["_b"]
            self._r = self._S.size()[0]
            self.activation = activation(config["activation"])

    def forward(self, X) -> torch.Tensor:
        r = self._r
        xU = torch.matmul(X, self._U[:, :r])
        xUS = torch.matmul(xU, self._S[:r, :r])
        out = torch.matmul(xUS, self._V[:, :r].T) + self._b
        return self.activation(out)

    @torch.no_grad()
    def step(self, s) -> None:
        lr = self.lr

        if not s:
            r = self._r
            K = torch.matmul(self._U, self._S)
            dK = torch.matmul(self._U.grad, self._S)
            K = K - lr * dK

            self._U1.data, _ = torch.linalg.qr(K, "reduced")
            L = torch.matmul(self._V, self._S.T)
            dL = torch.matmul(self._V.grad[:, :r], self._S.T)
            L = L - lr * dL
            self._V1.data, _ = torch.linalg.qr(L, "reduced")

            M = torch.matmul(self._U1.T, self._U)
            N = torch.matmul(self._V.T, self._V1)

            self._S.data = M @ self._S @ N
            self._U.data = self._U1
            self._V.data = self._V1
            self._b.data = self._b - lr * self._b.grad

            self._U.grad.zero_()
            self._S.grad.zero_()
            self._V.grad.zero_()
            self._b.grad.zero_()

        else:
            self._S.data = self._S - lr * self._S.grad
    
    @property
    def U(self):
        return self._U
    
    @property
    def S(self):
        return self._S
    
    @property
    def V(self):
        return self._V
    
    @property
    def b(self):
        return self._b
