import torch
import torch.nn as nn

class ActivationFactory:
    def __init__(self):
        self.activations = {"relu": nn.ReLU(), "linear": nn.Identity()}
    def __call__(self, type):
        return self.activations[type]
    

class LayerFactory:
    def __init__(self):
        self.classes = {"dense" : Denselayer, 
                        "vanillalowrank" : VanillaLowRank,
                        "lowRank" : LowRank}
    def __call__(self, config, lr):
        
            print(config["type"])
            return self.classes[config["type"]](config, lr)
        

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
    
    @torch.no_grad
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
        W = torch.matmul(torch.matmul(self._U, self._S), self._VT)
        return self.activation(torch.matmul(X, W) + self._b)
    
    def step(self):
        self._U.data = self._U - self.lr*self._U.grad
        self._S.data = self._S - self.lr*self._S.grad
        self._VT.data = self._VT - self.lr*self._VT.grad
        self._b.data = self._b - self.lr*self._b.grad
        self._U.grad.zero_()
        self._S.grad.zero_()
        self._VT.grad.zero_()
        self._b.grad.zero_()

class LowRank(nn.Module):
    def __init__(self, config, lr) -> None:
        super(LowRank, self).__init__()
        self._U = nn.Parameter(torch.randn(config["dim_in"], config["rank"]), requires_grad=True)
        self._S = nn.Parameter(torch.randn(config["rank"], config["rank"]), requires_grad=True)
        self._VT = nn.Parameter(torch.randn(config["rank"], config["dim_out"]), requires_grad=True)

        self.lr = lr
        activation = ActivationFactory()
        self.activation = activation(config["activation"])

    def forward(self, X):
        pass

    def step(self):
        pass