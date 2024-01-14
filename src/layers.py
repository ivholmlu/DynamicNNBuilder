import torch
import torch.nn as nn

class ActivationFactory:
    def __init__(self):
        try: 
            self.activations = {"relu": nn.ReLU(), "linear": nn.Identity()}
        except KeyError as e:
            print(f"Activation {e} is not a defined activation")

    def __call__(self, type):
        return self.activations[type]
    


class LayerFactory:
    def __init__(self):
        self.classes = {"dense" : Denselayer, 
                        "vanillalowrank" : VanillaLowRank,
                        "lowrank" : LowRank}
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
    
    @torch.no_grad()
    def step(self, s):
        if not s:
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
    
    @torch.no_grad
    def step(self, s):
        if not s:
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
        self._r = config["rank"]
        self.lr = lr
        activation = ActivationFactory()
        self.activation = activation(config["activation"])

        #Initiating paramerts with gradient
        self._U = nn.Parameter(torch.randn(config["dim_in"], self._r))
        self._S = nn.Parameter(torch.randn(self._r, self._r))
        self._V = nn.Parameter(torch.randn(config["dim_out"], self._r))
        self._b = nn.Parameter(torch.randn(config["dim_out"]))

        U1, _ = torch.linalg.qr(self._U, 'reduced')
        V1, _ = torch.linalg.qr(self._V, 'reduced')
        self._U.data = U1
        self._V.data = V1

        #initiating 'copies' of U and V to be used in step
        self._U1 = nn.Parameter(
                                torch.randn(config["dim_in"], self._r),
                                requires_grad=False)
        self._V1 = nn.Parameter(
                                torch.randn(config["dim_out"], self._r),
                                requires_grad=False)

    def forward(self, X):
        r = self._r
        xU = torch.matmul(X, self._U[:,:r])
        xUS = torch.matmul(xU, self._S[:r,:r])
        out = torch.matmul(xUS, self._V[:,:r].T) + self._b
        return self.activation(out)

    @torch.no_grad()
    def step(self, s):
        lr = self.lr

        if not s:
            r = self._r #Rank
            # UPDATING K(Finding new U)
            K = torch.matmul(self._U, self._S)
            dK = torch.matmul(self._U.grad, self._S)
            K = K - lr * dK
            self._U1.data , _ = torch.linalg.qr(K, "reduced") #R is not used

            #Updating L
            L = torch.matmul(self._V, self._S.T)
            dL = torch.matmul(self._V.grad[:, :r], self._S.T)
            L = L - lr * dL
            self._V1.data , _ = torch.linalg.qr(L, "reduced") # R is not used

            #Creating N and M
            M = self._U1.T @ self._U
            N = self._V.T @ self._V1

            # Updating S, U and V
            self._S.data = M @ self._S @ N
            self._U.data = self._U1
            self._V.data = self._V1
            self._b.data = self._b - lr * self._b.grad

            self._U.grad.zero_()
            self._S.grad.zero_()
            self._V.grad.zero_()
            self._b.grad.zero_()
            
        # Update new S value (Done with new loss calculated)
        else:
            self._S.data = self._S - lr * self._S.grad




