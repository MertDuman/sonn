import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperActivation2d(nn.Module):
    def __init__(self, in_channels, q):
        super().__init__()
        self.in_channels = in_channels
        self.q = q
        self.w = nn.Parameter(torch.zeros(self.in_channels, self.q))
        with torch.no_grad():
            self.w[:, 0] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        for i in range(1, self.q + 1):
            y.append(torch.einsum("c,nchw->nchw", self.w[:, i-1], (x**i)))
        return torch.stack(y).sum(dim=0)  # wq * x^q + wq-1 * x^q-1 + ...

    def get_poly(self, xmin=-10, xmax=10, ymin=-1, ymax=1, n=500):
        xmin = -10 if xmin is None else xmin
        xmax = 10 if xmax is None else xmax
        ymin = float("-inf") if ymin is None else ymin
        ymax = float("inf") if ymax is None else ymax
        n = 500 if n is None else n

        xx = torch.linspace(xmin, xmax, n).reshape(n, 1, 1, 1).tile(1, self.in_channels, 1, 1)
        with torch.no_grad():
            yy = self.forward(xx)
        xx = xx.squeeze()
        yy = yy.squeeze()

        out = []
        for i in range(self.in_channels):
            idx = torch.where(torch.bitwise_and(ymin <= yy[:, i] , yy[:, i] <= ymax))[0]
            out.append((xx[idx, i], yy[idx, i]))

        return out


class SuperActivation(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.w = nn.Parameter(torch.zeros(self.q))
        with torch.no_grad():
            self.w[0] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        for i in range(1, self.q + 1):
            y.append(self.w[i-1] * (x**i))
        return torch.stack(y).sum(dim=0)  # wq * x^q + wq-1 * x^q-1 + ...

    def get_poly(self, xmin=-10, xmax=10, ymin=-1, ymax=1, n=500):
        xmin = -10 if xmin is None else xmin
        xmax = 10 if xmax is None else xmax
        ymin = float("-inf") if ymin is None else ymin
        ymax = float("inf") if ymax is None else ymax
        n = 500 if n is None else n

        xx = torch.linspace(xmin, xmax, n)
        with torch.no_grad():
            yy = self.forward(xx)
        idx = torch.where(torch.bitwise_and(ymin <= yy , yy <= ymax))[0]

        return xx[idx], yy[idx]