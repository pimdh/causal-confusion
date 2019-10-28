import torch
import torch.nn as nn


class MLP(nn.Module):
    """Helper module to create MLPs."""
    def __init__(self, dims, activation=nn.ReLU):
        super().__init__()
        blocks = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
            blocks.append(nn.Linear(dim_in, dim_out))

            if i < len(dims)-2:
                blocks.append(activation())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SimplePolicy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state, mask):
        return self.net(state)


class UniformMaskPolicy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state, mask):
        mask = mask.to(state).expand_as(state)
        x = torch.cat([state * mask, mask], 1)
        return self.net(x)
