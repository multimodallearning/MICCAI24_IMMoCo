import torch
import torch.nn as nn
import torch.nn.functional as F


class TV_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss = 0
        for x in [x.real, x.imag]:
            loss += torch.sum(torch.abs(x[:, :-1] - x[:, 1:])) + torch.sum(
                torch.abs(x[:-1, :] - x[1:, :])
            )

        return loss


class GradientEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, x):
        return -torch.sum(torch.mul(x, torch.log(x + 1e-24)))

    def forward(self, x):

        dx = (x[:, :-1] - x[:, 1:]).abs()
        dy = ((x[:-1, :] - x[1:, :])).abs()

        # pad the gradient
        dx = F.pad(dx, (0, 1, 0, 0), mode="constant", value=0)
        dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0)

        gradient = dx + dy

        loss = self.entropy(gradient)

        return loss
