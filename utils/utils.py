import torch
import numpy as np
import random
from torch import nn

def set_seed(manualSeed):
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    CUDA = True if torch.cuda.is_available() else False
    if CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(manualSeed)

def get_accuracy(output, labels):
    if len(labels.shape) == 2:
        labels = torch.argmax(labels, dim=1)
    pred_y = torch.max(output, dim=1)[1]
    correct = torch.sum(pred_y == labels)
    return correct.item() * 1.0 / len(labels)


class BarlowLoss(nn.Module):
    def __init__(self, alpha=5e-4):
        """
        Implementation of the loss described in the paper
        Barlow Twins: Self-Supervised Learning via Redundancy Reduction
        https://arxiv.org/abs/2103.03230
        :param alpha: float
        """
        super(BarlowLoss, self).__init__()
        self.alpha = alpha

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, view1, view2):
        """
        :param view1: torch.Tensor, shape [batch_size, projection_dim]
        :param view2: torch.Tensor, shape [batch_size, projection_dim]
        :return: torch.Tensor, scalar
        """
        N = view1.shape[0]
        d = view1.shape[1]

        z1 = (view1 - view1.mean(0)) / view1.std(0)
        z2 = (view2 - view2.mean(0)) / view2.std(0)

        c = torch.mm(z1.T, z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        ssl_loss = on_diag + self.alpha * off_diag / d
        return ssl_loss




