import torch
import torch.nn as nn
from tqdm import tqdm

from message_passing import MessageLayer
from utils import random_adj


class RelationalLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias=False):
        super().__init__()
        self.source_layer = MessageLayer(in_dim, out_dim, bias=bias, name='in')
        self.out_layer = MessageLayer(in_dim, out_dim, bias=bias, name='out')

    def forward(self, X, A, normalise=True):
        X_in = self.source_layer(X, A, normalise=normalise)
        X_out = self.out_layer(X, A.t(), normalise=normalise)

        return X_in + X_out


class RGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adjacency: torch.Tensor, bias=False,
                 activation=nn.ReLU):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = adjacency.shape[0]

        # Trainable parameters
        self.loop_transform = nn.Linear(in_dim, out_dim, bias=bias)
        self.rel_transforms = [RelationalLayer(in_dim, out_dim, bias) for _ in range(adjacency.shape[0])]
        self.rel_transforms = nn.ModuleList(self.rel_transforms)

        self.adjacency = adjacency
        self.activation = activation()

    def forward(self, X):
        A = self.adjacency
        assert A.shape[0] == self.n_relations

        res = self.loop_transform(X)

        for rel in (range(self.n_relations)):
            adj = A[rel].detach()
            res += self.rel_transforms[rel](X, adj)

        res = self.activation(res)

        return res


if __name__ == '__main__':
    n = 169343
    e = 1166243
    d_0 = 128
    d_1 = 64
    n_r = 5

    X = torch.rand(n, d_0)
    A = [random_adj(n, e) for i in range(n_r)]
    A = torch.stack(A)

    rgcn = RGCN(d_0, d_1, n_r)
    res = rgcn(X, A)

    print(res)
