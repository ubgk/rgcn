import torch
import torch.nn as nn

from utils import degree, adjacency_norms


class MessageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, name=None, bias=False):
        super().__init__()
        self.name = name
        self.transform = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X, A, normalise=True):
        X = self.transform(X)
        X = A @ X

        if normalise:
            norm_vec = adjacency_norms(A).unsqueeze(-1) + 1e-30
            norm_vec = norm_vec.detach()
            return X / norm_vec

        return X


if __name__ == '__main__':
    adjacency = [
        [0., 0., 0., 0., 0.],  # Source: Node1
        [1., 0., 1., 0., 0.],  # Source: Node2
        [0., 1., 0., 1., 1.],  # Source: Node3
        [0., 0., 1., 0., 0.],  # Source: Node4
        [0., 0., 1., 0., 0.]  # Source: Node5
    ]
    torch.manual_seed(15)
    adjacency = torch.tensor(adjacency).to_sparse()
    nodes = [1., 2., 3., 4., 5.]
    # nodes = [2.] * 5

    nodes = torch.rand(5, 100)
    # nodes = torch.tensor(nodes)
    t = adjacency @ nodes
    print(t.shape)
    print(degree(adjacency))

    layer = MessageLayer(100, 4)
    print(layer(nodes, adjacency))
