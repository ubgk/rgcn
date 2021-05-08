import torch
import torch.nn as nn

from utils import adjacency2degree, avg_adjacency


class Transform(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU):
        super().__init__()
        self.transform = nn.Linear(in_dim, out_dim)
        self.act = act()

    def forward(self, input):
        x = self.transform(input)
        return self.act(x)


class MessageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_relation: int):
        super().__init__()

        self.transforms = [Transform(in_dim, out_dim) for _ in range(n_relation + 1)]


    def forward(self, x, adjacency, ):
        pass


adjacency = [
    [0., 1., 0., 0., 0.],  # Source: Node1
    [1., 0., 1., 0., 0.],  # Source: Node2
    [0., 1., 0., 1., 1.],  # Source: Node3
    [0., 0., 1., 0., 0.],  # Source: Node4
    [0., 0., 1., 0., 0.]  # Source: Node5
]
adjacency = torch.tensor(adjacency)
nodes = [1., 2., 3., 4., 5.]
nodes = torch.tensor(nodes)

message = adjacency @ nodes
degree = adjacency2degree(adjacency)

avg_adj = avg_adjacency(adjacency)
print(avg_adj @ nodes)
#
# print(adjacency)
# print(torch.inverse(degree))
