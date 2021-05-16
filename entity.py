from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from rgcn import RGCN


class EntityRGCN(nn.Module):
    def __init__(self, dims: List[int], adjacency: torch.Tensor):
        super().__init__()

        n_dims = len(dims)
        layers = [RGCN(dims[i], dims[i + 1], adjacency=adjacency) for i in range(n_dims - 1)]

        self.sequential = nn.Sequential(*layers)

    def forward(self, X):
        return self.sequential(X)

    def fit_one_cycle(self, n, X, Y, labeled_idx, loss=nn.CrossEntropyLoss(), optim=torch.optim.SGD):
        optim = optim(params=self.parameters(), lr=1e-3)
        for i in tqdm(range(n)):
            out = self(X)[labeled_idx]
            l = loss(out, Y)
            l.backward()
            print(l.item())
            optim.step()
            optim.zero_grad()


if __name__ == '__main__':
    n = 169343
    e = 1166243
    n_r = 3

    dims = [128, 10]

    X = torch.rand(n, dims[0])
    A = utils.random_adj(n, e, n_r)

    n_labeled = 200
    labeled_idx = torch.randint(low=0, high=n, size=(n_labeled, 1)).squeeze()
    # Y = torch.randint(low=0, high=9, size=(n_labeled, 1)).squeeze()
    Y = utils.random_labels(X, 10)[labeled_idx]
    net = EntityRGCN(dims=dims, adjacency=A)

    for p in net.named_children():
        print(p)

    net.fit_one_cycle(250, X, Y, labeled_idx)
