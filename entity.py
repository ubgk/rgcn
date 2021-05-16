from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from rgcn import RGCNLayer


class EntityRGCN(nn.Module):
    def __init__(self, dims: List[int], n_relations: int, **kwargs):
        super().__init__()

        n_dims = len(dims)
        layers = [RGCNLayer(dims[i], dims[i + 1], n_relations=n_relations, **kwargs)
                  for i in range(n_dims - 1)]

        self.sequential = nn.Sequential(*layers)

    def forward(self, input):
        return self.sequential(input)[0]

    def fit_one_cycle(self, n, X, Y, labeled_idx, A, loss=nn.CrossEntropyLoss(), optim=torch.optim.SGD):
        optim = optim(params=self.parameters(), lr=1e-3)
        for i in tqdm(range(n)):
            out = self((X, A))[labeled_idx]
            l = loss(out, Y)
            l.backward()
            print(l.item())
            optim.step()
            optim.zero_grad()


if __name__ == '__main__':
    n = 169343
    e = 1166243
    n_r = 2

    dims = [128, 10]

    X = torch.rand(n, dims[0])
    A = utils.random_adj(n, e, n_r)

    n_labeled = 200
    labeled_idx = torch.randint(low=0, high=n, size=(n_labeled, 1)).squeeze()
    Y = utils.random_labels(X, 10)[labeled_idx]
    net = EntityRGCN(dims=dims, decomposition=None,
                     n_relations=n_r, bias=False, n_bases=3)

    for p in net.named_children():
        print(p)

    net.fit_one_cycle(250, X, Y, labeled_idx, A)
