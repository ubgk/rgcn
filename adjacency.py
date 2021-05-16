from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class AdjacencyMatrix:
    edge_idx: torch.Tensor
    relation_type: torch.Tensor
    n_nodes: Union[int, None]

    def __post_init__(self):
        if self.n_nodes is None:
            self.n_nodes = len(self.edge_idx)

        self.relation_type = self.relation_type.squeeze()

    def __getitem__(self, rel) -> torch.Tensor:
        """
        :param rel: Relation type to return the adjacency matrix for
        :return: Adjacency matrix of size [n_nodes, n_nodes] with value 1 for existing edges, and zeroes elsewhere
        """
        rel_idx = (self.relation_type == rel)
        rel_idx = rel_idx.nonzero()
        rel_idx = rel_idx.squeeze()
        edges = self.edge_idx.T[rel_idx]
        vals = torch.ones(len(edges))

        return torch.sparse_coo_tensor(indices=edges.T, values=vals, size=(self.n_nodes, self.n_nodes))


if __name__ == '__main__':
    n = 50
    e = 300
    n_r = 3

    torch.manual_seed(15)

    pos = torch.randint(low=0, high=n, size=(2, e))
    rel = torch.randint(low=0, high=n_r, size=(1, e))
    val = torch.ones(e)

    edge_idx = torch.cat([rel, pos], dim=0)

    A = torch.sparse_coo_tensor(indices=edge_idx, values=val, size=(n_r, n,n))

