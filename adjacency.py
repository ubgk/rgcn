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
    n = 169343
    e = 1166243
    n_r = 5

    torch.manual_seed(15)

    pos = torch.randint(low=0, high=n, size=(2, e))
    val = torch.randint(low=0, high=n_r, size=(e, 1))

    A = AdjacencyMatrix(edge_idx=pos, relation_type=val, n_nodes=n)
    A[0]
