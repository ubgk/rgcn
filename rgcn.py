import torch
import torch.nn as nn

import utils
from utils import adjacency_norms


class RGCNDirectional(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 n_relations: int, decomposition: str = None, n_bases: int = 11, activation=nn.Identity):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.decomposition = decomposition
        self.n_bases = n_bases
        self.activation = activation()
        self.n_relations = n_relations

        # Trainable parameters
        if decomposition is None:
            self.weights = nn.Parameter(self.no_decompose_init())
        elif decomposition == 'basis':
            _v, basis_coeffs = self.scalar_decompose_init()
            self.bases, self.basis_coeffs = nn.Parameter(_v), nn.Parameter(basis_coeffs)

            print(self.bases.shape)
        else:
            raise ValueError(f'"{decomposition}" is not a known decomposition method!')

    def no_decompose_init(self):
        # Initialise the transformation matrices
        weights = torch.Tensor(self.n_relations, self.out_dim, self.in_dim)
        utils.glorot_init(weights)
        return weights

    def scalar_decompose_init(self):
        bases = torch.Tensor(self.n_bases, self.out_dim, self.in_dim)
        utils.glorot_init(bases)

        basis_coeffs = torch.Tensor(self.n_relations, self.n_bases)

        return bases, basis_coeffs

    def forward(self, input):
        X, A = input
        assert X.dim() == 2

        if self.decomposition is None:
            weights = self.weights
        else:
            bases = self.bases.expand(self.n_relations, self.n_bases, self.bases.shape[-2], self.bases.shape[-1])
            basis_coeffs = self.basis_coeffs.view(self.n_relations, self.n_bases, 1, 1)

            weights = bases * basis_coeffs
            weights = torch.sum(weights, dim=1)

        out = self._forward_relation(X, A, weights)

        return self.activation(out)

    def _forward_relation(self, X, A, W):
        assert A.shape[0] == self.n_relations

        n = X.shape[0]
        out = torch.zeros(n, self.out_dim)
        for r in range(self.n_relations):
            Ar = A[r].detach()
            H_r = X @ W[r].T
            x = Ar @ H_r

            norm_vec = adjacency_norms(Ar).unsqueeze(-1) + 1e-12
            out += x / norm_vec.detach()

        return out


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_relations: int, activation=nn.ReLU, bias: bool = True, **kwargs):
        super().__init__()
        # Nodes toward which the source nodes point
        self.neighbour_transform = RGCNDirectional(in_dim=in_dim, out_dim=out_dim, n_relations=n_relations * 2,  **kwargs)

        # Self-loops
        self.loop_transform = nn.Linear(in_dim, out_dim, bias=bias)

        self.activation = activation()

    def forward(self, input):
        X, A = input
        At = torch.transpose(A, -1, -2)

        A = torch.cat([A, At], dim=0)

        h_j = self.neighbour_transform((X, A))
        h_self = self.loop_transform(X)

        out = h_j + h_self
        out = self.activation(out)

        return out, A

if __name__ == '__main__':
    n = 169343
    e = 1166243
    d_0 = 128
    d_1 = 64
    n_r = 5

    X = torch.rand(n, d_0)
    A = utils.random_adj(n, e, n_r)
    print(A.shape)

    W = torch.rand(n_r, d_1, d_0)
    B = torch.rand(n_r, d_1)

    RGCN(d_0, d_1, A, decomposition='basis', bias=True)
