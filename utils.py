import torch


def adjacency_norms(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[-1]
    ones = torch.ones(n).unsqueeze(1)
    deg = adjacency @ ones
    deg = deg.squeeze()

    return deg


def degree(adjacency: torch.Tensor) -> torch.Tensor:
    assert adjacency.shape[-2] == adjacency.shape[-1]
    n = adjacency.shape[-1]

    deg = adjacency_norms(adjacency)

    if adjacency.is_sparse:
        idx = torch.linspace(0, n - 1, n, dtype=torch.int32)
        idx = torch.stack([idx, idx])
        return torch.sparse_coo_tensor(idx, deg)

    else:
        return torch.diag(deg)


def random_adj(n: int, e: int, n_r: int = 0) -> torch.Tensor:
    edge_idx = torch.randint(low=0, high=n, size=(2, e))
    val = torch.ones(e)

    if n_r:
        rel = torch.randint(low=0, high=n_r, size=(1, e))
        edge_idx = torch.cat([rel, edge_idx], dim=0)

        return torch.sparse_coo_tensor(indices=edge_idx, values=val, size=(n_r, n, n))

    return torch.sparse_coo_tensor(indices=edge_idx, values=val, size=(n, n))


def random_labels(X: torch.Tensor, n_labels: int) -> torch.Tensor:
    """
    A dumb transformation that can be approximated by Graph Convolution.
    :param X: Input features of size [n, dim0]
    :param n_labels: Tensor of size [n_labels]
    :return:
    """
    with torch.no_grad():
        dim0 = X.shape[-1]

        W = torch.rand(n_labels, dim0)
        H = X @ W.T
        Y = torch.argmax(H, dim=-1)
        return Y


def glorot_init(param: torch.Tensor):
    torch.nn.init.xavier_uniform_(param)
