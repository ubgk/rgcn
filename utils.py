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


def avg_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency2degree(adjacency)
    avg_adj = torch.inverse(degree) @ adjacency

    return avg_adj

def random_adj(n: int, e:int) -> torch.Tensor:
    pos = torch.randint(n, (2, e))
    val = torch.ones(e)

    return torch.sparse_coo_tensor(pos, val)

def glorot_init(param: torch.Tensor):
    torch.nn.init.xavier_uniform_(param)
