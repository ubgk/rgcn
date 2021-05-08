import torch


def adjacency2degree(adjacency: torch.Tensor, dim=1) -> torch.Tensor:
    degree = torch.sum(adjacency, dim=dim)
    degree = torch.diag(degree)
    return degree


def avg_adjacency(adjacency: torch.Tensor, dim=1) -> torch.Tensor:
    degree = adjacency2degree(adjacency, dim=dim)
    avg_adj = torch.inverse(degree) @ adjacency
    avg_adj = avg_adj * 10
    avg_adj = torch.round(avg_adj) / 10.
    return  avg_adj