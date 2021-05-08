import torch
import torch.nn as nn

n_features = 4
n_nodes = 6
n_relations = 1
dim_1 = 3

# Transformations
W = torch.rand(n_relations + 1,  dim_1, n_features)

# Initial States
H = torch.rand(n_features, n_nodes)

# Adjacency
A = torch.rand(n_relations, n_nodes, n_nodes) > 0.6
A = A * (1. - torch.eye(n_nodes))  # mask off the diagonal

auto_adjacency = torch.eye(n_nodes).unsqueeze(0)
A = torch.cat([auto_adjacency, A], dim=0)

print(A.shape)

H_1_for = torch.zeros(dim_1, n_nodes)
H_1 = torch.zeros(dim_1, n_nodes)

lin = nn.Linear(100, 5)

print(A)

## For loop
for i_node in range(n_nodes):
    auto_transform = W[0, :, :]
    auto_vec = H[:, i_node]
    new_self_vec = auto_transform @ auto_vec

    neighbours = A[:, i_node, :] != False
    for rel in range(1, n_relations+1):
        r_neigh = neighbours[rel, :].nonzero()

        r_vec = torch.zeros(dim_1, 1)
        for j_node in r_neigh:
            r_vec += W[rel, :, :] @ H[:, j_node]

        new_self_vec += r_vec.squeeze()

    H_1_for[:, i_node] = new_self_vec

## Matrices
for rel in range(0, n_relations+1):
    Ar = A[rel]
    Wr = W[rel]
    Hadj = Ar @ H.T
    H_1 += Wr @ Hadj.T

print("RESULTS")
print(H_1)
print(H_1_for)

print(torch.allclose(H_1, H_1_for))