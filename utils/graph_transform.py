import torch
import dgl
import torch_geometric
from torch_sparse import SparseTensor

def remove_self_loop(edge_index):
    assert edge_index.shape[0] == 2
    edges = dgl.graph((edge_index[0], edge_index[1])).to(edge_index.device)
    edges = edges.add_self_loop().remove_self_loop()
    edges = [a.long() for a in edges.edges()]
    edge_index = torch_geometric.data.Data(edge_index=torch.stack(edges)).edge_index
    return edge_index

# D-1/2 * A * D-1/2 or D-1 * A
def sparse_normalize(adj, symmetric_norm=True):
    assert isinstance(adj, SparseTensor)
    size = adj.size(0)
    ones = torch.ones(size).view(-1, 1).to(adj.device())
    degree = adj @ ones
    if symmetric_norm == False:
        degree = degree ** -1
        degree[torch.isinf(degree)] = 0
        return adj * degree
    else:
        degree = degree ** (-1/2)
        degree[torch.isinf(degree)] = 0
        d = SparseTensor(row=torch.arange(size), col=torch.arange(size), value=degree.squeeze().cpu(),
                         sparse_sizes=(size, size)).to(adj.device())
        adj = adj @ d
        adj = adj * degree
        return adj

# D-1/2 * A * D-1/2
def nomarlizeAdj(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj

# D-1 * A
def normalizeLelf(adj):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj
    return adj
