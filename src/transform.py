import torch
from torch_geometric.utils import degree, clustering

def add_node_degree_feature(data):
    row, col = data.edge_index
    deg = degree(row, num_nodes=data.num_nodes).unsqueeze(1)

    mean = deg.mean()
    std = deg.std()
    if std > 0:
        deg = (deg - mean) / std
    else:
        deg = deg - mean

    if data.x is not None and data.x.numel() > 0:
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        data.x = torch.cat([data.x.float(), deg.float()], dim=1)
    else:
        data.x = deg.float()

    return data

def add_clustering_coefficient(data):
    cc = clustering(data.edge_index, num_nodes=data.num_nodes).unsqueeze(1)

    if data.x is not None and data.x.numel() > 0:
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        data.x = torch.cat([data.x.float(), cc.float()], dim=1)
    else:
        data.x = cc.float()

    return data


