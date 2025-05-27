import torch
from torch_geometric.utils import degree

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

import torch
import networkx as nx
from torch_geometric.utils import to_networkx, degree

def add_clustering_coefficient(data):
    # Converti il grafo in NetworkX
    G = to_networkx(data, to_undirected=True)
    clustering_dict = nx.clustering(G)

    # Ordina i valori secondo l'ordine dei nodi
    clustering_values = [clustering_dict[i] for i in range(data.num_nodes)]
    clustering_tensor = torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)

    # Concatena le feature
    if data.x is not None and data.x.numel() > 0:
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        data.x = torch.cat([data.x.float(), clustering_tensor], dim=1)
    else:
        data.x = clustering_tensor

    return data



