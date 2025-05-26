import torch
from torch_geometric.utils import degree

def add_node_degree_feature(data):
    # Calcola il grado per ogni nodo
    row, col = data.edge_index
    deg = degree(row, num_nodes=data.num_nodes).unsqueeze(1)  # (num_nodes, 1)

    # Normalizza il grado: (grado - media) / std
    mean = deg.mean()
    std = deg.std()
    if std > 0:
        deg = (deg - mean) / std
    else:
        deg = deg - mean  # Evita divisione per 0

    # Se data.x esiste già, concateniamo il grado. Altrimenti, usiamo solo il grado.
    if data.x is not None and data.x.numel() > 0:
        data.x = torch.cat([data.x.float(), deg.float()], dim=1)
    else:
        data.x = deg.float()

    return data
