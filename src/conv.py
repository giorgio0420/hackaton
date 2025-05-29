import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

### GINE convolution (al posto di GIN e GCN)
def build_gine_layer(emb_dim):
    mlp = torch.nn.Sequential(
        torch.nn.Linear(emb_dim, 2*emb_dim),
        torch.nn.BatchNorm1d(2*emb_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(2*emb_dim, emb_dim)
    )
    return GINEConv(nn=mlp)
