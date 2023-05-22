import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, EdgeConv


class EdgeConvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(EdgeConvNet, self).__init__()

        self.convnetwork = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, X):
        return self.convnetwork(X)


class DynamicEdgeConvBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, aggr="mean", k=8):
        super(DynamicEdgeConvBlock, self).__init__()
        self.dynamicgraphconv = DynamicEdgeConv(nn=EdgeConvNet(input_dim, hidden_dim), aggr=aggr, k=k)

    def forward(self, X, batch=None):
        return self.dynamicgraphconv(X, batch=batch)


class EdgeConvBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, aggr="mean", skip=False):
        super(EdgeConvBlock, self).__init__()
        self.skip = skip
        self.graphconv = EdgeConv(nn=EdgeConvNet(input_dim, hidden_dim), aggr=aggr)

    def forward(self, X, edge_index):
        H = self.graphconv(X, edge_index)
        if self.skip: return torch.hstack((H, X))
        return H

