import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:

        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)]
            + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
            + [GCNConv(hidden_dim, output_dim)]
        )
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for _ in range(num_layers - 1)
            ]
        )
        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):

        for i in range(len(self.bns)):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.softmax(x)

        return x
