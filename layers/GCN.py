import math
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-15

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.weight_1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_1 = self.reset_parameters(self.weight_1)
        if dropout:
            self.gc_drop = nn.Dropout(dropout)
        else:
            self.gc_drop = lambda x: x
        self.bc = nn.BatchNorm1d(output_dim)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def forward(self, feat, adj, nlayer):
        z = self.gc_drop(torch.mm(feat, self.weight_1))
        z = torch.spmm(adj, z)
        for i in range(1,nlayer):
            z = torch.mm(adj, z)
        outputs = F.normalize(z, dim=1)
        return outputs
