import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from greatx.nn.layers import GCNConv



class MyGCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MyGCN,self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)


    def reset_parameters(self):
        self.gcn.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
