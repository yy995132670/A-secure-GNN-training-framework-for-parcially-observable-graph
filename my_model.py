import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn.pool import global_max_pool,global_mean_pool
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper
from torch import autograd

EPS = 1e-10


class MyLayer(torch.nn.Module):
    def __init__(self, mask, add_self_loops=True):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.mask = mask
    #卷积
    def forward(self, x, edge_index):
        # torch.autograd.set_detect_anomaly(True)
        row, col = edge_index
        A, B = x[row], x[col]
        att_score = F.cosine_similarity(A, B)
        # att_score = -F.pairwise_distance(A, B, p=2)

        edge_index = edge_index[:, self.mask]
        att_score = att_score[self.mask]

        #归一化
        row, col = edge_index
        row_sum = scatter(att_score, col, dim_size=x.size(0))
        att_score_norm = att_score / (row_sum[row] + EPS)

        #避免梯度爆炸
        if self.add_self_loops:
            degree = scatter(torch.ones_like(att_score_norm), col, dim_size=x.size(0))
            self_weight = 1.0 / (degree + 1)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            loop_index = torch.arange(
                0, x.size(0), dtype=torch.long, device=edge_index.device
            )
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
        att_score_norm = att_score_norm.exp()

        return edge_index, att_score_norm

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class MyModel(nn.Module):
    @wrapper
    def __init__(self, in_channels, out_channels, normalize, bias, mask):
        super().__init__()

        conv = []
        conv.append(MyLayer(mask=mask, add_self_loops=True))
        conv.append(
            GCNConv(
                in_channels,
                out_channels,
                add_self_loops=False,
                bias=bias,
                normalize=normalize,
            )
        )
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        for layer in self.conv:
            if isinstance(layer, MyLayer):
                edge_index, edge_weight = layer(x, edge_index)
            elif isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x


# class MyModel(nn.Module):
#     @wrapper
#     def __init__(self, in_channels, out_channels, normalize, bias, mask):
#         super().__init__()
#
#         self.conv1 = MyLayer(mask=mask, add_self_loops=True)
#         self.conv2 = GCNConv(
#             in_channels,
#             128,  # 中间层大小
#             add_self_loops=False,
#             bias=bias,
#             normalize=normalize,
#         )
#         self.fc = nn.Linear(128, out_channels)
#
#
#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         self.fc.reset_parameters()
#
#     def forward(self, x, edge_index, edge_weight=None):
#         edge_index, edge_weight = self.conv1(x, edge_index)
#         x = F.relu(self.conv2(x, edge_index, edge_weight))
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)  # 使用适合多类分类的激活函数
