# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import SAGEConv, global_mean_pool

class OSUG_SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=10, out_channels=2, embedding_size=128):
        super(OSUG_SAGE, self).__init__()

        self.embedding = nn.Embedding(in_channels, embedding_size)

        self.conv_layers_1 = SAGEConv(embedding_size, hidden_channels)
        self.bn_layers_1 = BatchNorm1d(hidden_channels)
        self.conv_layers_2 = torch.nn.ModuleList()
        self.bn_layers_2 = torch.nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.conv_layers_2.append(SAGEConv(hidden_channels, hidden_channels))
            self.bn_layers_2.append(BatchNorm1d(hidden_channels))

        # self.classifier = nn.Sequential(
        #             Linear(hidden_channels, hidden_channels),
        #             BatchNorm1d(hidden_channels),
        #             nn.ReLU(),
        #             Linear(hidden_channels, out_channels)
        #         )
        self.classifier = Linear(hidden_channels, out_channels)

    #     self._reset_parameters()

    # def _reset_parameters(self):
        
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             # nn.init.xavier_uniform_(p)
    #             nn.init.kaiming_uniform_(p, a=np.sqrt(5))

    def forward(self, x, edge_index, u_index, example_mark):
        # x shape [N, in_channels]
        # edge_index shape [2, E]

        h = self.embedding(x) # in_channels -> embedding_size
        h = self.conv_layers_1(h, edge_index) # embedding_size -> hidden_channels
        h = self.bn_layers_1(h)
        h = F.relu(h)
        for conv, bn in zip(self.conv_layers_2, self.bn_layers_2):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # 1. 使用全局平均池化结果作为图表示向量
        out = global_mean_pool(h, example_mark)
        
        # 2. 使用全局节点作为图表示向量
        # out = []
        # for u in u_index:
        #     out.append(h[u])
        # out = torch.stack(out, dim=0)
        # out = h[u_index]

        # Graph Classifier
        out = F.dropout(self.classifier(out), p=0.1, training=self.training)    # [B, out_channels]
    
        return out
    