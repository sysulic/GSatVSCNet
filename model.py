# -*- coding: utf-8 -*-

import torch.nn as nn
import numpy as np

from SCTGNet import TGOSUGNN,SCTGOSUGNN,TGOSUGNN2RNN
from SVNet import soft_trace_check,soft_trace_check_minmax

class SatVSCNet_TG(nn.Module):

    def __init__(self, device, embed_dim=256, num_layers=10, node_embed_dim=128, mlp_dim=128, num_classes=2, max_trace_len=5, soft_check_type='minmax'):
        '''
        输入：
            - embed_dim:      隐藏层嵌入维度, 公式embedding的维度;
            - num_layers:     图网络层数;
            - node_embed_dim: 图网络节点编码维度;
            - mlp_dim:        MLP中间层维度;
            - num_classes:    分类数目;
            - max_trace_len:  最大迹长度;
            - soft_check_type:激活函数选择minmax/ other
        '''
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.node_embed_dim = node_embed_dim
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.max_trace_len = max_trace_len
        self.soft_check_type = soft_check_type
        
        self.TGNet = TGOSUGNN(
                        device=self.device,
                        in_channels=12, 
                        hidden_channels=self.embed_dim, 
                        num_layers=self.num_layers, 
                        out_channels=2, 
                        embedding_size=self.node_embed_dim,
                        mlp_hidden_channels = self.mlp_dim,
                        max_trace_len=self.max_trace_len).to(self.device)

        self._reset_parameters()

    def _reset_parameters(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                nn.init.kaiming_uniform_(p, a=np.sqrt(5))

    def soft_trace_check(self, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch):
        if self.soft_check_type == 'minmax' or not self.training:
            p_sv_batch = soft_trace_check_minmax(self.device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=self.training)[:,0,0]
        else:
            p_sv_batch = soft_trace_check(self.device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=self.training)[:,0,0]
        
        return p_sv_batch
    
    def forward(self, x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch):
        '''
        模型逻辑：
            - p_{sv} = SVNet(TGNet(x));
        '''
        batch_size, max_num_var = atom_chose_batch.shape[0], atom_chose_batch.shape[1]

        state_sequence_batch, loop_batch = self.TGNet.generate_trace(x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var)

        p_sv_batch = self.soft_trace_check(atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch)

        return p_sv_batch, state_sequence_batch, loop_batch

class SatVSCNet_SC_TG(nn.Module):

    def __init__(self, device, embed_dim=256, num_layers=10, node_embed_dim=128, mlp_dim=128, num_classes=2, max_trace_len=5, soft_check_type='minmax'):
        '''
        输入：
            - embed_dim:      隐藏层嵌入维度, 公式embedding的维度;
            - num_layers:     图网络层数;
            - node_embed_dim: 图网络节点编码维度;
            - mlp_dim:        分类层嵌入维度;
            - num_classes:    分类数目;
            - max_trace_len:  最大迹长度;
            - soft_check_type:激活函数选择minmax/ other
        '''
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.node_embed_dim = node_embed_dim
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.max_trace_len = max_trace_len
        self.soft_check_type = soft_check_type

        self.SCTGNet = SCTGOSUGNN(
                        device=self.device,
                        in_channels=12, 
                        hidden_channels=self.embed_dim, 
                        num_layers=self.num_layers, 
                        out_channels=2, 
                        embedding_size=self.node_embed_dim,
                        mlp_hidden_channels = self.mlp_dim,
                        max_trace_len=self.max_trace_len).to(self.device)

        self._reset_parameters()

    def _reset_parameters(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                nn.init.kaiming_uniform_(p, a=np.sqrt(5))

    def soft_trace_check(self, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch):
        if self.soft_check_type == 'minmax' or not self.training:
            p_sv_batch = soft_trace_check_minmax(self.device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=self.training)[:,0,0]
        else:
            p_sv_batch = soft_trace_check(self.device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=self.training)[:,0,0]
        
        return p_sv_batch
    
    def forward(self, x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch):
        '''
        模型逻辑：
            - p_{sv} = SVNet(SCTGNet(x));
            - f = SCTGNet.embed;
            - p_{sc} = MLP(f);
        '''

        batch_size, max_num_var = atom_chose_batch.shape[0], atom_chose_batch.shape[1]

        state_sequence_batch, loop_batch, f_batch = self.SCTGNet.generate_trace(x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var)

        p_sv_batch = self.soft_trace_check(atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch)

        logits_sc_batch = self.SCTGNet.check_satisfiability(f_batch)

        return logits_sc_batch, p_sv_batch, state_sequence_batch, loop_batch

    def infer(self, x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch):
        '''
        用于测试时高效推理。
        '''

        batch_size, max_num_var = atom_chose_batch.shape[0], atom_chose_batch.shape[1]

        state_sequence_batch, loop_batch, f_batch = self.SCTGNet.generate_trace(x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var)

        logits_sc_batch = self.SCTGNet.check_satisfiability(f_batch)

        return logits_sc_batch, state_sequence_batch, loop_batch