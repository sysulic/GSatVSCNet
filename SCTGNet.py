# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,global_mean_pool

from LTLEmbed.model import OSUG_SAGE

class TGOSUGNN(OSUG_SAGE):

    def __init__(self, device, in_channels, hidden_channels=256, num_layers=10, out_channels=2, embedding_size=128, mlp_hidden_channels=128, max_trace_len=5):
        '''
        输入：
            - in_channels:          节点类型数目;
            - hidden_channels:      隐藏层嵌入维度;
            - num_layers:           层数;
            - out_channels:         输出维度;
            - embedding_size:       节点初始embedding的维度;
        '''
    
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, embedding_size)

        self.device = device
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        self.mlp_hidden_channels = mlp_hidden_channels
        self.max_trace_len = max_trace_len

        # self.proj = nn.Parameter(torch.randn(self.hidden_channels, self.embedding_size))

        # 解码迹
        self.decoder_conv = SAGEConv(self.hidden_channels, self.hidden_channels)
        self.decoder_bn = nn.BatchNorm1d(self.hidden_channels)

        # 预测迹中每个状态的原子命题真值
        self.predictor=nn.Sequential(
                            nn.Linear(self.hidden_channels, self.mlp_hidden_channels),
                            nn.BatchNorm1d(self.mlp_hidden_channels),
                            nn.ReLU(),
                            # nn.Dropout(p=0.1),
                            nn.Linear(self.mlp_hidden_channels, self.out_channels)
                        )
        
        # 预测迹中环开始位置
        self.loop_checker=nn.Sequential(
                            nn.Linear(self.hidden_channels * 2, self.mlp_hidden_channels),
                            nn.BatchNorm1d(self.mlp_hidden_channels),
                            nn.ReLU(),
                            # nn.Dropout(p=0.1),
                            nn.Linear(self.mlp_hidden_channels, 1)
                        )
    
    def embed_init(self, x):
        # x shape [N, in_channels]

        return self.embedding(x)

    def embed_ver_1(self, h, edge_index):
        # h shape [N, embedding_size]

        h = self.conv_layers_1(h, edge_index)
        h = self.bn_layers_1(h)
        h = F.relu(h)
        
        return h
    
    def embed_ver_2(self, h, edge_index):
        # h shape [N, hidden_channels]

        for conv, bn in zip(self.conv_layers_2, self.bn_layers_2):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # print(f'h:{h}')

        return h

    def encode(self, x, edge_index):

        h = self.embed_init(x)
        h = self.embed_ver_1(h, edge_index)
        h = self.embed_ver_2(h, edge_index)

        return h

    def decode(self, h, edge_index):

        h = self.decoder_conv(h, edge_index)
        h = self.decoder_bn(h)
        h = F.relu(h)

        return h

    def read_out(self, h, example_mark):
        return global_mean_pool(h, example_mark)

    def generate_trace(self, x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var):
        '''
        模型逻辑：
            1. h = GNN.encoder(x) //通过多层OSUG（不同参数）获取节点向量，h in R^{num_vertex * dimension_hidden}；
            2. For t from 1 to max_trace_len: 
                a. h = GNN.decode(h) //通过一层OSUG（为了支持可变迹长度共用参数）获取节点向量，h in R^{num_vertex * dimension_hidden}；
                b. s_t = GNN.readout(h) //计算第t个状态的状态向量，s_t in R^{batch_size * dimension_hidden}；
                c. h_{atom} = get_atom(h) //从所有的节点中得到原子命题对应的节点向量，h_{atom} in R^{batch_size * num_atom * dimension_hidden}；
                d. y_t = classfier(h_{atom}) //预测第t个状态的各个原子命题的真值情况，y_t in (0,1)^{batch_size * num_atom}；
            3. y = [y_1, ..., y_T] //y in (0,1)^{batch_size * trace_len * num_atom}；
            4. loop = softmax(MLP_2([s_i,s_T])) //利用状态向量计算环开始位置，loop in (0,1)^{T};
            5. return Y, loop
        '''
        h = self.encode(x_batch, edge_index_batch)

        s = torch.zeros(size=(batch_size,self.max_trace_len,self.hidden_channels),dtype=torch.float,device=self.device)
        state_sequence_batch = torch.zeros(size=(batch_size,self.max_trace_len,max_num_var),dtype=torch.float,device=self.device)
        for i in range(self.max_trace_len):
            h = self.decode(h, edge_index_batch)
            # print(f'h:{h}')
            s[:,i] = self.read_out(h, example_mark_batch)
            # print(f's[:,{i}]:{s[:,i]}')
            h_atom = torch.cat((h, torch.zeros(size=(1, h.shape[1]),dtype=torch.float,device=self.device)), dim=0)[G2T_index_batch]
            # print(f'h_atom:{h_atom}')
            state_sequence_batch[:,i] = torch.softmax(self.predictor(h_atom.view(-1,self.hidden_channels)), dim=1)[:,1].view(batch_size,max_num_var)
            # print(f'state_sequence_batch[:,{i}]:{state_sequence_batch[:,i]}')
            state_sequence_batch[:,i] = torch.mul(state_sequence_batch[:,i], atom_mask_batch) # 去除padding的影响
            # print(f'state_sequence_batch[:,{i}]:{state_sequence_batch[:,i]}')
            # x = torch.mm(h, self.proj)
            # print(f'x:{x}')
        
        # print(f's:{s}')
        # print(f'state_sequence_batch:{state_sequence_batch}')
        loop_batch = torch.zeros(size=(batch_size,self.max_trace_len,1),dtype=torch.float,device=self.device) # loop_batch in R^{batch_size * trace_len * 1}
        for i in range(self.max_trace_len):
            loop_batch[:,i]=self.loop_checker(torch.cat((s[:,i],s[:,-1]),dim=1))
            # print(f'cat((s[:,{i}],s[:,-1])):{torch.cat((s[:,i],s[:,-1]),dim=1)}')
            # print(f'loop_batch[:,{i}]:{loop_batch[:,i]}')
        loop_batch = torch.softmax(loop_batch,dim=1)
        
        return state_sequence_batch, loop_batch
    
class TGOSUGNN2RNN(OSUG_SAGE):

    def __init__(self, device, in_channels, hidden_channels=256, num_layers=10, out_channels=2, embedding_size=128, mlp_hidden_channels=128, max_trace_len=5):
        '''
        输入：
            - in_channels:          节点类型数目;
            - hidden_channels:      隐藏层嵌入维度;
            - num_layers:           层数;
            - out_channels:         输出维度;
            - embedding_size:       节点初始embedding的维度;
        '''
    
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, embedding_size)

        self.device = device
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        self.mlp_hidden_channels = mlp_hidden_channels
        self.max_trace_len = max_trace_len

        # 解码迹
        self.decoder_state_sequence = nn.GRU(input_size=self.hidden_channels,hidden_size=self.hidden_channels,num_layers=4,batch_first=True)
        self.decoder_loop = nn.GRU(input_size=self.hidden_channels,hidden_size=self.hidden_channels,num_layers=4,batch_first=True)

        # 预测迹中每个状态的原子命题真值
        self.predictor=nn.Sequential(
                            nn.Linear(self.hidden_channels, self.mlp_hidden_channels),
                            nn.BatchNorm1d(self.mlp_hidden_channels),
                            nn.ReLU(),
                            # nn.Dropout(p=0.1),
                            nn.Linear(self.mlp_hidden_channels, self.out_channels)
                        )
        
        # 预测迹中环开始位置
        self.loop_checker=nn.Sequential(
                            nn.Linear(self.hidden_channels * 2, self.mlp_hidden_channels),
                            nn.BatchNorm1d(self.mlp_hidden_channels),
                            nn.ReLU(),
                            # nn.Dropout(p=0.1),
                            nn.Linear(self.mlp_hidden_channels, 1)
                        )

    def embed_init(self, x):
        # x shape [N, in_channels]

        return self.embedding(x)

    def embed_ver_1(self, h, edge_index):
        # h shape [N, embedding_size]

        h = self.conv_layers_1(h, edge_index)
        h = self.bn_layers_1(h)
        h = F.relu(h)
        
        return h
    
    def embed_ver_2(self, h, edge_index):
        # h shape [N, hidden_channels]

        for conv, bn in zip(self.conv_layers_2, self.bn_layers_2):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        # print(f'h:{h}')

        return h

    def encode(self, x, edge_index):

        h = self.embed_init(x)
        h = self.embed_ver_1(h, edge_index)
        h = self.embed_ver_2(h, edge_index)

        return h

    def decode(self, h, s):

        h,s = self.decoder(h.view(-1,self.hidden_channels), s.view(-1,self.hidden_channels))
        h = self.decoder_bn(h)
        h = F.relu(h)

        return h

    def read_out(self, h, example_mark):
        return global_mean_pool(h, example_mark)

    def generate_trace(self, x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var):
        '''
        模型逻辑：
            1. h = GNN.encoder(x) //通过多层OSUG（不同参数）获取节点向量，h in R^{num_vertex * dimension_hidden}；
            2. h_{atom} = get_atom(h) //从所有的节点中得到原子命题对应的节点向量，h_{atom} in R^{batch_size * num_atom * dimension_hidden}；
            3. For t from 1 to max_trace_len: 
                a. s_t,history = LSTM.decode(h_{atom},history) //计算第t个状态的状态向量，s_t in R^{batch_size * num_atom * dimension_hidden}；
                b. y_t = classfier(s_t) //预测第t个状态的各个原子命题的真值情况，y_t in (0,1)^{batch_size * num_atom}；
            4. y = [y_1, ..., y_T] //y in (0,1)^{batch_size * trace_len * num_atom}；
            5. loop = softmax(MLP_2([mean(s_i,dim=1),mean(s_T,dim=1)])) //利用状态向量计算环开始位置，loop in (0,1)^{T};
            6. return Y, loop
        '''
        h = self.encode(x_batch, edge_index_batch)
        # print(f'h:{h}')
        h_atom = torch.cat((h, torch.zeros(size=(1, h.shape[1]),dtype=torch.float,device=self.device)), dim=0)[G2T_index_batch] # h_{atom} shape [batch_size * num_atom * dimension_hidden]
        # print(f'h_atom:{h_atom}')
        # print(f'input of decoder_state_sequence:{h_atom.view(-1,self.hidden_channels).unsqueeze(1).repeat(1, self.max_trace_len, 1)}')
        s, _ = self.decoder_state_sequence(h_atom.view(-1,self.hidden_channels).unsqueeze(1).repeat(1, self.max_trace_len, 1)) # s shape [batch_size * num_atom, max_trace_len, dimension_hidden]
        # print(f's:{s}')
        # s = s.view(batch_size,max_num_var,self.max_trace_len,self.hidden_channels).permute(0, 2, 1, 3) # s shape [batch_size, num_atom, max_trace_len, dimension_hidden] -> [batch_size, max_trace_len, num_atom, dimension_hidden]
        state_sequence_batch = torch.softmax(self.predictor(s.reshape(-1,self.hidden_channels)), dim=1)[:,1].view(batch_size,max_num_var,self.max_trace_len).permute(0, 2, 1) # state_sequence_batch shape [batch_size, max_trace_len, num_atom]
        state_sequence_batch = torch.mul(state_sequence_batch, atom_mask_batch.unsqueeze(1)) # 去除padding的影响
        # print(f'state_sequence_batch:{state_sequence_batch}')

        h_G = self.read_out(h, example_mark_batch) # h_{atom} shape [batch_size * dimension_hidden]
        # print(f'h_G:{h_G}')
        # print(f'input of decoder_loop:{h_G.unsqueeze(1).repeat(1, self.max_trace_len, 1)}')
        s, _ = self.decoder_loop(h_G.unsqueeze(1).repeat(1, self.max_trace_len, 1)) # s shape [batch_size, max_trace_len, dimension_hidden]，每一时刻所有点表示的平均用于表示状态
        loop_batch = torch.zeros(size=(batch_size,self.max_trace_len,1),dtype=torch.float,device=self.device) # loop_batch in R^{batch_size * trace_len * 1}
        for i in range(self.max_trace_len):
            loop_batch[:,i]=self.loop_checker(torch.cat((s[:,i],s[:,-1]),dim=1))
            # print(f'cat((s[:,{i}],s[:,-1])):{torch.cat((s[:,i],s[:,-1]),dim=1)}')
            # print(f'loop_batch[:,{i}]:{loop_batch[:,i]}')
        loop_batch = torch.softmax(loop_batch,dim=1)
        
        return state_sequence_batch, loop_batch

class SCTGOSUGNN(TGOSUGNN):

    def __init__(self, device, in_channels, hidden_channels=256, num_layers=10, out_channels=2, embedding_size=128, mlp_hidden_channels=128, max_trace_len=5):
        '''
        输入：
            - in_channels:          节点类型数目;
            - hidden_channels:      隐藏层嵌入维度;
            - num_layers:           层数;
            - out_channels:         输出维度;
            - embedding_size:       节点初始embedding的维度;
        '''
    
        super().__init__(device, in_channels, hidden_channels, num_layers, out_channels, embedding_size, mlp_hidden_channels, max_trace_len)
        
        # 预测可满足性
        self.sc_classifier=nn.Sequential(
                            nn.Linear(self.hidden_channels, self.mlp_hidden_channels),
                            nn.BatchNorm1d(self.mlp_hidden_channels),
                            nn.ReLU(),
                            # nn.Dropout(p=0.1),
                            nn.Linear(self.mlp_hidden_channels, self.out_channels)
                        )

    def generate_trace(self, x_batch, edge_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, batch_size, max_num_var):
        '''
        模型逻辑：
            1. h = GNN.encoder(x) //通过多层OSUG（不同参数）获取节点向量，h in R^{num_vertex * dimension_hidden}；
            2. f = GNN.readout(h) //得到公式嵌入向量
            2. For t from 1 to max_trace_len: 
                a. h = GNN.decode(h) //通过一层OSUG（为了支持可变迹长度共用参数）获取节点向量，h in R^{num_vertex * dimension_hidden}；
                b. s_t = GNN.readout(h) //计算第t个状态的状态向量，s_t in R^{batch_size * dimension_hidden}；
                c. h_{atom} = get_atom(h) //从所有的节点中得到原子命题对应的节点向量，h_{atom} in R^{batch_size * num_atom * dimension_hidden}；
                d. y_t = classfier(h_{atom}) //预测第t个状态的各个原子命题的真值情况，y_t in (0,1)^{batch_size * num_atom}；
            3. y = [y_1, ..., y_T] //y in (0,1)^{batch_size * trace_len * num_atom}；
            4. loop = softmax(MLP_2([s_i,s_T])) //利用状态向量计算环开始位置，loop in (0,1)^{T};
            5. return Y, loop, f
        '''
        h = self.encode(x_batch, edge_index_batch)
        f_batch = self.read_out(h, example_mark_batch)

        s = torch.zeros(size=(batch_size,self.max_trace_len,self.hidden_channels),dtype=torch.float,device=self.device)
        state_sequence_batch = torch.zeros(size=(batch_size,self.max_trace_len,max_num_var),dtype=torch.float,device=self.device)
        for i in range(self.max_trace_len):
            h = self.decode(h, edge_index_batch)
            # print(f'h:{h}')
            s[:,i] = self.read_out(h, example_mark_batch)
            # print(f's[:,{i}]:{s[:,i]}')
            h_atom = torch.cat((h, torch.zeros(size=(1, h.shape[1]),dtype=torch.float,device=self.device)), dim=0)[G2T_index_batch]
            # print(f'h_atom:{h_atom}')
            state_sequence_batch[:,i] = torch.softmax(self.predictor(h_atom.view(-1,self.hidden_channels)), dim=1)[:,1].view(batch_size,max_num_var)
            # print(f'state_sequence_batch[:,{i}]:{state_sequence_batch[:,i]}')
            state_sequence_batch[:,i] = torch.mul(state_sequence_batch[:,i], atom_mask_batch) # 去除padding的影响
            # print(f'state_sequence_batch[:,{i}]:{state_sequence_batch[:,i]}')
            # x = torch.mm(h, self.proj)
            # print(f'x:{x}')
        
        # print(f's:{s}')
        # print(f'state_sequence_batch:{state_sequence_batch}')
        loop_batch = torch.zeros(size=(batch_size,self.max_trace_len,1),dtype=torch.float,device=self.device) # loop_batch in R^{batch_size * trace_len * 1}
        for i in range(self.max_trace_len):
            loop_batch[:,i]=self.loop_checker(torch.cat((s[:,i],s[:,-1]),dim=1))
            # print(f'cat((s[:,{i}],s[:,-1])):{torch.cat((s[:,i],s[:,-1]),dim=1)}')
            # print(f'loop_batch[:,{i}]:{loop_batch[:,i]}')
        loop_batch = torch.softmax(loop_batch,dim=1)
        
        return state_sequence_batch, loop_batch, f_batch
    
    def check_satisfiability(self, f_batch):

        return self.sc_classifier(f_batch)
