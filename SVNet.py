# -*- coding: utf-8 -*-

import torch
import time

class CenteredLayer(torch.nn.Module):
    def __init__(self, device, is_training, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
        self.is_training = is_training
        self.device = device

    def forward(self, x):

        if self.is_training:
            x[x<0]=x[x<0] * 0.01
            x[x>1]=x[x>1] * 0.01 + 0.99
        else:
            x = torch.min(torch.max(torch.zeros(x.shape,device=self.device),x),torch.ones(x.shape,device=self.device)) # min(max(0,x),1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def soft_or(x1,x2):
    return 1-(-1/(-1+torch.log((1-x1)*(1-x2)+1e-100)))

def soft_and(x1,x2):
    return -1/(-1+torch.log(x1*x2+1e-100))


def soft_trace_check(device,chose_atom_,chose_right_,op_chose_weight,x,loop_start,is_training): # x : (batch_size,trace_len,vocab_len),loop_start:(batch_size,trace_len)
    '''
    输入：
        - chose_atom_:      in [0,1]^{batch_size * |AP| * formula_len}, 子公式i是否是原子命题j, 其中i是第2维度, j是第1维度;
        - chose_right_:     in [0,1]^{batch_size * formula_len * formula_len}, 子公式i的右子公式是否是子公式j, 其中i是第2维度, j是第1维度;
        - op_chose_weight:  in [0,1]^{batch_size * (1 + |OP|) * formula_len}, 子公式i的公式类型是否是j, 其中i是第2维度, j是第1维度, 公式类型按序如下: none,!,&,X,U,|,F,G;
        - x:                in [0,1]^{batch_size * trace_len * |AP|}, 每个状态的每个原子命题的真值情况;
        - loop_start:       in [0,1]^{batch_size * trace_len * 1}, 每个状态是否是环开始位置;
        - is_training:      是否是训练阶段 (true: 不使用torch.min和torch.max; false: 使用).
    输出：
        - all_x:    in [0,1]^{batch_size * trace_len * formula_len}, 每个子迹和每个子公式的满足情况.
    '''

    # print('forward start')
    net_k = chose_right_.shape[1]
    batch_size, trace_len, vocab_len = x.size()
    loop_start = loop_start.reshape((batch_size,trace_len))  # (batch_size, trace_len,1)
    all_x = torch.zeros((batch_size,trace_len, net_k), dtype=torch.float, device=device, requires_grad=False)# all_x :(batch_size,trace_len, net_k)

    # last_x: ( batch_size, trace_len,1)
    last_x = torch.zeros((batch_size,trace_len, 1), dtype=torch.float, device=device, requires_grad=False)
    # print('input:',x)
    # print('x',x.shape)
    # print('all_x',all_x.shape)
    # print('atom',chose_atom_.shape)
    # print('right',chose_right_.shape)
    for i in range(net_k):
        atom_x=x.bmm(chose_atom_)[:,:,net_k-i-1] # atom_x :(batch_size, trace_len)
        left_x=torch.cat((all_x[:,:,1:],last_x),dim=2)[:,:,net_k-i-1] # left_x :(batch_size,trace_len)

        right_x = all_x.bmm(chose_right_)[:,:,net_k-i-1] # right_x :(batch_size, trace_len)
        not_x = left_x * (-1) + 1 # not_x :(batch_size, trace_len)
        and_x=soft_and(left_x,right_x) # and_x :(batch_size, trace_len)
        or_x=soft_or(left_x,right_x) # or_x :(batch_size, trace_len)


        X_left_x = left_x[:,1:] # X_left_x :(batch_size, trace_len-1)
        end_left_x=torch.sum(loop_start*left_x,dim=1).reshape(batch_size,1) #end_left_x:(batch_size,1)
        X_left_x= torch.cat((X_left_x, end_left_x), dim=1) # X_left_x :(batch_size,trace_len)


        # print('right_x',right_x)
        # print('left_x',left_x)

        # print('all_x before until',all_x)
        until_x=right_x
        future_x=left_x
        global_x=soft_or(left_x,1)
        for j in range(trace_len):
            until_end_x = torch.sum(loop_start * until_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            future_end_x = torch.sum(loop_start * future_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            global_end_x = torch.sum(loop_start * global_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            # print('right_x',right_x)
            # print('left_x',left_x)
            until_x=soft_or(right_x,soft_and(left_x,torch.cat((until_x[:,1:],until_end_x),dim=1)))
            future_x=soft_or(left_x,torch.cat((future_x[:,1:],future_end_x),dim=1))
            global_x=soft_and(left_x,torch.cat((global_x[:,1:],global_end_x),dim=1))


        # print('not_x',not_x.shape)
        # print(not_x)
        # print('op_chose_weight',op_chose_weight[:,1,net_k-i-1].shape)
        # print(op_chose_weight[:,1,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape))
        # print(not_x*op_chose_weight[:, 1, net_k - i - 1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape))

        not_x=not_x*op_chose_weight[:,1,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        and_x = and_x * op_chose_weight[:,2,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        next_x = X_left_x * op_chose_weight[:,3,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape) # next_x :(batch_size,trace_len, net_k)
        until_x = until_x* op_chose_weight[:,4,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        or_x=or_x* op_chose_weight[:,5,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        future_x = future_x * op_chose_weight[:,6,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        global_x=global_x* op_chose_weight[:,7,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        # print('atom_x',atom_x)
        # print('not_x',not_x)
        # print('and_x',and_x)
        # print('next_x',next_x)
        # print('until_x',until_x)
        # print('or_x',or_x)
        # print('future_x',future_x)
        # print('soft_or(atom_x,not_x)',soft_or(atom_x,not_x))
        all_x[:, :, net_k - i - 1]=soft_or(soft_or(soft_or(atom_x,not_x),soft_or(next_x,and_x)),soft_or(soft_or(until_x,or_x),soft_or(future_x,global_x)))
        # all_x[:,:,net_k-i-1]=myrelu(atom_x+not_x+next_x+and_x+until_x+or_x+future_x+global_x)
        # print(i,all_x)

    # all_x = all_x - 0.5
    # all_x = torch.sigmoid(all_x * 5)
    # print('one time',time.time()-stime)
    # print('forward end')
    return all_x

# min max
def soft_trace_check_minmax(device,chose_atom_,chose_right_,op_chose_weight,x,loop_start,is_training): # x : (batch_size,trace_len,vocab_len),loop_start:(batch_size,trace_len)
    '''
    输入：
        - chose_atom_:      in [0,1]^{batch_size * |AP| * formula_len}, 子公式i是否是原子命题j, 其中i是第2维度, j是第1维度;
        - chose_right_:     in [0,1]^{batch_size * formula_len * formula_len}, 子公式i的右子公式是否是子公式j, 其中i是第2维度, j是第1维度;
        - op_chose_weight:  in [0,1]^{batch_size * (1 + |OP|) * formula_len}, 子公式i的公式类型是否是j, 其中i是第2维度, j是第1维度, 公式类型按序如下: none,!,&,X,U,|,F,G;
        - x:                in [0,1]^{batch_size * trace_len * |AP|}, 每个状态的每个原子命题的真值情况;
        - loop_start:       in [0,1]^{batch_size * trace_len * 1}, 每个状态是否是环开始位置;
        - is_training:      是否是训练阶段 (true: 不使用torch.min和torch.max; false: 使用).
    输出：
        - all_x:    in [0,1]^{batch_size * trace_len * formula_len}, 每个子迹和每个子公式的满足情况.
    '''

    # print('forward start')
    # stime=time.time()
    myrelu = CenteredLayer(device,is_training)
    net_k = chose_right_.shape[1]
    batch_size, trace_len, vocab_len = x.size()
    loop_start = loop_start.reshape((batch_size,trace_len))  # (batch_size, trace_len,1)
    all_x = torch.zeros((batch_size,trace_len, net_k), dtype=torch.float, device=device, requires_grad=False)# all_x :(batch_size,trace_len, net_k)

    # last_x: ( batch_size, trace_len,1)
    last_x = torch.zeros((batch_size,trace_len, 1), dtype=torch.float, device=device, requires_grad=False)
    # print('input:',all_x)
    # print('x',x.shape)
    # print('all_x',all_x.shape)
    chose_right_idxs=torch.argmax(chose_right_,1)
    for i in range(net_k):
        atom_x=x.bmm(chose_atom_)[:,:,net_k-i-1] # atom_x :(batch_size, trace_len)
        left_x=torch.cat((all_x[:,:,1:],last_x),dim=2)[:,:,net_k-i-1] # left_x :(batch_size,trace_len)

        # right_x = all_x.bmm(chose_right_)[:,:,net_k-i-1] # right_x :(batch_size, trace_len)
        right_x=all_x[torch.arange(all_x.size(0)),:, chose_right_idxs[:,net_k-i-1]] # right_x :(batch_size, trace_len)
        not_x = left_x * (-1) + 1 # not_x :(batch_size, trace_len)
        and_x=left_x+right_x-1 # and_x :(batch_size, trace_len)
        or_x=left_x+right_x # or_x :(batch_size, trace_len)


        X_left_x = left_x[:,1:] # X_left_x :(batch_size, trace_len-1)
        end_left_x=torch.sum(loop_start*left_x,dim=1).reshape(batch_size,1) #end_left_x:(batch_size,1)
        X_left_x= torch.cat((X_left_x, end_left_x), dim=1) # X_left_x :(batch_size,trace_len)


        # print('right_x',right_x)
        # print('left_x',left_x)

        # print('all_x before until',all_x)
        until_x=right_x
        future_x=left_x
        global_x=myrelu(left_x+1)
        for j in range(trace_len):
            until_end_x = torch.sum(loop_start * until_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            future_end_x = torch.sum(loop_start * future_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            global_end_x = torch.sum(loop_start * global_x, dim=1).reshape(batch_size, 1)  # end_x:(batch_size,1)
            # print('right_x',right_x)
            # print('left_x',left_x)
            until_x=myrelu(right_x+myrelu(left_x+torch.cat((until_x[:,1:],until_end_x),dim=1)-1))
            future_x=myrelu(left_x+torch.cat((future_x[:,1:],future_end_x),dim=1))
            global_x=myrelu(left_x+torch.cat((global_x[:,1:],global_end_x),dim=1)-1)

        # print('not_x',not_x.shape)
        # print(not_x)
        # print('op_chose_weight',op_chose_weight[:,1,net_k-i-1].shape)
        # print(op_chose_weight[:,1,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape))
        # print(not_x*op_chose_weight[:, 1, net_k - i - 1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape))

        not_x=not_x*op_chose_weight[:,1,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        and_x = and_x * op_chose_weight[:,2,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        next_x = X_left_x * op_chose_weight[:,3,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape) # next_x :(batch_size,trace_len, net_k)
        until_x = until_x* op_chose_weight[:,4,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        or_x=or_x* op_chose_weight[:,5,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        future_x = future_x * op_chose_weight[:,6,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        global_x=global_x* op_chose_weight[:,7,net_k-i-1].repeat_interleave(not_x.shape[1],0).reshape(not_x.shape)
        # print('atom_x',atom_x)
        # print('not_x',not_x)
        # print('and_x',and_x)
        # print('next_x',next_x)
        # print('until_x',until_x)
        # print('or_x',or_x)
        # print('future_x',future_x)
        all_x[:,:,net_k-i-1]=myrelu(atom_x+not_x+next_x+and_x+until_x+or_x+future_x+global_x)
        # print(i,all_x)

    # all_x = all_x - 0.5
    # all_x = torch.sigmoid(all_x * 100)
    # print('one time',time.time()-stime)
    # print('forward end')
    return all_x