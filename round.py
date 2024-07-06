# -*- coding: utf-8 -*-

import torch

def round_CF(state_sequence_batch, loop_batch, true_bound=0.5):
    '''
    使用characteristic function离散化迹
    '''
    hard_state_sequence_batch = torch.zeros_like(state_sequence_batch).float()
    hard_state_sequence_batch[state_sequence_batch > true_bound] = 1

    hard_loop_batch = loop_batch.eq(torch.max(loop_batch,dim=1).values.unsqueeze(dim=1).repeat(1,loop_batch.shape[1],1)).float()

    return hard_state_sequence_batch.to(torch.device("cpu")), hard_loop_batch.to(torch.device("cpu"))

def check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg):
    '''
    逻辑迹检测
    '''
    eval_res = [False] * len(hard_state_sequence_batch)
    hard_state_sequence_batch_cpu = hard_state_sequence_batch.numpy()
    for data_idx in range(hard_state_sequence_batch.shape[0]):  # 逐条数据处理
        loop_start = 0
        for lidx in range(hard_loop_batch.shape[1]):
            if hard_loop_batch[data_idx][lidx] == 1:
                loop_start = lidx
                break

        current_trace = hard_state_sequence_batch_cpu[data_idx]
        check_trace = [[] for _ in range(len(current_trace))]
        var_dict = eval_arg[0][data_idx]
        f_raw = eval_arg[1][data_idx]
        for s_idx in range(len(current_trace)):
            for key in var_dict:
                if current_trace[s_idx][var_dict[key]]:
                    check_trace[s_idx].append(key)

        # print(current_trace,check_trace,loop_start)
        eval_res[data_idx] = eval_func(f_raw, (check_trace, loop_start), var_dict.keys())
        return eval_res

def round_CF_check_logic(state_sequence_batch, loop_batch, eval_func, eval_arg, true_bound=0.5):
    '''
    使用characteristic function离散化迹，并进行逻辑迹检测
    '''
    hard_state_sequence_batch, hard_loop_batch = round_CF(state_sequence_batch, loop_batch, true_bound)

    eval_res = check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg)

    return eval_res, hard_state_sequence_batch, hard_loop_batch

def round_k_mutate(mutation_rate, init_hard_state_sequence_batch):
    '''
    使用k步突变的方法离散化迹
    '''
    A = torch.rand(size=mutation_rate.shape)
    C = A - mutation_rate
    D = torch.zeros_like(C)
    D[C < 0] = 1
    hard_state_sequence_batch = D * (1 - init_hard_state_sequence_batch) + (1 - D) * init_hard_state_sequence_batch

    return hard_state_sequence_batch

def round_k_mutate_check_logic(state_sequence_batch, loop_batch, eval_func, eval_arg, sample_num=10, temperature=0.5, true_bound=0.5):
    '''
    使用k步突变的方法离散化迹，并进行逻辑迹检测，突变由神经网络预测的信息熵决定
    '''
    init_hard_state_sequence_batch, init_hard_loop_batch = round_CF(state_sequence_batch, loop_batch, true_bound)
    hard_state_sequence_batch, hard_loop_batch = init_hard_state_sequence_batch, init_hard_loop_batch

    eval_res = check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg)

    mutation_rate = (- state_sequence_batch * torch.log2(state_sequence_batch + 1e-6) - (1 - state_sequence_batch) * torch.log2((1 - state_sequence_batch) + 1e-6)) * temperature
    mutation_rate = mutation_rate.to(torch.device("cpu"))

    for _ in range(sample_num):
        if all(eval_res):
            break

        hard_state_sequence_batch = round_k_mutate(mutation_rate, init_hard_state_sequence_batch)
        current_eval_res = check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg)
        eval_res = [a | b for a, b in zip(eval_res, current_eval_res)]

    return eval_res, hard_state_sequence_batch, hard_loop_batch

def round_k_sample(state_sequence_batch):
    '''
    使用k步突变的方法离散化迹
    '''
    A = torch.rand(size=state_sequence_batch.shape)
    C = A - state_sequence_batch
    hard_state_sequence_batch = torch.zeros_like(C)
    hard_state_sequence_batch[C < 0] = 1

    return hard_state_sequence_batch

def round_k_sample_check_logic(state_sequence_batch, loop_batch, eval_func, eval_arg, sample_num=10, true_bound=0.5):
    '''
    使用k步采样的方法离散化迹，并进行逻辑迹检测，采样的分布由神经网络预测值决定
    '''
    hard_state_sequence_batch, hard_loop_batch = round_CF(state_sequence_batch, loop_batch, true_bound)

    eval_res = check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg)

    state_sequence_batch_cpu = state_sequence_batch.to(torch.device("cpu"))

    for _ in range(sample_num):
        if all(eval_res):
            break

        hard_state_sequence_batch = round_k_sample(state_sequence_batch_cpu)
        current_eval_res = check_logic(hard_state_sequence_batch, hard_loop_batch, eval_func, eval_arg)
        eval_res = [a | b for a, b in zip(eval_res, current_eval_res)]

    return eval_res, hard_state_sequence_batch, hard_loop_batch
