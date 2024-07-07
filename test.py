# -*- coding: utf-8 -*-

import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import sys

from genData import GLDataSet, collate
from model import SatVSCNet_SC_TG
from logic_checker import check
from SVNet import soft_trace_check_minmax,soft_trace_check
from round import round_CF,round_CF_check_logic,round_k_mutate_check_logic,round_k_sample_check_logic


sys.setrecursionlimit(10 ** 5) # 工业级测例有的公式递归非常深

parser = ArgumentParser(description='Test SatVSCNet ((SC+TG)+SV)') # 用于测试train_v2训练的模型

parser.add_argument('--debug', type=int, default=1, help="debug")

parser.add_argument('--device', type=int, default=0, help="GPU number")
parser.add_argument('--sbm', type=str, default=None, help="saved best model")
parser.add_argument('--trp', type=str, default=None, help="test record path")
parser.add_argument('--ted', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/test_trace.json', help="test dataset")
parser.add_argument('--pted', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/test_trace_prep.json', help="preprocessing test dataset")
parser.add_argument('--vd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/dev_trace.json', help="validation dataset")
parser.add_argument('--pvd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/dev_trace_prep.json', help="preprocessing validation dataset")

parser.add_argument('--bs', type=int, default=1024, help="batch size")

parser.add_argument('--nl', type=int, default=10, help="number of layers") 
parser.add_argument('--hd', type=int, default=32, help="hidden dimension")
parser.add_argument('--ned', type=int, default=256, help="node embedding dimension")
parser.add_argument('--mtl', type=int, default=5, help="max trace length")
parser.add_argument('--sct', type=str, default='minmax', help="active function for soft checker")

parser.add_argument('--logic', type=int, default=1, help="whether to logical path checking")
parser.add_argument('--lt', type=str, default='cf', help="type of logical path checking (cf, km, ks)")
parser.add_argument('--round', type=int, default=1, help="whether to round the trace")

args = parser.parse_args()

if args.debug == 1:
    args = parser.parse_args([
        '--sbm','model/2024_04_13_17_00_58_sd_19921104_lr_0.001_wdr_0.0_bs_5_nl_2_hd_8_ned_4_mtl_3_mu_sc_1.0_mu_sv_1.0_mu_sp_0.0_sct_minmax/SatVSCNet_(SC+TG)+SV-step{5}-loss{0.9772}-acc{0.7500}-satacc{0.6667}.pth',
        '--trp','log/2024_04_13_17_00_58_sd_19921104_lr_0.001_wdr_0.0_bs_5_nl_2_hd_8_ned_4_mtl_3_mu_sc_1.0_mu_sv_1.0_mu_sp_0.0_sct_minmax/res_debug_detail',
        '--ted','data/test/debug.json',
        '--pted','data/test/debug_prep.json',
        '--vd','data/test/debug.json',
        '--pvd','data/test/debug_prep.json',
        '--bs','1',
        '--nl','2',
        '--hd','8',
        '--ned','4',
        '--mtl','3',
        '--sct','minmax',
        '--logic','1',
        '--lt','km',
        '--round','0'])

def setting():
    
    print(f"Testing: {args.sbm}")
    result_save = args.trp
    print(f"Result save to: {result_save}")
    print(f"GPU number: {args.device}")
    print(f"test dataset: {args.ted}")
    print(f"preprocessing test dataset: {args.pted}")
    print(f"validation dataset: {args.vd}")
    print(f"preprocessing validation dataset: {args.pvd}")
    print(f"batch size: {args.bs}")
    print(f"number of layers: {args.nl}")
    print(f"hidden dimension: {args.hd}")
    print(f"node embedding dimension: {args.ned}")
    print(f"max trace length: {args.mtl}")
    print(f"active function for soft checker: {args.sct}")
    print(f"whether to logical trace checking: {args.logic}")
    print(f"whether to round the trace: {args.round}")

    return result_save

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

def tensor_to_trace(trace_tensor, gamma_tensor, var_dict, true_bound):
    idx2var={var_dict[key]:key for key in var_dict.keys()}
    trace=[]
    for i in range(len(trace_tensor)):
        state=[]
        for j in range(len(trace_tensor[i])):
            if trace_tensor[i][j] > true_bound and j in idx2var.keys():
                state.append(idx2var[j])
        trace.append(state)
    loop_start=0
    for i in range(1,len(gamma_tensor)):
        if gamma_tensor[i]>gamma_tensor[loop_start]:
            loop_start=i
    
    return trace, loop_start

def get_num_predict_true(trace_batch, gamma_batch, true_bound):
    num_true = 0

    for i in range(trace_batch.shape[0]):
        for j in range(trace_batch.shape[1]):
            for k in range(trace_batch.shape[2]):
                num_true += 1 if trace_batch[i,j,k] > true_bound else 0
    for i in range(gamma_batch.shape[0]):
        for j in range(gamma_batch.shape[1]):
            for k in range(gamma_batch.shape[2]):
                num_true += 1 if gamma_batch[i,j,k] > true_bound else 0

    return num_true

def test(model, loader, true_bound=0.5, SAT_bound=0.5):
    '''
    测试，支持多实例测试。
    基于GNN的迹生成不需要选择超参true_bound和SAT_bound，因为其预测每个时刻的原子命题的赋值是投影到2维向量，然后取softmax，这默认是以0.5为分界。
    也不需要选择超参SAT_bound，因为round后的迹经过迹检测必定产生0或1的结果，且可满足性分类也是投影到2维向量，然后取softmax。
    '''
    
    result_save = setting()
    TP, FP, TN, FN = 1e-6, 1e-6, 1e-6, 1e-6
    check_correct = 0
    num_sat = 0
    res = []
    infer_total_time, sv_total_time = 0, 0
    num_predict_true = 0

    model.eval()
    with torch.no_grad():

        for x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, y_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch, inorder_list, var_dict_list, tree_tuple_list in tqdm(loader, desc='Testing', ncols=100):

            x_batch = x_batch.to(device)
            edge_index_batch = edge_index_batch.to(device)
            u_index_batch = u_index_batch.to(device)
            G2T_index_batch = G2T_index_batch.to(device)
            example_mark_batch = example_mark_batch.to(device)
            y_batch = y_batch.to(device)

            op_chose_batch = op_chose_batch.to(device)
            atom_chose_batch = atom_chose_batch.to(device)
            right_chose_batch = right_chose_batch.to(device)
            atom_mask_batch = atom_mask_batch.to(device)

            infer_start = time.time()

            logits_sc_batch, state_sequence_batch, loop_batch = model.infer(x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch)
            p_sc_batch = torch.argmax(logits_sc_batch, dim=1)

            infer_end = time.time()

            num_predict_true += get_num_predict_true(state_sequence_batch, loop_batch, true_bound)

            sv_start = time.time()

            if args.logic == 1:
                if args.lt == 'cf':
                    p_sv_batch, state_sequence_batch, loop_batch = round_CF_check_logic(state_sequence_batch, loop_batch, check, (var_dict_list, tree_tuple_list), true_bound)
                elif args.lt == 'km':
                    p_sv_batch, state_sequence_batch, loop_batch = round_k_mutate_check_logic(state_sequence_batch, loop_batch, check, (var_dict_list, tree_tuple_list))
                else:
                    p_sv_batch, state_sequence_batch, loop_batch = round_k_sample_check_logic(state_sequence_batch, loop_batch, check, (var_dict_list, tree_tuple_list))
                
                p_sv_batch = torch.tensor(p_sv_batch, dtype=torch.float, device=device)
            
            else:
                if args.round == 1:
                    state_sequence_batch, loop_batch = round_CF(state_sequence_batch, loop_batch, true_bound)
                    state_sequence_batch, loop_batch = state_sequence_batch.to(device), loop_batch.to(device)

                if args.sct == 'minmax':
                    p_sv_batch = soft_trace_check_minmax(device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=False)[:,0,0]
                else:
                    p_sv_batch = soft_trace_check(device, atom_chose_batch, right_chose_batch, op_chose_batch, state_sequence_batch, loop_batch, is_training=False)[:,0,0]

            sv_end = time.time()

            # 评估
            for i in range(y_batch.shape[0]):
                # v1: if p_{sc} <= 0.5: return UNSAT; else: return SAT
                # predict = bool(p_sc_batch[i] <= SAT_bound)
                # v2 (double check): if p_{sc} <= 0.5 and p_{sv} <= 0.5: return UNSAT; else: return SAT
                predict = not bool((p_sc_batch[i] <= SAT_bound) and (p_sv_batch[i] <= SAT_bound))
                
                expect = bool(y_batch[i])
                if expect and expect == predict:
                    TP += 1
                elif expect and expect != predict:
                    FN += 1
                elif not expect and expect == predict:
                    TN += 1
                else:
                    FP += 1

                if expect:
                    num_sat += 1
                    check_correct += 1 if p_sv_batch[i] > SAT_bound else 0

                infer_total_time += infer_end - infer_start
                sv_total_time += sv_end - sv_start

                state_sequence, loop_begin = tensor_to_trace(state_sequence_batch[i].to(torch.device("cpu")), loop_batch[i].to(torch.device("cpu")), var_dict_list[i], true_bound)
                res.append((inorder_list[i], predict, p_sc_batch[i].cpu().item(), p_sv_batch[i], state_sequence, loop_begin, infer_end - infer_start, sv_end - sv_start))

        Acc, Pre, Rec, F1 = (TP + TN) / (TP + TN + FP + FN), \
                            TP / (TP + FP), \
                            TP / (TP + FN), \
                            (2 * TP) / (2 * TP + FP + FN)

        print('Total test time (FE + TG): %.4f' % (infer_total_time))
        print('Total test time (SV): %.4f' % (sv_total_time))
        print('Tatal predict true times: %d' % (num_predict_true))
        print('(TP, TN, FP, FN) = (%d, %d, %d, %d)' % (TP, TN, FP, FN))
        print('(Acc, P, R, F1) = (%.4f, %.4f, %.4f, %.4f)' % (Acc, Pre, Rec, F1))
        print('Acc of SV (%d / %d) = %.4f' % (check_correct, num_sat, check_correct / num_sat))

        with open(result_save, "w") as f:
            for l in tqdm(res, desc='Writing', ncols=100):
                print(l[0], "sat" if l[1] else "unsat", l[2], l[3], l[4], l[5], l[6], l[7], sep='\t', file=f)

if __name__ == '__main__':

    test_dataset = GLDataSet(args.ted, prep_path=args.pted)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=collate, shuffle=False)
    # if args.debug == 1:
    #     test_loader = DataLoader(test_dataset, batch_size=args.bs, collate_fn=collate, shuffle=False)
    # else:
    #     test_loader = DataLoader(test_dataset, num_workers=8, pin_memory=torch.cuda.is_available(), batch_size=args.bs, collate_fn=collate, shuffle=False)

    model = SatVSCNet_SC_TG(
                device, 
                embed_dim=args.hd, 
                num_layers=args.nl,
                node_embed_dim=args.ned,
                mlp_dim=128, 
                max_trace_len=args.mtl,
                soft_check_type=args.sct).to(device)
    print(f"Load model at {args.sbm}.")
    model.load_state_dict(torch.load(args.sbm, map_location=device))

    test(model, test_loader)

