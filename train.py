# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import time
from tqdm import tqdm
import random
import numpy as np
import os
from os.path import join
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from genData import GLDataSet, collate
from model import SatVSCNet_SC_TG

parser = ArgumentParser(description='Train SatVSCNet ((SC+TG)+SV)')

parser.add_argument('--debug', type=int, default=1, help="debug")

parser.add_argument('--device', type=int, default=0, help="GPU number")
parser.add_argument('--sd', type=int, default=random.randint(10000000, 99999999), help="seed")
parser.add_argument('--tm', type=str, default=None, help="trained model")
parser.add_argument('--bl', type=float, default=None, help="bast loss")
parser.add_argument('--trd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/train_trace.json', help="training dataset")
parser.add_argument('--ptrd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/train_trace_prep.json', help="preprocessing training dataset")
parser.add_argument('--vd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/dev_trace.json', help="validation dataset")
parser.add_argument('--pvd', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/dev_trace_prep.json', help="preprocessing validation dataset")

parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--wdr', type=float, default=0, help="weight decay rate")
parser.add_argument('--bs', type=int, default=1024, help="batch size")
parser.add_argument('--e', type=int, default=1024, help="epochs")

parser.add_argument('--nl', type=int, default=10, help="number of layers") 
parser.add_argument('--hd', type=int, default=32, help="hidden dimension")
parser.add_argument('--ned', type=int, default=256, help="node embedding dimension")
parser.add_argument('--mtl', type=int, default=5, help="max trace length")
parser.add_argument('--mu_sc', type=float, default=1, help="auxiliary loss coefficient of SC")
parser.add_argument('--mu_sv', type=float, default=1, help="auxiliary loss coefficient of SV")
parser.add_argument('--mu_sp', type=float, default=0, help="auxiliary loss coefficient of sharpen")
parser.add_argument('--sct', type=str, default='minmax', help="active function for soft checker")

args = parser.parse_args()

if args.debug == 1:
    args = parser.parse_args([
        '--sd','19921104',
        '--trd','data/test/debug.json',
        '--ptrd','data/test/debug_prep.json',
        '--vd','data/test/debug.json',
        '--pvd','data/test/debug_prep.json',
        '--lr','1e-3',
        '--wdr','0',
        '--bs','5',
        '--e','16',
        '--nl','2',
        '--hd','8',
        '--ned','4',
        '--mtl','3',
        '--mu_sc','1',
        '--mu_sv','1',
        '--mu_sp','0',
        '--sct','minmax'])

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

# para
early = 1024          # 早停参数，early次评估未更新best model，结束训练
clip_grads = True   # 是否进行梯度截断

def setting():
    print('###### model setting ######')
    print(f'GPU number={args.device}')
    print(f'seed={args.sd}')
    print(f'trained model={args.tm}')
    print(f'training dataset={args.trd}')
    print(f'preprocessing training dataset={args.ptrd}')
    print(f'validation dataset={args.vd}')
    print(f'preprocessing validation dataset={args.pvd}')
    print(f'learning rate={args.lr}')
    print(f'weight decay rate={args.wdr}')
    print(f'batch size={args.bs}')
    print(f'epochs={args.e}')
    print(f'number of layers={args.nl}')
    print(f'hidden dimension={args.hd}')
    print(f'node embedding dimension={args.ned}')
    print(f'max trace length={args.mtl}')
    print(f'auxiliary loss coefficient of SC={args.mu_sc}')
    print(f'auxiliary loss coefficient of SV={args.mu_sv}')
    print(f'auxiliary loss coefficient of sharpen={args.mu_sp}')
    print(f'active function for soft checker={args.sct}')
    print('###########################')

    # 设置模型的存储
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_dir_name = f"{time_str}_sd_{args.sd}_lr_{args.lr}_wdr_{args.wdr}_bs_{args.bs}_nl_{args.nl}_hd_{args.hd}_ned_{args.ned}_mtl_{args.mtl}_mu_sc_{args.mu_sc}_mu_sv_{args.mu_sv}_mu_sp_{args.mu_sp}_sct_{args.sct}"
    model_dir = join('model', model_dir_name)
    os.makedirs(model_dir)
    print(f"Save model at {model_dir}.")
    # if not args.ts and not exists(model_dir):
    #     os.makedirs(model_dir)
    # if not args.ts:
    #     print(f"Save model at {model_dir}.")

    # 设置训练log的存储
    log_dir = join('log', model_dir_name)
    os.makedirs(log_dir)
    summaryWriter = SummaryWriter(log_dir)
    return model_dir, summaryWriter

# 设置随机种子
def seed_everywhere(seed):
    torch.manual_seed(seed)     # cpu
    torch.cuda.manual_seed(seed)    # gpu
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

# 绘制模型训练情况
def plotCurve(valLosses,model_dir):
    plt.figure()
    plt.xlabel('Training step')
    plt.ylabel('Validation Loss')
    plt.title("Learning Curve")
    plt.grid()
    plt.plot(range(1, len(valLosses) + 1), valLosses, 'o-', color="r")
    plt.savefig(join(model_dir, 'train_curve.jpg'))
    plt.show()

# 根据秒数获得format的时间
def get_time_str(ts) -> str:
    day = ts // 86400
    hour = ts % 86400 // 3600
    minute = ts % 3600 // 60
    sec = ts % 60
    if day > 0:
        return f"{day:.0f} d {hour:.0f} h {minute:.0f} m {sec:.0f} s"
    elif hour > 0:
        return f"{hour:.0f} h {minute:.0f} m {sec:.0f} s"
    elif minute > 0:
        return f"{minute:.0f} m {sec:.0f} s"
    else:
        return f"{sec:.0f} s"

def compute_entropic_regularization(state_seq, loop):
    entropy = - torch.sum(torch.sum(state_seq * torch.log(state_seq + 1e-5), dim=1), dim=1)
    entropy += - torch.sum(torch.sum(loop * torch.log(loop + 1e-5), dim=1), dim=1)

    return torch.mean(entropy)

def train():

    global bestPath
    seed_everywhere(args.sd)

    model_dir,summaryWriter = setting()

    train_dataset = GLDataSet(args.trd, prep_path=args.ptrd)
    if args.debug == 1:
        train_loader = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, num_workers=8, pin_memory=torch.cuda.is_available(), batch_size=args.bs, collate_fn=collate, shuffle=True)

    dev_dataset = GLDataSet(args.vd, prep_path=args.pvd)
    if args.debug == 1:
        dev_loader = DataLoader(dev_dataset, batch_size=args.bs, collate_fn=collate, shuffle=False)
    else:
        dev_loader = DataLoader(dev_dataset, num_workers=8, pin_memory=torch.cuda.is_available(), batch_size=args.bs, collate_fn=collate, shuffle=False)
        
    # supervised SC and weakly-supervised TG:   SatVSCNet_SC_TG
    model = SatVSCNet_SC_TG(
                device, 
                embed_dim=args.hd, 
                num_layers=args.nl,
                node_embed_dim=args.ned,
                mlp_dim=128, 
                max_trace_len=args.mtl,
                soft_check_type=args.sct).to(device)
    if args.tm is not None:
        print(f"Load model at {args.tm}.")
        model.load_state_dict(torch.load(args.tm, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdr)
    criterion_sc = nn.CrossEntropyLoss()
    criterion_sv = nn.MSELoss()
    # criterion_sv = nn.BCELoss()

    best_model, best_loss, best_acc = None, args.bl, None
    val_loss = []
    epsilon = 0 # 评估次数
    total_step = 0 # 记录总的迭代次数
    epoch_time = []     # 记录训练时间

    for epoch in range(args.e):
        start = time.time()
        train_time, val_time = 0, 0
        print('Epoch %d / %d.' % (epoch + 1, args.e))

        # 训练
        model.train()
        for x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, y_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch, _, _, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.e}', ncols=100): # x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, y_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch, inorder_list
            train_time_start = time.time()

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

            optimizer.zero_grad()

            logits_sc_batch, p_sv_batch, _, _ = model(x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch)
            
            # compute sc loss
            loss_sc = criterion_sc(logits_sc_batch.view(-1,2), y_batch)

            # compute sv loss
            loss_sv = criterion_sv(p_sv_batch, y_batch.float())

            # # compute faraway 0.5 for probability
            # loss_sp = compute_entropic_regularization(state_sequence_batch, loop_batch)

            # loss = args.mu_sc * loss_sc + args.mu_sv * loss_sv + args.mu_sp * loss_sp 
            loss = args.mu_sc * loss_sc + args.mu_sv * loss_sv
            loss.backward()

            if clip_grads:
                nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            total_step += 1

            summaryWriter.add_scalar("training step loss of SC", loss_sc.cpu().item(), total_step)
            summaryWriter.add_scalar("training step loss of SV", loss_sv.cpu().item(), total_step)
            # summaryWriter.add_scalar("training step loss of sharpen", loss_sp.cpu().item(), total_step)
            summaryWriter.add_scalar("training step loss", loss.cpu().item(), total_step)

            train_time += time.time() - train_time_start

        # 输出loss
        # print('Training Done. Train Loss: %.4f (%.2f * %.4f + %.2f * %.4f + %.2f * %.4f + %.2f * %.4f).' % (loss.cpu().item(), args.mu_sc, loss_sc.cpu().item(), args.mu_sv, loss_sv.cpu().item(), args.mu_sp, loss_sp.cpu().item()))
        print('Training Done. Train Loss = %.4f (%.2f * %.4f + %.2f * %.4f).' % (loss.cpu().item(), args.mu_sc, loss_sc.cpu().item(), args.mu_sv, loss_sv.cpu().item()))
        summaryWriter.add_scalar("training loss of SC", loss_sc.cpu().item(), total_step)
        summaryWriter.add_scalar("training loss of SV", loss_sv.cpu().item(), total_step)
        # summaryWriter.add_scalar("training loss of sharpen", loss_sp.cpu().item(), total_step)
        summaryWriter.add_scalar("training loss", loss.cpu().item(), total_step)

        # 验证
        model.eval()
        with torch.no_grad():
            # 计算验证集的loss和acc
            dev_loss_sc, dev_loss_sv, dev_loss_sp = 0, 0, 0
            acc_cnt, acc_cnt_sc, acc_cnt_sv = 0, 0, 0
            acc_sat_cnt_sc, acc_sat_cnt_sv, sat_cnt, acc_unsat_cnt_sc, acc_unsat_cnt_sv, unsat_cnt = 0, 0, 0, 0, 0, 0
            for x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, y_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch, _, _, _ in tqdm(dev_loader, desc=f'Epoch {epoch + 1}/{args.e}', ncols=100):
                val_time_start = time.time()
                
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

                logits_sc_batch, p_sv_batch, _, _ = model(x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch)
                
                dev_loss_sc += criterion_sc(logits_sc_batch.view(-1,2), y_batch).cpu().item() * y_batch.shape[0]
                dev_loss_sv += criterion_sv(p_sv_batch, y_batch.float()).cpu().item() * y_batch.shape[0]
                # dev_loss_sp += compute_entropic_regularization(state_sequence_batch, loop_batch).cpu().item()

                p_sc_batch = torch.argmax(logits_sc_batch,dim=1)
                p_sv_batch[p_sv_batch <= 0.5] = 0
                p_sv_batch[p_sv_batch > 0.5] = 1
                p = 1 - ((1 - p_sc_batch) * (1 - p_sv_batch)) # p_sc_batch = 0 and p_sv_batch = 0 -> p = 0
                r = 1 - torch.abs(p - y_batch) # p = y -> r = 1
                acc_cnt += int(torch.einsum('b->', r).cpu().item())
                r = 1 - torch.abs(p_sc_batch - y_batch)
                acc_cnt_sc += int(torch.einsum('b->', r).cpu().item())
                r = 1 - torch.abs(p_sv_batch - y_batch)
                acc_cnt_sv += int(torch.einsum('b->', r).cpu().item())
                acc_sat_cnt_sc += int(torch.dot(p_sc_batch.float(), y_batch.float()).cpu().item()) # y = 1 and p = 1 -> +1
                acc_sat_cnt_sv += int(torch.dot(p_sv_batch, y_batch.float()).cpu().item()) 
                sat_cnt += int(torch.einsum('b->', y_batch).cpu().item())
                acc_unsat_cnt_sc += int(torch.dot(1 - p_sc_batch.float(), 1 - y_batch.float()).cpu().item()) # y = 0 and p = 0 -> +1
                acc_unsat_cnt_sv += int(torch.dot(1 - p_sv_batch, 1 - y_batch.float()).cpu().item())
                unsat_cnt += int(torch.einsum('b->', 1 - y_batch).cpu().item())

                val_time += time.time() - val_time_start

            dev_loss_sc = dev_loss_sc / len(dev_dataset)
            dev_loss_sv = dev_loss_sv / len(dev_dataset)
            # dev_loss = args.mu_sc * dev_loss_sc + args.mu_sv * dev_loss_sv + args.mu_sp * dev_loss_sp
            dev_loss = args.mu_sc * dev_loss_sc + args.mu_sv * dev_loss_sv
            dev_acc, dev_acc_sc, dev_acc_sv = acc_cnt / len(dev_dataset), acc_cnt_sc / len(dev_dataset), acc_cnt_sv / len(dev_dataset)
            dev_sat_acc_sc, dev_unsat_acc_sc = acc_sat_cnt_sc / sat_cnt, acc_unsat_cnt_sc / unsat_cnt
            dev_sat_acc_sv, dev_unsat_acc_sv = acc_sat_cnt_sv / sat_cnt, acc_unsat_cnt_sv / unsat_cnt

            print(f'Validation Done. Dev Loss = {dev_loss:.4f}, Dev Loss of SC = {dev_loss_sc:.4f}, Dev Loss of SV = {dev_loss_sv:.4f}, '
                    f'Best Loss = {np.inf if best_loss is None else best_loss:.4f}.')
            print(f'Acc (SAT bound = 0.5) = {acc_cnt} / {len(dev_dataset)} = {dev_acc * 100:.2f}%, '
                    f'Acc of SC (SAT bound = 0.5) = {acc_cnt_sc} / {len(dev_dataset)} = {dev_acc_sc * 100:.2f}%, '
                    f'Acc of SC for SAT example (SAT bound = 0.5) = {acc_sat_cnt_sc} / {sat_cnt} = {dev_sat_acc_sc * 100:.2f}%, '
                    f'Acc of SC for UNSAT example (SAT bound = 0.5) = {acc_unsat_cnt_sc} / {unsat_cnt} = {dev_unsat_acc_sc * 100:.2f}%, '
                    f'Acc of SV (SAT bound = 0.5) = {acc_cnt_sv} / {len(dev_dataset)} = {dev_acc_sv * 100:.2f}%, '
                    f'Acc of SV for SAT example (SAT bound = 0.5) = {acc_sat_cnt_sv} / {sat_cnt} = {dev_sat_acc_sv * 100:.2f}%, '
                    f'Acc of SV for UNSAT example (SAT bound = 0.5) = {acc_unsat_cnt_sv} / {unsat_cnt} = {dev_unsat_acc_sv * 100:.2f}%.')
            
            summaryWriter.add_scalar("validation loss of SC", dev_loss_sc, total_step)
            summaryWriter.add_scalar("validation loss of SV", dev_loss_sv, total_step)
            # summaryWriter.add_scalar("validation loss of sharpen", dev_loss_sp, total_step)
            summaryWriter.add_scalar("validation loss", dev_loss, total_step)
            summaryWriter.add_scalar("validation acc of SC", dev_acc_sc * 100, total_step)
            summaryWriter.add_scalar("validation acc of SC in SAT", dev_sat_acc_sc * 100, total_step)
            summaryWriter.add_scalar("validation acc of SC in UNSAT", dev_unsat_acc_sc * 100, total_step)
            summaryWriter.add_scalar("validation acc of SV", dev_acc_sv * 100, total_step)
            summaryWriter.add_scalar("validation acc of SV in SAT", dev_sat_acc_sv * 100, total_step)
            summaryWriter.add_scalar("validation acc of SV in UNSAT", dev_unsat_acc_sv * 100, total_step)
            summaryWriter.add_scalar("validation acc", dev_acc * 100, total_step)
            val_loss.append(dev_loss)

            end = time.time()
            elapsed = end - start
            print(f"Time elapsed: {get_time_str(elapsed)} ({get_time_str(train_time)} + {get_time_str(val_time)} + {get_time_str(elapsed - train_time - val_time)}).")
            epoch_time.append(elapsed)
                
            # 基于最小loss更新best model 
            if best_loss is None or dev_loss <= best_loss:
                best_model = model
                best_loss = dev_loss
                bestPath = join(model_dir, 'SatVSCNet_(SC+TG)+SV-step{%d}-loss{%.4f}-acc{%.4f}-satacc{%.4f}.pth' % (total_step, dev_loss, dev_acc, dev_sat_acc_sv))
                torch.save(best_model.state_dict(), bestPath)
                print(f"Best model save at {bestPath}.")
                epsilon = 0
            else:
                epsilon += 1
                # 判断是否早停
                if epsilon >= early:
                    print(f"Done due to early stopping.")
                    break
            
            # 基于最大acc更新best model 
            # if best_acc is None or dev_acc >= best_acc:
            #     best_model = model
            #     best_acc = dev_acc
            #     bestPath = join(model_dir, 'SatVSCNet_(SC+TG)+SV-step{%d}-loss{%.2f}-acc{%.2f}.pth' % (total_step, dev_loss, dev_acc))
            #     torch.save(best_model.state_dict(), bestPath)
            #     print(f"Best model save at {bestPath}.")
            #     epsilon = 0
            # else:
            #     epsilon += 1
            #     # 判断是否早停
            #     if epsilon >= early:
            #         print(f"Done due to early stopping.")
            #         break

    plotCurve(val_loss,model_dir)
    print(f"Done training. Best model save at {bestPath}.")
    print(f"Avg training time: {get_time_str(np.mean(epoch_time))}.")
    print(f"Total time trained: {get_time_str(np.sum(epoch_time))}.")

def show_model(Model):
    model = Model(
                device, 
                embed_dim=args.hd, 
                num_layers=args.nl,
                node_embed_dim=args.ned,
                mlp_dim=128, 
                max_trace_len=args.mtl,
                soft_check_type=args.sct).to(device)
    print(model)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

if __name__ == '__main__':

    train()
