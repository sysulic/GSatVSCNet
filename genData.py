# -*- coding: utf-8 -*-

import torch
from os.path import exists
import json
import sys
from tqdm import trange
from collections import deque
from copy import deepcopy
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
from torch.utils.data import Dataset
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually

from logic_checker import list_to_nested_tuple

sys.setrecursionlimit(10 ** 5)
json.decoder.MAX_DEPTH = 10 ** 5

bop = ['&', '|', 'U']
uop = ['!', 'X', 'F', 'G']
node_op_type = {'&': 4, '|': 5, '!': 6, 'X': 7, '': 0}
binOp = [LTLfAnd, LTLfOr, LTLfUntil]
uOp = [LTLfNot, LTLfNext, LTLfEventually, LTLfAlways]
Op = [LTLfNot, LTLfAnd, LTLfNext, LTLfUntil, LTLfOr, LTLfEventually, LTLfAlways]

# node_mapper #  sub_for 0  expanded_subfor 1  atom 2  root 3  & 4  | 5  ！6  X 7 
map_com = {(0, 0, 0, 0, 0, 0, 0, 0) : 0, (0, 0, 0, 1, 1, 0, 0, 0) : 1, (0, 0, 0, 1, 0, 1, 0, 0) : 2, (0, 0, 0, 1, 0, 0, 1, 0) : 3, (0, 0, 0, 1, 0, 0, 0, 1) : 4,
         (1, 0, 0, 0, 1, 0, 0, 0) : 5, (1, 0, 0, 0, 0, 1, 0, 0) : 6, (1, 0, 0, 0, 0, 0, 1, 0) : 7, (1, 0, 0, 0, 0, 0, 0, 1) : 8, (0, 0, 1, 0, 0, 0, 0, 0) : 9,
         (0, 1, 0, 0, 1, 0, 0, 0) : 10, (0, 1, 0, 0, 0, 0, 0, 1) : 11}  # 0 Global Node
map_sim = {(0, 0, 0, 0, 0, 0, 0, 0) : 0, (0, 0, 0, 1, 1, 0, 0, 0) : 1, (0, 0, 0, 1, 0, 1, 0, 0) : 2, (0, 0, 0, 1, 0, 0, 1, 0) : 3, (0, 0, 0, 1, 0, 0, 0, 1) : 4,
         (1, 0, 0, 0, 1, 0, 0, 0) : 1, (1, 0, 0, 0, 0, 1, 0, 0) : 2, (1, 0, 0, 0, 0, 0, 1, 0) : 3, (1, 0, 0, 0, 0, 0, 0, 1) : 4, (0, 0, 1, 0, 0, 0, 0, 0) : 5,
         (0, 1, 0, 0, 1, 0, 0, 0) : 2, (0, 1, 0, 0, 0, 0, 0, 1) : 4}  # 0 Global Node

parser = LTLfParser()

class GLDataSet(Dataset):
    def __init__(self, path, prep_path=None, node_map=0):
        self.path = path
        self.prep_path = prep_path
        
        self.data = []
        self.var_dict = {}
        self.idx = 0
        if not node_map:
            self.node_map = map_com
        else:
            self.node_map = map_sim
        if self.prep_path is None or not exists(self.prep_path):
            self.process()
        self.read_data_from_preprocess()
        # super().__init__(root, None, None, None) 
        # self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        return self.data[item]

    def read_data_from_preprocess(self):
        '''
        从预处理数据集读取数据
        '''
        print(f"Processing data from {self.prep_path}.")
        with open(self.prep_path, "r") as f:
            dataset = json.load(f)
            for i in trange(len(dataset), ncols=100, desc=f'Processing'):
                data = dataset[i]

                tree_tuple = list_to_nested_tuple(data['tree_tuple'])

                self.data.append((data['x'], data['edge_index'], data['y'], data['ver_list'], data['num_node'],
                                  data['u_index'], data['inorder'], data['op_list'], data['atom_list'],
                                  data['right_list'], data['var_dict'], data['T2G'], tree_tuple))

    def process(self):

        print(f"Processing data from {self.path}.")
        with open(self.path, "r") as f:
            dataset = json.load(f)
            data_list = []

            for i in trange(len(dataset), ncols=100, desc=f'Processing'):
                data = dataset[i]
                f_raw, y = data['inorder'], data['issat']
                inorder = f_raw
                y = 1 if y else 0    
                f_raw = parser(f_raw)
                self.idx = 0
                
                subformulas = self.extract_subformulas(f_raw)
                # print(subformulas)

                expanded_subformulas = self.expand_all_subformulas(subformulas)
                # print(expanded_subformulas)
                x, edge_index, ver_list, u_index = self.ltl_to_coo(expanded_subformulas)

                y = torch.tensor(y, dtype=torch.long)
                num_node = len(ver_list)

                self.var_dict = {}
                op_list, atom_list, right_list = [], [], []
                tree_tuple = self.read_diff_tree(f_raw, op_list, atom_list, right_list)
                
                T2G = {}    # atom_list ： ver_list
                for ver in ver_list:
                    if ver in self.var_dict:
                        T2G[self.var_dict[ver]] = ver_list.index(ver)
                T2G = tuple(T2G.items())

                example = {}
                example['x'] = x.tolist()
                example['edge_index'] = edge_index.tolist()
                example['y'] = y.tolist()
                example['ver_list'] = ver_list
                example['num_node'] = num_node
                example['u_index'] = u_index
                example['inorder'] = inorder
                example['op_list'] = op_list
                example['atom_list'] = atom_list
                example['right_list'] = right_list
                example['var_dict'] = self.var_dict
                example['T2G'] = T2G
                example['tree_tuple'] = tree_tuple
                data_list.append(example)
            
        with open(self.prep_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_list, indent=4))
            f.close
        print(f"Save Preprocess data in {self.prep_path}")

    def read_atom(self, a: str):
        '''
        输出：原子命题index
        '''

        if a not in self.var_dict: # 如果原子命题不在字典中
            idx = len(self.var_dict)
            self.var_dict[a] = idx

        else: # 如果原子命题在字典中
            idx = self.var_dict[a]

        return idx
    
    def read_diff_tree(self, f, op_list, atom_list, right_list):
        '''
        构建可微的LTL树结构.
        
        输入:   
            - f:                                    LTL公式.
            - preorder_traversal = (op,atom,right): 先序遍历序列.
        
        输出:   
            - 二叉树元组（op，左子树，右子树），子树如果是原子命题是字符串，否则为二叉树元组。
        '''
        current_t = len(atom_list)
        op_str, left_element, right_element = None, None, None

        op_list.append(-1)
        atom_list.append(-1)
        right_list.append(-1)

        if isinstance(f, LTLfAnd):
            op_list[current_t] = 2
            op_str = '&'
        elif isinstance(f, LTLfOr):
            op_list[current_t] = 5
            op_str = '|'
        elif isinstance(f, LTLfUntil):
            op_list[current_t] = 4
            op_str = 'U'
        elif isinstance(f, LTLfNot):
            op_list[current_t] = 1
            op_str = '!'
        elif isinstance(f, LTLfNext):
            op_list[current_t] = 3
            op_str = 'X'
        elif isinstance(f, LTLfEventually):
            op_list[current_t] = 6
            op_str = 'F'
        elif isinstance(f, LTLfAlways):
            op_list[current_t] = 7
            op_str = 'G'

        # binOp
        for _, t in enumerate(binOp):
            if isinstance(f, t):
                if isinstance(f.formulas[0], LTLfAtomic):
                    left_element = f.formulas[0].s
                    PA_l = self.read_atom(f.formulas[0].s)
                    op_list.append(-1)
                    atom_list.append(PA_l)
                    right_list.append(-1)
                else:
                    left_element = self.read_diff_tree(f.formulas[0], op_list, atom_list, right_list)

                right_list[current_t] = len(atom_list)
                
                if len(f.formulas) == 2:
                    if isinstance(f.formulas[1], LTLfAtomic):
                        right_element = f.formulas[1].s
                        PA_r = self.read_atom(f.formulas[1].s)
                        op_list.append(-1)
                        atom_list.append(PA_r)
                        right_list.append(-1)
                    else:
                        right_element = self.read_diff_tree(f.formulas[1], op_list, atom_list, right_list)
                else: # 如果不止两个分支，需要递归分解为两个分支的子公式
                    nf = deepcopy(f)
                    nf.formulas = nf.formulas[1:]
                    right_element = self.read_diff_tree(nf, op_list, atom_list, right_list)
                
                return (op_str, left_element, right_element)

        # unary
        for _, t in enumerate(uOp): 
            if isinstance(f, t):
                if isinstance(f.f, LTLfAtomic):
                    left_element = f.f.s
                    PA_l = self.read_atom(f.f.s)
                    op_list.append(-1)
                    atom_list.append(PA_l)
                    right_list.append(-1)
                else:
                    left_element = self.read_diff_tree(f.f, op_list, atom_list, right_list)

                return (op_str, left_element)

    def get_subformulas(self, formula):
        """
        get the op and sub-formulas in formula
        :param formula: str, LTL formula
        :return: op: LTL operator, sub1: left sub-formula sub2: right sub-formula
        """
        depth = 0
        op = ''
        is_bop = 0
        sub1 = ''
        sub2 = ''
        for i,char in enumerate(formula):
            if depth == 0 and char in uop:
                op = char
            elif depth == 0 and char in bop:
                op = char
                is_bop = 1
            elif char == '(':
                if depth == 0:
                    start = i
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0 and not is_bop:
                    sub1 = formula[start:i+1]
                if depth == 0 and is_bop:
                    sub2 = formula[start:i+1]
        return op, sub1, sub2

    def extract_subformulas(self, formula):
        """
        非递归方式抽取LTL公式中的所有子公式
        :param formula: str, LTL公式
        :return: dict, 包含所有子公式的字典，key为子公式编号，value为子公式字符串
        """
        subformulas = {}  # 用字典存储子公式
        sym_sub = {} # 存储子公式对称值和键
        que = deque([(formula, "0")]) # 符号队列
        while que:
            f, f_id = que.pop()
            if f_id not in subformulas:
                # binary
                for i, t in enumerate(binOp): 
                    if isinstance(f,t): 
                        op = bop[i]
                        sub1 = f.formulas[0]
                        # 是否出现
                        is_ocur = str(sub1)
                        is_ocur = self.rem_par(is_ocur)

                        if is_ocur not in sym_sub:
                            if isinstance(sub1, LTLfAtomic):
                                sub1 = self.rem_par(str(sub1))
                            else:  
                                self.idx += 1
                                que.appendleft((sub1, f'{self.idx}'))
                                sub1 = f'{self.idx}'
                        else:
                            sub1 = sym_sub[is_ocur]                     

                        if len(f.formulas) == 2:    # 两个分支
                            sub2 = f.formulas[1]
                            is_ocur1 = is_ocur
                            is_ocur = str(sub2)
                            is_ocur = self.rem_par(is_ocur)

                            if is_ocur == is_ocur1:
                                sub2 = sub1
                            elif is_ocur not in sym_sub:
                                if isinstance(sub2, LTLfAtomic):
                                    sub2 = self.rem_par(str(sub2))
                                else:
                                    self.idx += 1
                                    que.appendleft((sub2, f'{self.idx}'))
                                    sub2 = f'{self.idx}'
                            else: 
                                sub2 = sym_sub[is_ocur]
                            if op == 'U':
                                subformulas[f'{f_id}'] = f'({sub1}){op}({sub2})'
                                sym_sub[f'{sub1} {op} {sub2}'] = f'{f_id}'
                            else:       # 满足不变性 & |
                                subformulas[f'{f_id}'] = f'({sub1}){op}({sub2})'
                                sym_sub[f'{sub1} {op} {sub2}'] = f'{f_id}'
                                sym_sub[f'{sub2} {op} {sub1}'] = f'{f_id}'

                        else: # 如果不止两个分支，则存为多叉树，后续不会被展开 这里还没解决多叉树的置换不变性
                                nf = deepcopy(f)                              
                                sub = nf.formulas

                                ids = 1         # 记录出现的子公式index
                                sub_s = ""      # sub_formulas
                                sym_s = ""      # sym_sub
                                for i, ssub in enumerate(sub):
                                    n = len(sub) - 1 
                                    if isinstance(ssub, LTLfAtomic):
                                        sub_s += f'({str(ssub)})'
                                        sym_s += f'{str(ssub)}'
                                        if i != n:
                                            sub_s += op
                                            sym_s += f' {op} '

                                    else:

                                        is_ocur = str(ssub)
                                        is_ocur = self.rem_par(is_ocur)
                                        if is_ocur not in sym_sub:
                                            self.idx += 1
                                            que.appendleft((ssub, f'{self.idx}'))
                                            sub_s += f'({self.idx})'
                                            sym_s += f'{self.idx}'
                                            ids += 1
                                        else:
                                            is_ocur = sym_sub[is_ocur]
                                            sub_s += f'({is_ocur})'
                                            sym_s += f'{is_ocur}'

                                        if i != n:
                                            sub_s += op
                                            sym_s += f' {op} '

                                subformulas[f'{f_id}'] = sub_s
                                sym_sub[sym_s] = f'{f_id}'
                                sym_sub[sym_s] = f'{f_id}'

                # unary
                for i, t in enumerate(uOp): 
                    if isinstance(f, t):
                        op = uop[i]
                        sub1 = f.f
                        is_ocur = str(sub1)
                        is_ocur = self.rem_par(is_ocur)

                        if is_ocur not in sym_sub:   # 是否出现  
                            if isinstance(sub1, LTLfAtomic):
                                sub1 = self.rem_par(str(sub1))
                            else:
                                self.idx += 1
                                que.appendleft((sub1, f'{self.idx}'))
                                sub1 = f'{self.idx}'
                        else:
                            
                            sub1 = sym_sub[is_ocur]

                        subformulas[f'{f_id}'] = f'{op}({sub1})'
                        sym_sub[f'{op}({sub1})'] = f'{f_id}'
        # print(sym_sub)
        return subformulas

    def one_step_expansion(self, key, formula):
        """
        one step unfold the formula
        :param key: str, no. of formula
        :param formula: str, LTL formula
        :return: str, the unfolded formula
        """
        no_need = ["&", "|", "!", "X"]
        op, sub1, sub2 = self.get_subformulas(formula)
        nx_ltl = ''
        if op in no_need:
            return formula, 0
        elif op == "F":
            nx_ltl = f"{sub1}|(X({key}))"
        elif op == 'U':
            nx_ltl = f"{sub2}|({sub1}&(X({key})))"
        elif op == 'G':
            nx_ltl = f"{sub1}&(X({key}))"
        elif op == '':
            return formula, 0
        else:
            raise ValueError("Invalid LTL formula")
        return nx_ltl, 1

    def expand_all_subformulas(self, subformulas):
        expanded_subformulas = {}
        for key, value in subformulas.items():
            expanded_value, is_expanded = self.one_step_expansion(key, value)
            expanded_subformulas[key] = (expanded_value, is_expanded)
        return expanded_subformulas

    def ltl_to_coo(self, formula_dic):
        """
        get the data likes coo of the LTL formula.
        :param formula_dic: key: no. of formula, value:(subformula, whether unfolded)
        """
        
        sub_dict = deepcopy(formula_dic)

        vertices = dict()
        for key in sub_dict:
            if key == '0' and key not in vertices:
                vertices[key] = [3, 0]
            elif key not in vertices:
                vertices[key] = [0, 0]

        for key, value in sub_dict.items():
            op, sub1, sub2 = self.get_subformulas(value[0])
            vertices[key][1] = node_op_type[op]      

            sub = value[0].split(op)
            if op in bop and len(sub) == 2:
                if sub1[1:-1] not in vertices:
                    vertices[sub1[1:-1]] = [value[1], 0]
                if sub2[1:-1] not in vertices:
                   vertices[sub2[1:-1]] = [value[1], 0]
            elif len(sub) > 2:
                for ssub in sub:
                    if ssub[1:-1] not in vertices:
                        vertices[ssub[1:-1]] = [0, 0]
            
            elif op in uop:
                if sub1[1:-1] not in vertices: 
                    vertices[sub1[1:-1]] = [value[1], 0]

            
        tmp = deepcopy(vertices)

        for key, value in tmp.items():
            op, sub1, sub2 = self.get_subformulas(key)

            if op != '':
                vertices[key][1] = node_op_type[op] 
            if sub1 != '' and sub1[1:-1] not in vertices:
                vertices[sub1[1:-1]] = [value[0], node_op_type[self.get_subformulas(sub1[1:-1])[0]]]
            if sub2 != '' and sub2[1:-1] not in vertices:
                vertices[sub2[1:-1]] = [value[0], node_op_type[self.get_subformulas(sub2[1:-1])[0]]]

        ver_list = []
        x = []

        for key, value in vertices.items():
            ver_list.append(key)
            y = [0, 0, 0, 0, 0, 0, 0, 0]
            if key.startswith('p'):
                vertices[key] = 2
                y[2] = 1
            else:
                y[value[0]] = 1
                y[value[1]] = 1

            x.append(y)

        edge_index = [[],[]]
        for f_id, subformula in sub_dict.items():
            parent_idx = ver_list.index(f_id)
            op, sub1, sub2 = self.get_subformulas(subformula[0])

            sub = subformula[0].split(op)
            if op in bop and len(sub) >= 2:
                for ssub in sub:
                    edge_index[0].append(parent_idx)
                    edge_index[1].append(ver_list.index(ssub[1:-1]))
                    
                    edge_index[0].append(ver_list.index(ssub[1:-1]))
                    edge_index[1].append(parent_idx)


            elif op in uop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))

                
                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)

        for ver in ver_list:
            parent_idx = ver_list.index(ver)
            op, sub1, sub2 = self.get_subformulas(ver)
            if op in bop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))

                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)


                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub2[1:-1]))

                edge_index[0].append(ver_list.index(sub2[1:-1]))
                edge_index[1].append(parent_idx)


            if op in uop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))


                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)

        x.append([0, 0, 0, 0, 0, 0, 0, 0])  # global node

        xx = []  
        for i in x:
            xx.append(self.node_map[tuple(i)])

        ver_list.append('U')       
        u_index = len(ver_list) - 1
        for i in range(len(ver_list)-1):
            edge_index[0].append(i)
            edge_index[1].append(u_index)

            edge_index[0].append(u_index)
            edge_index[1].append(i)
        x = torch.tensor(xx, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # print(ver_list)
        return x, edge_index, ver_list, u_index
    
    def rem_par(self, formula):
        if formula.startswith("(") and formula.endswith(")"):
            counter = 0
            for i in range(1, len(formula)-1):
                if formula[i] == "(":
                    counter += 1
                elif formula[i] == ")":
                    counter -= 1
                if counter < 0:
                    break
            if counter == 0:
                formula = formula[1:-1]
                formula = self.rem_par(formula)
        return formula
    
    def download(self):
        pass

def collate(batch):

    return get_batch_matrix(batch)

def get_batch_matrix(batch):
    '''
    输入: [x, edge_index, y, ver_list, num_node, u_index, inorder, op_list, atom_list, right_list, var_dict, T2G, tree_tuple] * batch_size
        - x in num_vertex   : OSUG图节点类型信息，x[i]表示节点i的类型，i是节点编号。其中，9表示该节点是原子命题节点，0代表全局节点，其余0-11分别代表某种类型的节点。
        - edge_index in 2 * num_vertex  : OSUG图边信息，edge_list[0][i]表示第i条边的初始节点编号，edge_list[1][i]表示第i条边的终止节点编号（有向）。
        - y    : OSUG图分类（1：SAT，0：UNSAT）。
        - ver_list in num_vertex        : OSUG图节点标记的公式，ver_list[i]表示节点i标记的公式。
        - num_node : OSUG图节点个数。
        - u_index  : OSUG图全局节点索引（标记为“U”）。
        - inorder  : 公式的中缀表示。
        - op_list in formula_len    : 语法树先序遍历，如果是逻辑运算符，则为逻辑运算符id（0:none（padding不同长度的公式时使用）, 1:!, 2:&, 3:X, 4:U, 5:|, 6:F, 7:G）；否则为-1。
        - atom_list in formula_len  : 语法树先序遍历，如果是原子命题，则为原子命题id；否则为-1。
        - right_list in formula_len : 语法树先序遍历，如果存在右子公式，则为右子公式id；否则为-1;
        - var_dict  : 语法树的原子命题和id对应pair。
        - T2G       : 语法树的原子命题id和OSUG图节点id对应pair。
        - tree_tuple: 二叉树元组（op，左子树，右子树），子树如果是原子命题是字符串，否则为二叉树元组。

    输出：
        - x_batch in N^{all_num_ver}                        : x_batch[i]表示节点i的类型。
        - edge_index_batch in N^{2, all_num_edge}           : edge_index_batch[i][j]表示节点id，<edge_index_batch[0][i],edge_index_batch[1][i]>表示从节点edge_index_batch[0][i]到edge_index_batch[1][i]有边。
        - u_index_batch in N^{batch_size}                   : u_index_batch[i]表示第i个实例的全局节点。
        - G2T_index_batch in N^{batch_size * max_var_num}   : G2T_index_batch[i][j]表示第i个实例的语法树的原子命题j对应的图节点id。
        - example_mark_batch in N^{all_num_ver}             : example_mark_batch[i]表示节点i属于的实例，实例标识从0开始。
        - y_batch in N^{batch_size}                         : y_batch[i]表示第i个实例的分类（0: UNSAT，1: SAT）。
        - inorder_list in batch_size                        : 公式的中缀表示。
        - op_chose_batch in [0,1]^{batch_size * (1 + |OP|) * max_formula_len}           : 子公式i的公式类型是否是j, 其中i是第2维度, j是第1维度, 公式类型按序如下: none,!,&,X,U,|,F,G。
        - atom_chose_batch in [0,1]^{batch_size * |AP| * max_formula_len}               : 子公式i是否是原子命题j, 其中i是第2维度, j是第1维度。
        - right_chose_batch in [0,1]^{batch_size * max_formula_len * max_formula_len}   : 子公式i的右子公式是否是子公式j, 其中i是第2维度, j是第1维度。
        - atom_mask in R^{batch_size * |AP|}    : 与公式有关的原子命题mask。
        - tree_tuple_list                       : 二叉树元组列表。
    '''
    
    batch_size = len(batch)
    op_2_dim = 1+len(Op)
    max_formula_len = max(len(batch[i][7]) for i in range(batch_size))

    # 以下操作在CPU运行，所以非常慢！！！！！

    # OSUG图结构
    x_batch = torch.cat([torch.tensor(batch[i][0]) for i in range(batch_size)]) # 注意，这里将一个batch中所有的图合成一个大图，所以图节点的id顺移

    offset = [0] + [sum(batch[j][4] for j in range(i)) for i in range(1, batch_size)] # 每个实例节点id的偏移

    edge_index_batch = torch.cat([torch.tensor(batch[i][1]) + torch.tensor(offset[i]) for i in range(batch_size)], 1)

    u_index_batch = (x_batch == 0).nonzero(as_tuple=False).squeeze(-1) # 0是全局节点类型

    max_var_num = max(len(batch[i][11]) for i in range(batch_size)) # batch中最大的原子命题数目
    G2T_index_batch = -1 * torch.ones(size=(batch_size, max_var_num), dtype=torch.int64)
    for i in range(batch_size):
        T2G = batch[i][11]
        for j in T2G:
            G2T_index_batch[i][j[0]] = j[1] + offset[i]
    
    example_mark_batch = torch.cat([i*torch.ones((len(batch[i][0]), ),dtype=torch.int64) for i in range(batch_size)])

    y_batch = torch.cat([torch.tensor([batch[i][2]]) for i in range(batch_size)])

    inorder_list = [batch[i][6] for i in range(batch_size)]
    var_dict_list = [batch[i][10] for i in range(batch_size)]
    tree_tuple_list = [batch[i][12] for i in range(batch_size)]
    
    op_chose_tmp = pad_sequence([tensor(batch[i][7]) for i in range(batch_size)], padding_value=0).t() # op_chose_tmp in batch_size * max_formula_len: padding 不同长度的公式。padding_value=0，因为op id=0是none，预留标记padding的
    op_chose_tmp = torch.where(op_chose_tmp==-1, op_2_dim, op_chose_tmp) # op_chose_tmp in batch_size * max_formula_len: 标记非op为1+len(Op)
    op_chose_tmp = torch.index_select(one_hot(op_chose_tmp, num_classes=op_2_dim + 1), 2, torch.tensor(range(op_2_dim))) # op_chose_tmp in batch_size * max_formula_len * (1 + |OP|): one-hot编码op选择。注意，非op是1+len(Op)维被舍弃
    op_chose_batch = torch.transpose(op_chose_tmp, dim0=1, dim1=2).float() # op_chose_tmp in batch_size * (1 + |OP|) * max_formula_len
    
    atom_chose_tmp = pad_sequence([tensor(batch[i][8]) for i in range(batch_size)], padding_value=max_var_num).t() # atom_chose_tmp in batch_size * max_formula_len: padding 不同长度的公式。padding_value=max_var_num，因为标记非原子命题为max_var_num
    atom_chose_tmp = torch.where(atom_chose_tmp==-1, max_var_num, atom_chose_tmp) # atom_chose_tmp in batch_size * max_formula_len: 标记非原子命题为max_var_num
    atom_chose_tmp = torch.index_select(one_hot(atom_chose_tmp, num_classes=max_var_num + 1), 2, torch.tensor(range(max_var_num))) # atom_chose_tmp in batch_size * max_formula_len * |AP|: one-hot编码原子命题选择。注意，非原子命题是max_var_num维被舍弃
    atom_chose_batch = torch.transpose(atom_chose_tmp, dim0=1, dim1=2).float() # atom_chose_tmp in batch_size * |AP| * max_formula_len
    
    right_chose_tmp = pad_sequence([tensor(batch[i][9]) for i in range(batch_size)], padding_value=max_formula_len).t() # right_chose_tmp in batch_size * max_formula_len: padding 不同长度的公式。padding_value=max_formula_len，因为标记没有右子公式为max_formula_len
    right_chose_tmp = torch.where(right_chose_tmp==-1, max_formula_len, right_chose_tmp) # right_chose_tmp in batch_size * max_formula_len: 标记没有右子公式为max_formula_len
    right_chose_tmp = torch.index_select(one_hot(right_chose_tmp, num_classes=max_formula_len + 1), 2, torch.tensor(range(max_formula_len))) # atom_chose_tmp in batch_size * max_formula_len * max_formula_len: one-hot编码右子公式选择。注意，没有右子公式是max_formula_len维被舍弃
    right_chose_batch = torch.transpose(right_chose_tmp, dim0=1, dim1=2).float()

    atom_mask_batch = torch.zeros(size=(batch_size, max_var_num), dtype=torch.int64) # atom_mask_batch in batch_size * |AP|
    atom_tmp = torch.index_select(torch.nonzero(atom_chose_batch, as_tuple=False), dim=1, index=torch.tensor([0,1])) # atom_tmp in num_nonzero * 2: 表示(第几个实例，出现在实例中的原子命题id)， torch.nonzero(atom_chose_batch, as_tuple=False): 获得非0元素索引
    atom_tmp = [ele for ele in atom_tmp.t()] # atom_tmp[0][i]: 第i个非0值属于哪个实例；atom_tmp[1][i]: 第i个非0值
    atom_mask_batch.index_put_(atom_tmp, torch.tensor([1])) # atom_mask_batch[atom_tmp[0][i]][atom_tmp[1][i]] = 1
    atom_mask_batch = atom_mask_batch.float()

    # for i in range(len(batch)):
    #     print(f'inorder_list: {inorder_list[i]}')
    #     print(f'inorder: {batch[i][6]}')
    #     print(f'tree_tuple_list: {tree_tuple_list[i]}')
    #     print(f'tree_tuple: {batch[i][12]}')
    #     print(f'y_batch: {y_batch[i]}')
    #     print(f'y: {batch[i][2]}')

    #     print('Tree')
    #     print(f'op_list: {batch[i][7]}')
    #     print(f'atom_list: {batch[i][8]}')
    #     print(f'right_list: {batch[i][9]}')
    #     print(f'var_dict: {batch[i][10]}')
    #     print(f'T2G: {batch[i][11]}')

    #     print('OSUG')
    #     print(f'x: {batch[i][0]}')
    #     print(f'edge_index: {batch[i][1]}')
    #     print(f'ver_list: {batch[i][3]}')
    #     print(f'num_node: {batch[i][4]}')
    #     print(f'u_index: {batch[i][5]}')

    # print('batch Tree')
    # print(f'op_chose_batch: {op_chose_batch}')
    # print(f'atom_chose_batch: {atom_chose_batch}')
    # print(f'right_chose_batch: {right_chose_batch}')
    # print(f'atom_mask_batch: {atom_mask_batch}')
    
    # print('batch OSUG')
    # print(f'x_batch: {x_batch}')
    # print(f'edge_index_batch: {edge_index_batch}')
    # print(f'u_index_batch: {u_index_batch}')
    # print(f'example_mark_batch: {example_mark_batch}')
    # print(f'G2T_index_batch: {G2T_index_batch}')
        
    # print_OSUG(x_batch.tolist(),edge_index_batch.tolist(),[item for sublist in [batch[i][3] for i in range(batch_size)] for item in sublist])
    
    return x_batch, edge_index_batch, u_index_batch, G2T_index_batch, example_mark_batch, y_batch, atom_mask_batch, op_chose_batch, atom_chose_batch, right_chose_batch, inorder_list, var_dict_list, tree_tuple_list

if __name__ == '__main__':

    trd = 'data/test/debug.json'
    ptrd = 'data/test/debug_prep.json'

    dataset = GLDataSet(trd, prep_path=ptrd)