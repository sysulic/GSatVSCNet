# Getting Started

## environment

```
cuda = 12.1
python = 3.9.18
torch = 2.1.0+cu121
torch-cluster = 1.6.3+pt21cu121
torch-geometric = 2.4.0
torch-scatter = 2.1.2+pt21cu121
torch-sparse = 0.6.18+pt21cu121
torch-spline-conv = 1.2.2+pt21cu121
ltlf2dfa = 1.0.1
tensorboard = 2.15.0
tqdm = 4.66.1
```

We have completed all the experiments for our paper on the NVIDIA A100-PCIE-40GB GPU.
To replicate the results for the OSUG model in Table 3 and the best OSUG model in Figure 3, please execute the following commands:

```
python test.py --sbm ./best_model/SatVSCNet_\(SC+TG\)+SV-step\{36424\}-loss\{0.0997\}-acc\{0.9894\}-satacc\{0.9042\}.pth --trp ./log/result.txt --ted data/LTLSATUNSAT-\{and-or-not-F-G-X-until\}-100-random/\[200-250\)/test_trace.json --pted data/LTLSATUNSAT-\{and-or-not-F-G-X-until\}-100-random/\[200-250\)/test_nv_1024_trace_prep.json --hd 512 --device 4
```

Estimated time: For every 2000 LTL formulas, the testing process requires 1 minutes of pre-processing time and 2 minute of testing time.

The datasets reported in the paper need to be run individually. The dataset files are located in the "data" directory. You will need to modify the --ted and --pted files, where --pted is the pre-processed file generated from the first run of --ted.

# Detailed Description

To replicate the experiments and reproduce the results of the best model reported in the paper, please execute the following commands:

```
python train.py --debug 0 --device 4 --bs 1024 --nl 10 --hd 512 --ned 256 --mtl 5 --mu_sc 1 --mu_sv 1 --mu_sp 0 --sd 87710306 --lr 0.001 --wdr 0 --sct minmax
```
by running the testing command, you can obtain:
```
Total test time (FE + TG): 2980.7709
Total test time (SV): 27.8194
Tatal predict true times: 135251
(TP, TN, FP, FN) = (957, 988, 12, 43)
(Acc, P, R, F1) = (0.9725, 0.9876, 0.9570, 0.9721)
Acc of SV (1 / 1000) = 0.0010
```

`Total test time (FE + TG)` Corresponds to the `Running time` in line chart 3 in Figure 3
`ACC` in `(Acc, P, R, F1)` Corresponds to the `Semantic accuracy of trace generation` in line chart 2 in Figure 3
`F1` in `(Acc, P, R, F1)` Corresponds to the `F1 score of satisfiability checking` in line chart 1 in Figure 3
