# Learning to SAT-verifiably Check LTL Satisfiability via Differentiable Trace Checking

Code, datasets, and a technical report for the paper "Learning to SAT-verifiably Check LTL Satisfiability via Differentiable Trace Checking" published in ISSTA24.

## Prerequisites

```
cuda = 12.1
python = 3.9.18
torch = 2.1.0+cu121
torch-geometric = 2.4.0
ltlf2dfa = 1.0.1
tensorboard = 2.15.0
tqdm = 4.66.1
```

We have completed all the experiments for our paper on the NVIDIA A100-PCIE-40GB GPU.
The installation of `torch_geometric` may be difficult.
Try `pip install torch_geometric`.
If fail follow the following steps:
- step in the website : [pyg wheel](https://data.pyg.org/whl/)
    find the correct version of your torch
    for example: `torch-2.1.0+cu121`, where the version of torch is 2.1.0 and that of cuda is 12.1.
- step in and download all wheels we need:
    if using python=3.9, sys=linux, then
    ``` 
        pyg_lib-0.4.0+pt21cu121-cp39-cp39-linux_x86_64.whl
        torch_cluster-1.6.3+pt21cu121-cp39-cp39-linux_x86_64.whl
        torch_scatter-2.1.2+pt21cu121-cp39-cp39-linux_x86_64.whl
        torch_sparse-0.6.18+pt21cu121-cp39-cp39-linux_x86_64.whl
        torch_spline_conv-1.2.2+pt21cu121-cp39-cp39-linux_x86_64.whl 
    ```
- install all the wheels downloaded and `pip install torch_geometric`

## Use examples

To replicate the experiments and reproduce the results of the best model (VSCNet-G) reported in the paper, please execute the following commands:

```
python train.py --debug 0 --device 0 --lr 0.001 --wdr 0 --bs 1024 --nl 10 --hd 512 --ned 256 --mtl 5 --mu_sc 1 --mu_sv 1 --mu_sp 0  --sct minmax
```

Select the converged model for testing.
Please execute the following commands, where `[bast model]` should be replaced with the path of the best model, e.g., `model/2024_04_07_19_24_33_sd_87710306_lr_0.001_wdr_0_bs_1024_nl_10_hd_512_ned_256_mtl_5_mu_sc_1.0_mu_sv_1.0_mu_sp_0.0_sct_minmax/SatVSCNet_(SC+TG)+SV-step{36424}-loss{0.0997}-acc{0.9894}-satacc{0.9042}.pth`:

```
python test.py --debug 0 --device 0 --sbm "[bast model]" --trp "./result_1.txt" --ted "./data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/test_trace.json" --pted "./data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/test_trace_prep.json" --bs 1 --nl 10 --hd 512 --ned 256 --mtl 5
```

The dataset files are located in the "data" directory. You will need to modify the --ted and --pted files, where --pted is the pre-processed file generated from the first run of --ted.

By running the testing command, you can obtain:
```
Total test time (FE + TG): 174.2188
Total test time (SV): 16.8735
Tatal predict true times: 1068203
(TP, TN, FP, FN) = (9939, 9854, 146, 61)
(Acc, P, R, F1) = (0.9896, 0.9855, 0.9939, 0.9897)
Acc of SV (9298 / 10000) = 0.9298
```

The sum of the value of `Total test time (FE + TG)` and the value of `Total test time (SV)` corresponds to `time`.
`Acc`, `P`, `R`, and `F1` in `(Acc, P, R, F1)` corresponds to `acc.` `pre.`, `rec.`, and `F1` respectively.
`Acc of SV` corresponds to `sacc`.

## Citation

Please consider citing the following paper if you find our codes helpful. 
Thank you!

```
@inproceedings{LuoLQCWDF24,
  author       = {Weilin Luo and
                  Pingjia Liang and
                  Junming Qiu and
                  Polong Chen and
                  Hai Wan and
                  Jianfeng Du and
                  Weiyuan Fang},
  title        = {Learning to SAT-verifiably Check LTL Satisfiability via Differentiable Trace Checking},
  booktitle    = {ISSTA},
  year         = {2024}
}
```