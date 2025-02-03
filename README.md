# TokenButler: Token Importance Is Predictable


![TokenButler Logo](https://github.com/abdelfattah-lab/TokenButler/blob/main/tokenbutlerlogo.png)

This repository contains code to reproduce the experiments in the paper: TokenButler: Token Importance Is Predictable. 

All of our results, traces from experiments are located in `ablation_results/`


## Evaluation
Please download our trained TokenButler predictor models from `To Be Added`

To evaluate, example scripts are provided in `eval_scan.sh`

### Example script (Please update checkpoint path after downloading models):
```
bash eval_any.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC ExpPred meta-llama/Llama-3.2-3B 1024 16 "/home/ya255/projects/all_contextual/expt_model/42_meta-llama_Llama-3.2-3B_False_llama_qk_128_4_c4_realnewslike_0.5_True_False_finetune_None_None_5000_False_False_1_False_False_False_False_4_8_2/0.001_16_None_False_1000_20_1024_fixed_40pc_ExpPred_AllContextual_Jan9_1000_4_2048_1024_1_28_4_16_4_MSE_False_False_L3_3B_2k_1PC.csv_L3_3B_2k_1PC_True_0.38571428571428584_20250111-042334.pt"
```

### Modes supported: 
- **TokenButler:** `ExpPred`
- **Oracle:** `oracle`
- **H2O:** `h2o_true`
- **SnapKV:** `snapkv`
- **Quest:** `quest`


**Note on our evaluation strategy:** SnapKV and H2O are slow because they operate in a purely decode setting. With a 50% token budget, we simulate the entire input sequence as if it were fully decoded, allocating tokens proportionally to the sequence length at each decode step. This approach helps profile prefill-eviction-based methods by accurately emulating their token eviction policies across the full input sequence.

To change downstream evaluation models, modify `task_list`, to evaluate on a smaller subset, modify `eval_subset`. To modify token sparsities being evaluated, modify `{10..60..10}` as desired.


# Training
Our training scripts are located in `train_predictors.sh`

We provide scripts for the following models:
- **meta-llama/Llama-3.2-1B**
- **meta-llama/Llama-3.2-3B**
- **meta-llama/Llama-2-7b-hf**
- **meta-llama/Llama-3.1-8B**
- **mistralai/Mistral-7B-v0.1**
- **Qwen/Qwen2.5-3B**
- **Qwen/Qwen2.5-7B**
- **microsoft/Phi-3.5-mini-instruct**
- **microsoft/Phi-3-mini-4k-instruct**

Training requires 1 A6000 GPU for these variants, model parallelism support for larger trainin runs is WIP!

# Predictor Architecture

![Predictor Architecture](https://github.com/abdelfattah-lab/TokenButler/blob/main/mainfig.png)

# Custom Synethtic Task

![Custom Synthetic Task](https://github.com/abdelfattah-lab/TokenButler/blob/main/datasetfig.png)

## Citation


Coming soon!
