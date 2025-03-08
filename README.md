# TokenButler: Token Importance Is Predictable


![TokenButler Logo](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/tokenbutlerlogo.png)

This repository contains code to train and evaluate 'token importance' predictors.

All of our results, traces from experiments are located in `ablation_results/`

Note: Our predictor design has improved since the arXiv paper release (We added a layer-norm to stabilize training). Further, to focus on the main predictor design and training-eval scripts, we have removed the ablation scripts. To reproduce the original results and predictor models, please checkout commit `0412fc24a3b770e4d82e6d7064a8172f24c5fcd3` and download the old models.

**We aim to update and improve our models, specifically by training them for longer on a more diverse dataset. Stay tuned for updates on better predictors!**


## Installation

`conda create --name TokenButler python=3.10`

`conda activate TokenButler`

`python -m pip install -r requirements.txt`

## Evaluation
Please download our trained TokenButler predictor models from this [Drive Link](https://drive.google.com/drive/folders/1psNZ1SU0LaZJ-x5MQGH59CzYSmeT4yRf?usp=sharing)

To evaluate, example scripts are provided in `scripts/eval_scan.sh`

### Example script (Please update checkpoint path after downloading models):
```
bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC ExpPred meta-llama/Llama-3.2-3B 1024 16 "<PATH TO CHECKPOINT>"
```

### Modes supported: 
- **TokenButler:** `ExpPred`
- **Oracle:** `oracle`
- **H2O:** `h2o_true` (Generation not supported)
- **SnapKV:** `snapkv` (Generation not supported)
- **Quest:** `quest` (Generation not supported)


**Important note on our evaluation strategy:** To properly test token-eviction/selection methods in a longer decode setting, we _simulate_ token eviction based strategies (SnapKV and H2O) in a purely decode setting. For example, with a 50% token budget, we simulate the entire input sequence as if it were fully decoded, allocating tokens **proportionally to the sequence length** at each decode step. This approach helps profile prefill-eviction-based methods by accurately emulating their token eviction policies across the full input sequence. Unfortunately, this also makes accuracy evaluation slower.

To change downstream evaluation models, modify `task_list`, to evaluate on a smaller subset, modify `eval_subset`. To modify token sparsities being evaluated, modify `{10..60..10}` as desired.


# Training
Our training scripts are located in `scripts/train_predictors.sh`

We provide scripts for the following models:

- **deepseek-ai/DeepSeek-R1-Distill-Llama-8B**
- **meta-llama/Llama-3.2-1B**
- **meta-llama/Llama-3.2-3B**
- **meta-llama/Llama-2-7b-hf**
- **meta-llama/Llama-3.1-8B**
- **mistralai/Mistral-7B-v0.1**
- **Qwen/Qwen2.5-3B**
- **Qwen/Qwen2.5-7B**
- **microsoft/Phi-3.5-mini-instruct**
- **microsoft/Phi-3-mini-4k-instruct**

Training requires 1 A6000 GPU for these variants. Longer-context training is possible using --model_parallelism

# Reasoning Model TokenButler Results
|Method     |Sparsity (%)      |Perplexity|BBH Causal Judgement|MMLU-Pro           |
|-----------|------------------|----------|--------------------|-------------------|
|Dense      |0    |15.87     |0.55  |0.274 |
|TokenButler|12.2 |15.90     |0.56  |0.275 |
|TokenButler|31.0 |15.99     |0.55  |0.273 |
|TokenButler|49.8 |16.22     |0.56  |0.273 |
|TokenButler|68.2 |16.99     |0.55  |0.263 |
|Oracle     |12.2 |15.85     |0.56  |0.273 |
|Oracle     |31.0 |15.76     |0.55  |0.273 |
|Oracle     |49.8 |15.66     |0.54  |0.271 |
|Oracle     |68.3 |15.71     |0.51  |0.271 |


# Predictor Architecture

![Predictor Architecture](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/mainfig.png)

# Custom Synthetic Task

![Custom Synthetic Task](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/datasetfig.png)

## Citation


Coming soon!
