# TokenButler: Token Importance Is Predictable


![TokenButler Logo](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/tokenbutlerlogo.png)

This repository contains code to train and evaluate 'token importance' predictors.

All of our results, traces from experiments are located in `ablation_results/`

Note: Our predictor design has improved since the arXiv paper release (We added a layer-norm to stabilize training). Further, to focus on the main predictor design and training-eval scripts, we have removed the ablation scripts. To reproduce the original results and predictor models, please checkout commit `0412fc24a3b770e4d82e6d7064a8172f24c5fcd3` and download the old models. 

For the latest, new models, try the huggingface integration. [Wandb-Logs](https://wandb.ai/akhauriyash/TrainTokenButler) for trained models.

## Huggingface Integration

We support the following models directly through huggingface-transformers:

- **DeepSeek-R1-Distill-Llama-8B-Butler**
- **Llama-3.1-8B-Butler**
- **Llama-2-7b-hf-Butler**
- **Llama-3.2-3B-Butler**
- **Llama-3.2-1B-Butler**

The collection of models can be found [here](https://huggingface.co/collections/akhauriyash/tokenbutler-67cf181b5762d0d60e5f312b)

Simply run `test_hf.py` or:

```
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

question = "If millionaires have butlers, why don't million dollar language models have a butler too? I think its because "

model_name = "akhauriyash/DeepSeek-R1-Distill-Llama-8B-Butler"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator(question, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)

print(response[0]['generated_text'][len(question):])
```

Note that the 'default' configured sparsity is 50%. Further, there is a 'sliding window' of 128 and 8 'anchor tokens'. To 'change' the sparsity, you can use the following function after loading the model. Please note that the 'fixed' is the only supported strategy at the moment, which 'fixes' the sparsity of each layer (except the first) at the 'pc' (percentage) mentioned. This can also be found at `test_hf.py`. Sliding window and anchor tokens can be changed in a similar manner.

```
def set_sparsity(model, sparsity):
    for module in model.modules():
        if module.__class__.__name__.__contains__("AttentionExperimental"):
            module.token_sparse_method = sparsity
            module.set_token_sparsity()
    return model

model = set_sparsity(model, "fixed_60pc")
```

## Installation

```
conda create --name TokenButler python=3.10
conda activate TokenButler
python -m pip install -r requirements.txt
```

## Evaluation
Please download our trained (old) TokenButler predictor models from this [Drive Link](https://drive.google.com/drive/folders/1psNZ1SU0LaZJ-x5MQGH59CzYSmeT4yRf?usp=sharing)


To evaluate, example scripts are provided in `scripts/eval_scan.sh`, checkout commit `0412fc24a3b770e4d82e6d7064a8172f24c5fcd3`. Decode-generation may not work at this commit.

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
