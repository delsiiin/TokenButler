#!/bin/bash


#### Llama 3.2 1B
# bash eval_scan.sh L3_1B_2k.csv L3_1B_2k ExpPred meta-llama/Llama-3.2-1B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k.csv L3_1B_2k oracle meta-llama/Llama-3.2-1B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k.csv L3_1B_2k h2o_true meta-llama/Llama-3.2-1B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k.csv L3_1B_2k quest meta-llama/Llama-3.2-1B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k.csv L3_1B_2k snapkv meta-llama/Llama-3.2-1B 1024 32 <PATH_TO_CHECKPOINT>

#### Llama 3.2 1B 1% predictor size
# bash eval_scan.sh L3_1B_2k_1PC.csv L3_1B_2k_1PC ExpPred meta-llama/Llama-3.2-1B 512 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k_1PC.csv L3_1B_2k_1PC oracle meta-llama/Llama-3.2-1B 512 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k_1PC.csv L3_1B_2k_1PC h2o_true meta-llama/Llama-3.2-1B 512 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k_1PC.csv L3_1B_2k_1PC quest meta-llama/Llama-3.2-1B 512 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_1B_2k_1PC.csv L3_1B_2k_1PC snapkv meta-llama/Llama-3.2-1B 512 16 <PATH_TO_CHECKPOINT>

#### Llama 3.2 3B
# bash eval_scan.sh L3_3B_2k.csv L3_3B_2k ExpPred meta-llama/Llama-3.2-3B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k.csv L3_3B_2k oracle meta-llama/Llama-3.2-3B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k.csv L3_3B_2k h2o_true meta-llama/Llama-3.2-3B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k.csv L3_3B_2k quest meta-llama/Llama-3.2-3B 1024 32 <PATH_TO_CHECKPOINT>

#### Llama 3.2 3B 1% predictor size
# bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC ExpPred meta-llama/Llama-3.2-3B 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC oracle meta-llama/Llama-3.2-3B 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC h2o_true meta-llama/Llama-3.2-3B 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC quest meta-llama/Llama-3.2-3B 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC snapkv meta-llama/Llama-3.2-3B 1024 16 <PATH_TO_CHECKPOINT>

#### Llama 3.1 8B
# bash eval_scan.sh L3_8B_1k.csv L3_8B_1k ExpPred meta-llama/Llama-3.1-8B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_8B_1k.csv L3_8B_1k oracle meta-llama/Llama-3.1-8B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_8B_1k.csv L3_8B_1k h2o_true meta-llama/Llama-3.1-8B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_8B_1k.csv L3_8B_1k quest meta-llama/Llama-3.1-8B 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L3_8B_1k.csv L3_8B_1k snapkv meta-llama/Llama-3.1-8B 1024 32 <PATH_TO_CHECKPOINT>

#### Llama 2 7B
# bash eval_scan.sh L2_7B_2k.csv L2_7B_2k ExpPred meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k.csv L2_7B_2k oracle meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k.csv L2_7B_2k h2o_true meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k.csv L2_7B_2k quest meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k.csv L2_7B_2k snapkv meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>

#### Mistral7B
# bash eval_scan.sh M7B_1k.csv M7B_1k ExpPred mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh M7B_1k_oracle.csv M7B_1k_oracle oracle mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh M7B_1k_h2o_true.csv M7B_1k_h2o_true h2o_true mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh M7B_1k_quest.csv M7B_1k_quest quest mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh M7B_1k_streamingLLM.csv M7B_1k_streamingLLM streamingLLM mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh M7B_1k.csv M7B_1k snapkv mistralai/Mistral-7B-v0.1 1024 32 <PATH_TO_CHECKPOINT>

#### Phi 3.5 mini instruct
# bash eval_scan.sh P35mini_1k_1PC.csv P35mini_1k_1PC ExpPred microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P35mini_1k_1PC_oracle.csv P35mini_1k_1PC_oracle oracle microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P35mini_1k_1PC_h2o_true.csv P35mini_1k_1PC_h2o_true h2o_true microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P35mini_1k_1PC_quest.csv P35mini_1k_1PC_quest quest microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P35mini_1k_1PC_streamingLLM.csv P35mini_1k_1PC_streamingLLM streamingLLM microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P35mini_1k_1PC.csv P35mini_1k_1PC snapkv microsoft/Phi-3.5-mini-instruct 1024 16 <PATH_TO_CHECKPOINT>

#### Phi 3 mini 4k instruct
# bash eval_scan.sh P3mini_1k_1PC.csv P3mini_1k_1PC ExpPred microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P3mini_1k_1PC_oracle.csv P3mini_1k_1PC_oracle oracle microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P3mini_1k_1PC_h2o_true.csv P3mini_1k_1PC_h2o_true h2o_true microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P3mini_1k_1PC_quest.csv P3mini_1k_1PC_quest quest microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P3mini_1k_1PC_streamingLLM.csv P3mini_1k_1PC_streamingLLM streamingLLM microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh P3mini_1k_1PC.csv P3mini_1k_1PC snapkv microsoft/Phi-3-mini-4k-instruct 1024 16 <PATH_TO_CHECKPOINT>


# bash eval_scan.sh L2_7B_2k_ABL.csv L2_7B_2k_ABL oracle meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k_ABL_init.csv L2_7B_2k_ABL_init init_oracle meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>
# bash eval_scan.sh L2_7B_2k_ABL_lookahead.csv L2_7B_2k_ABL_lookahead lookahead_oracle meta-llama/Llama-2-7b-hf 1024 32 <PATH_TO_CHECKPOINT>

# Positional arguments
result_file=$1
wname=$2
eval_llm_mode=$3
model_path=$4
intdim=$5
dDash=$6
model_load_path=$7

# Hardcoded values
task_list="winogrande,hellaswag,piqa,arc_easy"
eval_subset=1000


# Loop over token sparse methods
for perc in {10..60..10}; do
    token_sparse_method="fixed_${perc}pc"

    # Build the command dynamically
    cmd="python main.py \
        --proj_name AllContextual_ICML \
        --no_wandb \
        --model_path ${model_path} \
        --token_sparse_method ${token_sparse_method} \
        --model_mode eval \
        --finetune_dataset c4_realnewslike \
        --train_subset_fac 4 \
        --train_seqlen 1024 \
        --eval_llm_mode ${eval_llm_mode} \
        --lfunc MSE \
        --result_file ${result_file} \
        --wname ${wname} \
        --pred_lr 1e-3 \
        --sliding_window 4 \
        --dDash ${dDash} \
        --intdim ${intdim} \
        --task_list '${task_list}' \
        --eval_subset ${eval_subset} \
        --eval_wk2_seqlen 1024 --num_tok_per_page 4" \
        

    # Include --model_load_path if it is not None
    if [ "$model_load_path" != "None" ]; then
        cmd="${cmd} --model_load_path \"${model_load_path}\""
    fi

    # # if [ "$eval_llm_mode" != "h2o_true" ]; then
    cmd="${cmd} --do_downstream_eval"
    # # fi

    # if model_path contains "mistral", add --architecture mistral
    if [[ $model_path == *"mistral"* ]]; then
        cmd="${cmd} --architecture mistral"
    fi

    # if model_path contains "mistral", add --architecture mistral
    if [[ $model_path == *"Qwen"* ]]; then
        cmd="${cmd} --architecture qwen"
    fi

    # if model_path contains "mistral", add --architecture mistral
    if [[ $model_path == *"Phi-3"* ]]; then
        cmd="${cmd} --architecture phi3"
    fi

    # Execute the command
    echo "$cmd"
    eval $cmd

done