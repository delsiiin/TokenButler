python reasoning_eval.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_30pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file C4_L3_8B_R1_1K.csv \
    --wname C4_L3_8B_R1_1K \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --rpj_train_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/42_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_False_llama_qk_128_4_c4_realnewslike_0.5_False_False_finetune_None_None_200_False_False_0_None_False_False_1_False_False_False_False_4_8/2_0.001_32_None_False_200_20_1024_fixed_40pc_ExpPred_TokenButler_Reasoning_1000_4_1024_512_1_32_4_16_4_MSE_False_False_C4_L3_8B_R1_1K.csv_C4_L3_8B_R1_1K_False_True_0.3875000000000002_20250304-154324.pt" \
    --do_downstream_eval \
    --eval_subset 1 \
    --model_parallelism \
    --task_list "aime24_nofigures"



# python train.py     \
#     --proj_name TokenButler_Reasoning \
#     --architecture llama    \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --token_sparse_method fixed_40pc     \
#     --model_mode finetune   \
#     --finetune_dataset c4_realnewslike     \
#     --train_subset_fac 4      \
#     --eval_llm_mode ExpPred     \
#     --lfunc MSE     \
#     --result_file "C4_L3_8B_R1_8K.csv"     \
#     --wname C4_L3_8B_R1_8K.csv     \
#     --pred_lr 1e-3     \
#     --dDash 32     \
#     --intdim 1024     \
#     --eval_subset 1000      \
#     --rpj_train_seqlen 8192 \
#     --flash_attn   \
#     --model_parallelism



# python train.py     \
#     --proj_name TokenButler_Reasoning \
#     --architecture llama    \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --token_sparse_method fixed_40pc     \
#     --model_mode finetune   \
#     --finetune_dataset c4_realnewslike     \
#     --train_subset_fac 4      \
#     --eval_llm_mode ExpPred     \
#     --lfunc MSE     \
#     --result_file "C4_L3_8B_R1_2K_BS4.csv"     \
#     --wname C4_L3_8B_R1_2K_BS4     \
#     --pred_lr 1e-3     \
#     --dDash 32     \
#     --intdim 1024     \
#     --eval_subset 1000      \
#     --rpj_train_seqlen 2048 \
#     --model_parallelism


# python train.py     \
#     --proj_name TokenButler_Reasoning \
#     --architecture llama    \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --token_sparse_method fixed_40pc     \
#     --model_mode finetune   \
#     --finetune_dataset c4_realnewslike     \
#     --train_subset_fac 4      \
#     --eval_llm_mode ExpPred     \
#     --lfunc MSE     \
#     --result_file "C4_L3_8B_R1_1K.csv"     \
#     --wname C4_L3_8B_R1_1K     \
#     --pred_lr 1e-3     \
#     --dDash 32     \
#     --intdim 1024     \
#     --eval_subset 1000      \
#     --rpj_train_seqlen 1024 

#     # python train.py         --proj_name TokenButler_Reasoning     --architecture llama        --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B     --token_sparse_method fixed_40pc         --model_mode finetune       --finetune_dataset c4_realnewslike         --train_subset_fac 1          --eval_llm_mode ExpPred         --lfunc MSE         --result_file "C4_L3_8B_R1.csv"         --wname C4_L3_8B_R1.csv         --pred_lr 1e-3         --dDash 32         --intdim 1024         --eval_subset 1000          --rpj_train_seqlen 4096     --flash_attn       --model_parallelism