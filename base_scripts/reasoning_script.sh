python train.py     \
    --proj_name TokenButler_Reasoning \
    --architecture llama    \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_40pc     \
    --model_mode finetune   \
    --finetune_dataset c4_realnewslike     \
    --train_subset_fac 4      \
    --eval_llm_mode ExpPred     \
    --lfunc MSE     \
    --result_file "C4_L3_8B_R1_8K.csv"     \
    --wname C4_L3_8B_R1_8K.csv     \
    --pred_lr 1e-3     \
    --dDash 32     \
    --intdim 1024     \
    --eval_subset 1000      \
    --rpj_train_seqlen 8192 \
    --flash_attn   \
    --model_parallelism



python train.py     \
    --proj_name TokenButler_Reasoning \
    --architecture llama    \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_40pc     \
    --model_mode finetune   \
    --finetune_dataset c4_realnewslike     \
    --train_subset_fac 4      \
    --eval_llm_mode ExpPred     \
    --lfunc MSE     \
    --result_file "C4_L3_8B_R1_2K_BS4.csv"     \
    --wname C4_L3_8B_R1_2K_BS4     \
    --pred_lr 1e-3     \
    --dDash 32     \
    --intdim 1024     \
    --eval_subset 1000      \
    --rpj_train_seqlen 2048 \
    --model_parallelism


    # python train.py         --proj_name TokenButler_Reasoning     --architecture llama        --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B     --token_sparse_method fixed_40pc         --model_mode finetune       --finetune_dataset c4_realnewslike         --train_subset_fac 1          --eval_llm_mode ExpPred         --lfunc MSE         --result_file "C4_L3_8B_R1.csv"         --wname C4_L3_8B_R1.csv         --pred_lr 1e-3         --dDash 32         --intdim 1024         --eval_subset 1000          --rpj_train_seqlen 4096     --flash_attn       --model_parallelism