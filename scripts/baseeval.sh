

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_10pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file TEST_L38B_ORACLE.csv \
    --wname TEST_L38B_ORACLE \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000



python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_30pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file TEST_L38B_ORACLE.csv \
    --wname TEST_L38B_ORACLE \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000




python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file TEST_L38B_ORACLE.csv \
    --wname TEST_L38B_ORACLE \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000




python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_70pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file TEST_L38B_ORACLE.csv \
    --wname TEST_L38B_ORACLE \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000




python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_90pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file TEST_L38B_ORACLE.csv \
    --wname TEST_L38B_ORACLE \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000

