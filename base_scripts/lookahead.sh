### Llama-3.2-1B Training Script
python train.py \
    --proj_name TestRun_TokenButler \
    --model_path meta-llama/Llama-3.2-1B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --rpj_train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "L3_1B_1k_L1.csv" \
    --wname L3_1B_1k_L1 \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --lookahead 1 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024

### Llama-3.2-1B Training Script
python train.py \
    --proj_name TestRun_TokenButler \
    --model_path meta-llama/Llama-3.2-1B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --rpj_train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "L3_1B_1k_L0.csv" \
    --wname L3_1B_1k_L0 \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --lookahead 0 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024

### Llama-3.2-1B Training Script
python train.py \
    --proj_name TestRun_TokenButler \
    --model_path meta-llama/Llama-3.2-1B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --rpj_train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "L3_1B_1k_L2.csv" \
    --wname L3_1B_1k_L2 \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --lookahead 2 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024
