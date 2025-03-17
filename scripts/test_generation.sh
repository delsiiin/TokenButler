python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.2-3B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_8B_2k.csv" \
    --no_wandb \
    --wname L3_8B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,triviaqa" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/42_meta-llama_Llama-3.2-3B_False_llama_qk_128_4_c4_realnewslike_0.5_True_False_finetune_None_None_5000_False_False_1_False_False_False_False_4_8_2/0.001_16_None_False_1000_20_1024_fixed_40pc_ExpPred_AllContextual_Jan9_1000_4_2048_1024_1_28_4_16_4_MSE_False_False_L3_3B_2k_1PC.csv_L3_3B_2k_1PC_True_0.38571428571428584_20250111-042334.pt" 

python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.2-3B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode oracle \
    --result_file "L3_8B_2k.csv" \
    --no_wandb \
    --wname L3_8B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,triviaqa" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TrainTokenButler_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_3B_2k.csv_L3_3B_2k_False_False_2000_False_redpajama_1024_1_1_20_0.001_1024/16_False_4_1000_ExpPred_fixed_40pc_True_False_0_None_False_False_4_8_2_16_1024_False_False_True_28_0.38571428571428584__best.pt" 


python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.1-8B \
    --architecture llama \
    --token_sparse_method fixed_1pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_8B_2k.csv" \
    --no_wandb \
    --wname L3_8B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,triviaqa" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024

python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.1-8B \
    --architecture llama \
    --token_sparse_method fixed_1pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode dense \
    --result_file "L3_8B_2k.csv" \
    --no_wandb \
    --wname L3_8B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,triviaqa" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024


python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.1-8B \
    --architecture llama \
    --token_sparse_method fixed_1pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode oracle \
    --result_file "L3_8B_2k.csv" \
    --no_wandb \
    --wname L3_8B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,triviaqa" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024
