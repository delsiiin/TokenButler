

python3 train.py \
    --proj_name AllContextualNew \
    --model_path togethercomputer/LLaMA-2-7B-32K \
    --token_sparse_method fixed_10pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "predictor.csv" \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/togethercomputer_LLaMA-2-7B-32K_False_llama_qk_128_4_redpajama_0.5_True_False_finetune_None___tokens_, _heads___0.001_16_False_True_False_False_1000_10_512_2_2/False_fixed_40pc_ble_bnhll_ExpPred_AllContextualLarge_1000_3_512_512_512_1_32_4_16_4_False_MSE_True_False_False_True_testing.csv_MSE_RPJ_AttenPred_PInit_True_0.3875000000000002__best.pt" \
    --wname predictor \
    --predictor_type ble_bnhll \
    --add_attn \
    --olayer 2 \
    --max_norm 10 \
    --no_mask_loss \
    --pred_lr 1e-3 \
    --l0_output \
    --dDash 16 \
    --ilayer 2 \
    --intdim 512 \
    --no_wandb \
    --do_downstream_eval --task_list "winogrande,squadv2" \
    --eval_subset 2500


python3 train.py \
    --proj_name AllContextualNew \
    --model_path togethercomputer/LLaMA-2-7B-32K \
    --token_sparse_method fixed_20pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "predictor.csv" \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/togethercomputer_LLaMA-2-7B-32K_False_llama_qk_128_4_redpajama_0.5_True_False_finetune_None___tokens_, _heads___0.001_16_False_True_False_False_1000_10_512_2_2/False_fixed_40pc_ble_bnhll_ExpPred_AllContextualLarge_1000_3_512_512_512_1_32_4_16_4_False_MSE_True_False_False_True_testing.csv_MSE_RPJ_AttenPred_PInit_True_0.3875000000000002__best.pt" \
    --wname predictor \
    --predictor_type ble_bnhll \
    --add_attn \
    --olayer 2 \
    --max_norm 10 \
    --no_mask_loss \
    --pred_lr 1e-3 \
    --l0_output \
    --dDash 16 \
    --ilayer 2 \
    --intdim 512 \
    --no_wandb \
    --do_downstream_eval --task_list "winogrande,squadv2" \
    --eval_subset 2500


python3 train.py \
    --proj_name AllContextualNew \
    --model_path togethercomputer/LLaMA-2-7B-32K \
    --token_sparse_method fixed_30pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "predictor.csv" \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/togethercomputer_LLaMA-2-7B-32K_False_llama_qk_128_4_redpajama_0.5_True_False_finetune_None___tokens_, _heads___0.001_16_False_True_False_False_1000_10_512_2_2/False_fixed_40pc_ble_bnhll_ExpPred_AllContextualLarge_1000_3_512_512_512_1_32_4_16_4_False_MSE_True_False_False_True_testing.csv_MSE_RPJ_AttenPred_PInit_True_0.3875000000000002__best.pt" \
    --wname predictor \
    --predictor_type ble_bnhll \
    --add_attn \
    --olayer 2 \
    --max_norm 10 \
    --no_mask_loss \
    --pred_lr 1e-3 \
    --l0_output \
    --dDash 16 \
    --ilayer 2 \
    --intdim 512 \
    --no_wandb \
    --do_downstream_eval --task_list "winogrande,squadv2" \
    --eval_subset 2500


python3 train.py \
    --proj_name AllContextualNew \
    --model_path togethercomputer/LLaMA-2-7B-32K \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "predictor.csv" \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/togethercomputer_LLaMA-2-7B-32K_False_llama_qk_128_4_redpajama_0.5_True_False_finetune_None___tokens_, _heads___0.001_16_False_True_False_False_1000_10_512_2_2/False_fixed_40pc_ble_bnhll_ExpPred_AllContextualLarge_1000_3_512_512_512_1_32_4_16_4_False_MSE_True_False_False_True_testing.csv_MSE_RPJ_AttenPred_PInit_True_0.3875000000000002__best.pt" \
    --wname predictor \
    --predictor_type ble_bnhll \
    --add_attn \
    --olayer 2 \
    --max_norm 10 \
    --no_mask_loss \
    --pred_lr 1e-3 \
    --l0_output \
    --dDash 16 \
    --ilayer 2 \
    --intdim 512 \
    --no_wandb \
    --do_downstream_eval --task_list "winogrande,squadv2" \
    --eval_subset 2500


python3 train.py \
    --proj_name AllContextualNew \
    --model_path togethercomputer/LLaMA-2-7B-32K \
    --token_sparse_method fixed_70pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --eval_llm_mode ExpPred \
    --lfunc MSE \
    --result_file "predictor.csv" \
    --model_load_path "/home/ya255/projects/all_contextual/expt_model/togethercomputer_LLaMA-2-7B-32K_False_llama_qk_128_4_redpajama_0.5_True_False_finetune_None___tokens_, _heads___0.001_16_False_True_False_False_1000_10_512_2_2/False_fixed_40pc_ble_bnhll_ExpPred_AllContextualLarge_1000_3_512_512_512_1_32_4_16_4_False_MSE_True_False_False_True_testing.csv_MSE_RPJ_AttenPred_PInit_True_0.3875000000000002__best.pt" \
    --wname predictor \
    --predictor_type ble_bnhll \
    --add_attn \
    --olayer 2 \
    --max_norm 10 \
    --no_mask_loss \
    --pred_lr 1e-3 \
    --l0_output \
    --dDash 16 \
    --ilayer 2 \
    --intdim 512 \
    --no_wandb \
    --do_downstream_eval --task_list "winogrande,squadv2" \
    --eval_subset 2500
