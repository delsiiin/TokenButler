# python generate_test.py        \
#  --proj_name AllContextual_ICML        \
#  --no_wandb        \
#  --model_path meta-llama/Llama-3.2-1B        \
#  --sliding_window 16 \
#  --token_sparse_method fixed_70pc        \
#  --model_mode eval        \
#  --finetune_dataset c4_realnewslike        \
#  --train_subset_fac 4        \
#  --rpj_train_seqlen 1024        \
#  --eval_llm_mode snapkv        \
#  --lfunc MSE        \
#  --result_file DEBUG_L3_1B_2k.csv        \
#  --wname DEBUG_L3_1B_2k        \
#  --pred_lr 1e-3        \
#  --dDash 32        \
#  --intdim 1024  

#  python generate_test.py        \
#  --proj_name AllContextual_ICML        \
#  --no_wandb        \
#  --model_path meta-llama/Llama-3.2-1B        \
#  --sliding_window 16 \
#  --token_sparse_method fixed_70pc        \
#  --model_mode eval        \
#  --finetune_dataset c4_realnewslike        \
#  --train_subset_fac 4        \
#  --rpj_train_seqlen 1024        \
#  --eval_llm_mode quest        \
#  --lfunc MSE        \
#  --result_file DEBUG_L3_1B_2k.csv        \
#  --wname DEBUG_L3_1B_2k        \
#  --pred_lr 1e-3        \
#  --dDash 32        \
#  --intdim 1024  


#  python generate_test.py        \
#  --proj_name AllContextual_ICML        \
#  --no_wandb        \
#  --model_path meta-llama/Llama-3.2-1B        \
#  --sliding_window 16 \
#  --token_sparse_method fixed_70pc        \
#  --model_mode eval        \
#  --finetune_dataset c4_realnewslike        \
#  --train_subset_fac 4        \
#  --rpj_train_seqlen 1024        \
#  --eval_llm_mode oracle        \
#  --lfunc MSE        \
#  --result_file DEBUG_L3_1B_2k.csv        \
#  --wname DEBUG_L3_1B_2k        \
#  --pred_lr 1e-3        \
#  --dDash 32        \
#  --intdim 1024  

#  python generate_test.py        \
#  --proj_name AllContextual_ICML        \
#  --no_wandb        \
#  --model_path meta-llama/Llama-3.2-1B        \
#  --sliding_window 16 \
#  --token_sparse_method fixed_70pc        \
#  --model_mode eval        \
#  --finetune_dataset c4_realnewslike        \
#  --train_subset_fac 4        \
#  --rpj_train_seqlen 1024        \
#  --eval_llm_mode h2o_true        \
#  --lfunc MSE        \
#  --result_file DEBUG_L3_1B_2k.csv        \
#  --wname DEBUG_L3_1B_2k        \
#  --pred_lr 1e-3        \
#  --dDash 32        \
#  --intdim 1024  

