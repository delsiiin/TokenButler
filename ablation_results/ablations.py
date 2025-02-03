import json
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from datasets import load_dataset
from functools import partial
import gc
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from random import seed, sample

from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import lm_eval
from collections import Counter
import math
import json
import pprint
import csv
from torch.nn import CrossEntropyLoss

from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import get_cosine_schedule_with_warmup

from huggingface_hub import login

from utils import FlattenedDataset, plot_thresholds, graph_headtok_pos_affinity, plot_and_save_head_agreement, compute_head_agreement_all_examples, plot_decode_jsd_violin, compute_layer_jsd
from utils import compute_layer_percentage_match_vectorized, plot_decode_percdrift_vectorized
from utils import compute_rank_agreement_all_examples, plot_and_save_rank_agreement, plot_decode_drift_trajectory
### New imports
# from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN
from datasets import load_from_disk
# from datasets import set_seed

from torch.utils.data import DataLoader
from scipy.stats import kendalltau, spearmanr
import pandas as pd
from torch.cuda.amp import autocast
import dotenv
import wandb
import re
scaler = GradScaler('cuda', enabled=True)
global dowandb
dotenv.load_dotenv()
hftoken = os.getenv("HFTOKEN")
# login(token=hftoken)

torch.backends.cuda.enable_flash_sdp(True)

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*\'\[\]]', '_', name)

def args_to_name(args, timestamp=True):
    args_dict = vars(args)
    args_dict = args_dict.copy()
    # remove longbench_datasets, task_list from args_dict
    args_dict.pop("longbench_datasets", None)
    args_dict.pop("task_list", None)
    model_descr = list(args_dict.values())
    # Split the model description into two parts
    split_point = len(model_descr) // 2
    folder_part = model_descr[:split_point]
    file_part = model_descr[split_point:]
    # Create a sanitized folder name from the first part
    folder_name = "_".join([str(elem) for elem in folder_part])
    folder_name = sanitize_filename(folder_name)
    # Create a sanitized file name from the second part
    file_name = "_".join([str(elem) for elem in file_part])
    file_name = sanitize_filename(file_name)
    # Add timestamp to file name
    if timestamp:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = file_name + "_" + timestamp
    return folder_name, file_name

def save_model(args, model, note=None):
    if note is None:
        timestamp = True
    else:
        timestamp = False
    passargs = args
    passargs.model_resume_path = None

    folder_name, file_name = args_to_name(passargs, timestamp)
    # Final path
    folder_path = "expt_model/" + folder_name
    if not os.path.exists("expt_model"):
        os.makedirs("expt_model")
    os.makedirs(folder_path, exist_ok=True)
    # Complete save path
    model_savepath_name = os.path.join(folder_path, file_name)
    if note is not None:
        model_savepath_name += "_" + note
    print(f"Model will be saved at: {model_savepath_name}")
    model_producer_layer = get_producer_layers(model)
    torch.save([layer_p.state_dict() for layer_p in model_producer_layer], model_savepath_name + ".pt")

def save_checkpoint(args, model, optimizer, scheduler, step, epoch, note=None):
    """
    Saves a model checkpoint with detailed experimental information in the name.
    
    Args:
        args: Experimental arguments for generating the checkpoint name.
        model: The model to save.
        optimizer: The optimizer state to save.
        scheduler: The scheduler state to save.
        step: Current training step.
        epoch: Current epoch.
        note: Additional note to append to the checkpoint file name.
    """
    # Generate folder and file name based on args
    timestamp = note is None
    
    passargs = args
    passargs.model_resume_path = None
    folder_name, file_name = args_to_name(passargs, timestamp)

    # Create folder path for checkpoints
    folder_path = os.path.join("checkpoints", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Append step and epoch to the file name for clarity
    checkpoint_file_name = f"{file_name[-40:]}"
    model_producer_layer = get_producer_layers(model)
    checkpoint_path = os.path.join(folder_path, f"{checkpoint_file_name}.pt")

    # Save the checkpoint
    torch.save({
        'wandb_step': wandb.run.step,
        'step': step,
        'epoch': epoch,
        'model_state_dict': [layer_p.state_dict() for layer_p in model_producer_layer],
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")


def get_producer_layers(model):
    """
    Traverses the model to find the producer layer (layer_idx=0).cc
    """
    producer_modules = []
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental") and module.layer_idx == 0:
            producer_modules.append(module)
    return producer_modules
    

def set_inference_mode(model, mode: bool):
    """
    Sets the inference_mode flag for all LlamaAttentionExperimental modules in the model.
    
    Args:
        model: The neural network model.
        mode (bool): The mode to set (True for inference, False for training).
    """
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.inference_mode = mode

def run_long_bench_evaluation(model, tokenizer, args):
    """
    Runs the LongBench evaluation on specified datasets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model_name = args.model_path.lower()
    model_type = args.model_path.split("/")[-1].split('_')[0]
        
    if not model_type in MODEL2MAXLEN:
        raise ValueError(f"Model {model_type} not supported")
    
    max_length = MODEL2MAXLEN.get(model_type, 2048)
    print(f"Running LongBench evaluation on model: {model_name}")
    print(f"Max length: {max_length}")
    datasets = args.longbench_datasets
    dataset2prompt = DATASET2PROMPT
    dataset2maxlen = DATASET2MAXLEN

    results = {}
    for dataset in datasets:
        print(f"Evaluating dataset: {dataset}")
        start_time = time.time()
        if args.eval_subset is None:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test[:20%]')
        # only take a subset of the data
        prompt_format = dataset2prompt.get(dataset, "{text}")
        max_gen = dataset2maxlen.get(dataset, 256)
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for dataset {dataset}: {elapsed_time/60} minutes")

        # calculate score
        predictions, answers, lengths = [], [], []
        all_classes = None
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            if "length" in pred:
                lengths.append(pred["length"])
            all_classes = pred.get("all_classes", None)
        score = scorer(dataset, predictions, answers, all_classes)
        print(f"Dataset: {dataset}")
        print(f"Score: {score}")
        results[dataset] = score
    if dowandb:
        for dataset in results:
            wandb.log({
                f"{dataset}_longbench_score": results[dataset]
            })
    return results

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def build_chat(tokenizer, prompt, model_name):
    # Copy from KIVI
    if "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        except ImportError:
            pass  # FastChat not installed
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm.tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # Truncate to fit max_length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # Adjust if necessary
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_ids.input_ids.shape[-1]
        with autocast():
            if dataset == "samsum":
                output = model.generate(
                    **input_ids,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input_ids,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj.get("all_classes", None), "length": json_obj.get("length", None)})
    return preds

def run_lm_eval_zero_shot(model, tokenizer, batch_size=1, max_length=512, task_list=["arc_easy", "hellaswag"], limit=None, flash_attn=False, train_headpredictor=False):
    for module in model.modules():
        # Here, we should take care to set head predictor flash attention mode appropriately
        module.flash_attn = False
    model.seqlen = max_length
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    task_manager = lm_eval.tasks.TaskManager()
    print(f"Evaluating on tasks: {task_list}")
    # autocast
    with autocast():
        with torch.no_grad():
            results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=task_list,
                task_manager=task_manager,
                log_samples=False,
                limit=limit
            )
    res = make_table(results)
    print(res)
    for module in model.modules():
        module.flash_attn = flash_attn
        if hasattr(module, 'is_head_predictor'):
            if train_headpredictor == False:
                module.flash_attn = False
    return results['results']

def evaluate_wikitext2(model, tokenizer, args, testenc=None, traintime_subset=False):
    """
    Evaluates the model on the Wikitext-2 dataset using perplexity.

    Args:
        model: The neural network model.
        tokenizer: The tokenizer corresponding to the model.
        args: Arguments containing configuration details (e.g., eval_subset, eval_wk2_seqlen).
        traintime_subset (bool): Whether to use a smaller subset for evaluation.

    Returns:
        The perplexity of the model on the test dataset.
    """
    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    # Use a subset of data if traintime_subset or args.eval_subset is specified
    if traintime_subset:
        dataset = dataset.select(range(1000))
    elif args.eval_subset is not None:
        dataset = dataset.select(range(args.eval_subset))

    # Concatenate all text entries
    concatenated_text = "\n\n".join(dataset["text"])
    tokenized_output = tokenizer(concatenated_text, truncation=False)

    # Split tokenized input into chunks of max_seq_len
    input_ids = torch.tensor(tokenized_output["input_ids"]).to(model.device)
    max_seq_len = args.eval_wk2_seqlen
    num_chunks = (len(input_ids) + max_seq_len - 1) // max_seq_len  # Ceiling division
    input_chunks = input_ids.split(max_seq_len)

    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.calc_hitrates = True
            module.calibrate_thresholds = args.calibrate_thresholds
            module.test_with_thresholds = args.test_with_thresholds

    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    progress_bar = tqdm.tqdm(range(num_chunks), desc="Evaluating Wikitext-2")
    losses = []
    avg_tok_hit_rate = []
    avg_head_hit_rate = []
    threshold_mean = []
    true_threshmean = []
    effective_sparsity_list = []
    head_tokpos_affinity = {}
    decode_tokpos_affinity = {}
    head_act_magn = {}
    num_decode_test = 50

    for chunk in progress_bar:
        batch = input_chunks[chunk].unsqueeze(0).to(model.device)
        if batch.size(1) < 2:  # Skip sequences too short for meaningful evaluation
            continue
        with autocast():
            with torch.no_grad():
                outputs = model(batch, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                target = batch[:, 1:]

                loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
                total_loss += loss.item() * target.numel()
                losses.append(loss.item())
                total_tokens += target.numel()
            tok_hit_rates, tok_mean_rank_corr, tok_max_rank_corr = [], [], []
            head_hit_rates, head_mean_rank_corr, head_max_rank_corr = [], [], []
            layeridx = 0
            for module in model.modules():
                if module.__class__.__name__.endswith("AttentionExperimental"):
                    ## Store the last next-word prediction if seq-len is 1024
                    if batch.size(1) == 1024:
                        if module.layer_idx not in head_tokpos_affinity:
                            head_tokpos_affinity[module.layer_idx] = []
                        if module.layer_idx not in decode_tokpos_affinity:
                            decode_tokpos_affinity[module.layer_idx] = []
                        if module.layer_idx not in head_act_magn:
                            head_act_magn[module.layer_idx] = []
                        head_tokpos_affinity[module.layer_idx].append(module.attn_weights[:, :, -1, :].squeeze().cpu().detach())
                        decode_tokpos_affinity[module.layer_idx].append(torch.softmax(module.attn_unadj_weights[:, :, -num_decode_test:, :-num_decode_test], dim=-1).squeeze().cpu().detach())
                        head_act_magn[module.layer_idx].append(module.attn_head_weights.squeeze().cpu().detach())

                if module.__class__.__name__.endswith("AttentionExperimental") and module.layer_idx != 0:
                    try:
                        tok_hit_rates.append(module.tok_hit_acc)
                        tok_mean_rank_corr.append(module.tok_mean_rank_corr)
                        tok_max_rank_corr.append(module.tok_max_rank_corr)
                        head_hit_rates.append(module.head_hit_acc)
                        head_mean_rank_corr.append(module.head_mean_rank_corr)
                        head_max_rank_corr.append(module.head_max_rank_corr)
                        ### Threshold variance investigation
                        if hasattr(module, 'threshmean'):
                            threshold_mean.append(module.threshmean.cpu().detach())
                            true_threshmean.append(module.true_threshmean.cpu().detach())
                        if hasattr(module, 'effective_sparsity'):
                            effective_sparsity_list.append(module.effective_sparsity)
                        ### Threshold variance investigation
                        layeridx += 1
                    except:
                        layeridx += 1
                        continue
            tok_hit_rates = torch.tensor(tok_hit_rates).mean().item()
            tok_mean_rank_corr = torch.tensor(tok_mean_rank_corr).mean().item()
            tok_max_rank_corr = torch.tensor(tok_max_rank_corr).mean().item()
            head_hit_rates = torch.tensor(head_hit_rates).mean().item()
            head_mean_rank_corr = torch.tensor(head_mean_rank_corr).mean().item()
            head_max_rank_corr = torch.tensor(head_max_rank_corr).mean().item()

            avg_tok_hit_rate.append(tok_hit_rates)
            avg_head_hit_rate.append(head_hit_rates)

    for keyx in head_tokpos_affinity:
        head_tokpos_affinity[keyx] = torch.stack(head_tokpos_affinity[keyx])
    
    for keyx in decode_tokpos_affinity:
        decode_tokpos_affinity[keyx] = torch.stack(decode_tokpos_affinity[keyx])
    
    for keyx in head_act_magn:
        head_act_magn[keyx] = torch.stack(head_act_magn[keyx])



    # graph_headtok_pos_affinity(head_tokpos_affinity, args)

    rank_agreement = compute_rank_agreement_all_examples(head_tokpos_affinity, args)
    # plot_and_save_rank_agreement(rank_agreement, args)

    # agreement_values = compute_head_agreement_all_examples(head_tokpos_affinity)
    # plot_and_save_head_agreement(agreement_values, args)
    
    # layer_jsd = compute_layer_jsd(decode_tokpos_affinity)
    # plot_decode_jsd_violin(layer_jsd, args)

    # layer_match = compute_layer_percentage_match_vectorized(decode_tokpos_affinity, top_k=0.1)
    # plot_decode_percdrift_vectorized(layer_match, args)

    plot_decode_drift_trajectory(decode_tokpos_affinity, top_k=0.1, args=args)
    # import pdb; pdb.set_trace()
    exit()

    print(f"Average Token Hit Rate: {100*sum(avg_tok_hit_rate) / len(avg_tok_hit_rate)}%")
    print(f"Average Head Hit Rate: {100*sum(avg_head_hit_rate) / len(avg_head_hit_rate)}%")
    ### Threshold variance investigation
    if args.calibrate_thresholds:
        threshold_tensor = torch.stack([x for x in threshold_mean if x.size(-1)==1024]).view(-1, 31, 32, 1024)
        true_threshold_tensor = torch.stack([x for x in true_threshmean if x.size(-1)==1024]).view(-1, 31, 32, 1024)
        mean_threshold_postattn, mean_threshold_predpresm = plot_thresholds(threshold_tensor, true_threshold_tensor)
        print(mean_threshold_predpresm)
        print("Please Store Calibration Values Appropriately In threshold_calib_dict to enable testing with thresholding!")
    if args.test_with_thresholds:
        effective_sparsity_list = torch.tensor(effective_sparsity_list)
        mean_sparsity = effective_sparsity_list.mean().item()
        stddev_sparsity = effective_sparsity_list.std().item()
        print("You tried to use calibrated values for testing expected sparsity.")
        print("Mean Sparsity: ", mean_sparsity)
        print("Stddev Sparsity: ", stddev_sparsity)
    ### Threshold variance investigation
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexity evaluation completed: {perplexity.item()}")
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.calc_hitrates = False
    if torch.isnan(perplexity):
        import pdb; pdb.set_trace()
    return perplexity.item(), None


def decode_tokenized_input(input_ids, tokenizer):
    """
    Decodes tokenized input IDs back to text using the tokenizer.

    Args:
        input_ids (torch.Tensor): The tokenized input IDs.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        str: The decoded text.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return text


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')
    # add a seed
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Selected model')
    parser.add_argument('--task_list', type=lambda s: [item for item in s.split(',')], default=["arc_easy", "hellaswag"], help='Comma-separated list of tasks for evaluation')
    parser.add_argument('--offloading', action='store_true', help='Whether to use offloading')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral", "qwen", "glm", "phi3"])
    parser.add_argument('--channel', type=str, default="qk", choices=["q", "k", "qk"])
    parser.add_argument('--heavy_const', type=int, default=128, help='Heavy constant')
    parser.add_argument('--q_bits', type=int, default=4, help='Quantization bits')
    parser.add_argument('--finetune_dataset', type=str, default="wikitext", choices=["wikitext", "c4", "c4_realnewslike", "alpaca", "redpajama"], help='Dataset to use for fine-tuning')
    parser.add_argument('--head_sparsity_aggression', type=float, default=0.5)
    parser.add_argument('--do_downstream_eval', action='store_true', help='Whether to perform downstream evaluation.')
    # Added arguments for LongBench evaluation
    parser.add_argument('--do_longbench_eval', action='store_true', help='Whether to perform LongBench evaluation.')
    parser.add_argument('--longbench_datasets', type=lambda s: [item for item in s.split(',')], 
                        default=["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"],
                        help='Comma-separated list of datasets for LongBench evaluation')
    # Custom control arguments
    parser.add_argument('--model_mode', type=str, default="eval", choices=["eval", "finetune", "shadowllm"])
    parser.add_argument('--model_load_path', type=str, default=None, help='Path to load model')
    parser.add_argument('--model_resume_path', type=str, default=None, help='Path to resume training (includes optimizer, scheduler, and step states).')
    parser.add_argument('--save_interval', type=int, default=5000, help='Number of steps after which to save a checkpoint.')

    # Current focus 
    parser.add_argument('--calibrate_thresholds', action='store_true', help='Calibrate Per-Head Token Thresholding.')
    # Current focus 
    parser.add_argument('--randomize_init', action='store_true', help='Very Experimental! Tries to train predictor on RANDOMLY initialized transformer...')
    parser.add_argument('--test_with_thresholds', action='store_true', help='Test With Per-Head Token Thresholding, must have calibrated before!')
    parser.add_argument('--gfac', type=int, default=1)
    parser.add_argument('--immediate_train', action='store_true', help='Cosine without warmup for fast tests')
    parser.add_argument('--ssmize_predictor', action='store_true', help='SSM instead of Attn in predictor')
    parser.add_argument('--flash_attn', action='store_true', help='Use Flash Attention')
    parser.add_argument('--train_headpredictor', action='store_true', help='Train Head Predictor')
    parser.add_argument('--min_sparse_index', type=int, default=4, help="Num of Sink Tokens")
    parser.add_argument('--attn_reduce_factor', type=int, default=8, help="reduce factor for token predictor attention")
    parser.add_argument('--head_attn_reduce_factor', type=int, default=2, help="reduce factor for head predictor attention")
    parser.add_argument('--pred_lr', type=float, default=1e-3, help='Predictor learning rate')
    parser.add_argument('--dDash', type=int, default=16, help='Attn Red-dim')
    parser.add_argument('--skip_outlier', type=int, default=None, help='Skip backprop when task loss is outlier, stabilizes training. Not done on WK2, only RPJ.')

    parser.add_argument('--no_pred_causal_mask', action='store_true', help='Enable or disable causal mask application')
    parser.add_argument('--evalgap', type=int, default=1000, help='eval gap during training')
    parser.add_argument('--max_norm', type=int, default=20, help='Max Norm')
    parser.add_argument('--intdim', type=int, default=512, help='Int-Proc Dim')
    parser.add_argument('--token_sparse_method', type=str, default="progressive_5pc", help="LazyLLM, progressive_xpc, fixed_xpc...")
    parser.add_argument('--eval_llm_mode', type=str, default="TopSparse", help="oracle, lookahead_magnitude, lookahead_firstlayer_magnitude, predictor, h2o, streamingLLM")
    # Not primay focus
    parser.add_argument('--proj_name', type=str, default="AllContextual", help="Name for wandb project")

    parser.add_argument('--eval_subset', type=int, default=None)
    parser.add_argument('--train_subset_fac', type=int, default=None)
    parser.add_argument('--rpj_train_seqlen', type=int, default=512)
    parser.add_argument('--eval_wk2_seqlen', type=int, default=512)
    parser.add_argument('--grad_accum_steps', type=int, default=1)

    parser.add_argument('--producer_frequency', type=int, default=32, help="Gap of appearance for producer layer in model.")

    parser.add_argument('--group_factor', type=int, default=4, help='Group factor for H2O, reduces acc, like Quest num_tok_per_page')
    parser.add_argument('--num_tok_per_page', type=int, default=16, help='Number of tokens per page for Quest')
    parser.add_argument('--stream_llm_start_size', type=int, default=4, help='Num-sink tokens to keep for StreamingLLM')

    parser.add_argument('--lfunc', type=str, default="MSE", help="MSE, KLD,  JSD...")

    parser.add_argument('--no_wandb', action='store_true', help='Enable or disable wandb logging')
    parser.add_argument('--no_wikitext_eval', action='store_true', help='Whether to perform Wikitext evaluation.')
    
    parser.add_argument('--result_file', type=str, default="all_results.csv", help="Where to save results.")
    parser.add_argument('--wname', type=str, default=None, help="Name for wandb run")
    args = parser.parse_args()
    args.do_wikitext_eval = not args.no_wikitext_eval

    # dowandb is opposite of no_wandb
    dowandb = not args.no_wandb
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    assert args.model_mode != "finetune", "Fine-tuning not supported for ablation study."
    assert args.flash_attn == False, "Flash Attention not supported for ablation study."

    if dowandb:
        if args.wname is not None:
            wandb.init(project=args.proj_name, name=args.wname, config=args)
        else:
            wandb.init(project=args.proj_name, config=args)

    model_path = args.model_path
    channel_path = "config/" + model_path + ".json"
    if "Yarn-Llama" in model_path or "Llama-2-7b-hf" in model_path or "longchat-7b" in model_path or "togetherco" in model_path:
        channel_path = "config/" + "meta-llama/Llama-2-7b-hf" + ".json"

    testenc_wk2 = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    if "70b" in model_path:
        device_map = {"model.embed_tokens": 0, "model.norm": 7, "lm_head": 7}
        for i in range(80):
            device_map[f"model.layers.{i}"] = i // 10
    else:
        device_map = "cuda"
    if args.model_mode == "eval":
        kwargs = {"torch_dtype": torch.float16, "device_map": device_map}
    else:
        kwargs = {"torch_dtype": torch.float32, "device_map": device_map}

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, use_fast=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, use_auth_token=True, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, offload_folder=None, trust_remote_code=True, use_auth_token=True, **kwargs).cuda()
    if args.randomize_init:
        model = AutoModelForCausalLM.from_config(config).cuda()

        
    args.producer_frequency = config.num_hidden_layers

    try:
        with open(channel_path, "r") as f:
            channel_config = json.load(f)
    except:
        print(f"Channel config not found at {channel_path}")
        channel_config = None

    if args.architecture == "llama" and "Yarn-Llama" not in model_path:
        print("Running module replacement")
        if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
            if args.ssmize_predictor:
                raise NotImplementedError("SSMize deprecated for ExpPred. Ref to commit 053db60eaafac33611e86110f6110d8c8e4afe25 for implementation.")
            else:
                from modify_llama_ablations import convert_kvcache_experimental, convert_llama_channel_config_experimental
                from modify_llama_ablations import LlamaAttentionExperimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Llama Ablation Study.")
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
        if channel_config is not None:
            model = convert_llama_channel_config_experimental(model, channel_config, args.channel)

    elif args.architecture == "mistral":
        print("Running Mistral module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_mistral_ablations import convert_kvcache_experimental
            from modify_mistral_ablations import MistralAttentionExperimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Mistral yet")

        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    elif args.architecture == "phi3":
        print("Running Phi3 module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_phi3_ablations import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Phi3 yet")

        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    elif args.architecture == "qwen":
        print("Running Qwen module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_qwen_ablations import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Qwen yet")
    
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported")
    # else:
        # raise NotImplementedError(f"Architecture {args.architecture} not supported")

    token_sparsity_list = []
    for module in model.modules():
        # If module's class name ends with AttentionExperimental
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.eval_llm_mode = args.eval_llm_mode
            module.gfac = args.gfac
            module.token_sparse_method = args.token_sparse_method
            module.set_token_sparsity()
            token_sparsity_list.append(1. - module.sparse_aggression)
            module.stream_llm_start_size = args.stream_llm_start_size
            module.num_tok_per_page = args.num_tok_per_page
            module.group_factor = args.group_factor
            module.lfunc = args.lfunc
            module.producer_frequency = args.producer_frequency
            module.dDash = args.dDash
            module.attn_reduce_factor = args.attn_reduce_factor
            module.head_attn_reduce_factor = args.head_attn_reduce_factor
            module.intdim = args.intdim
            module.flash_attn = args.flash_attn
            module.train_headpredictor = args.train_headpredictor
            module.min_sparse_index = args.min_sparse_index
            module.num_layers_pred = module.producer_frequency  # Literally the gap is the number of layers to predict for.

            if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
                if module.layer_idx % args.producer_frequency == 0:
                    module.update_predictor()

    num_params = 0
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            # check if attribute exists
            if hasattr(module, 'sparse_token_predictor'):
                num_params += sum(p.numel() for p in module.sparse_token_predictor.parameters())
                
    print("\n\n\n Number of Parameters in Sparse Token Predictor (in millions): ", num_params / 1e6, "\n\n\n")
    try:
        print("\n\n\n Expected Sparsity For Method: ", sum(token_sparsity_list) / len(token_sparsity_list), "\n\n\n")
    except:
        pass
        # import pdb; pdb.set_trace()
    # avg_token_sparsity = sum(token_sparsity_list) / len(token_sparsity_list)
    args.net_sparsity = 0
    if dowandb:
        wandb.log({
            "avg_token_sparsity": avg_token_sparsity
        })
    model = model.cuda()

    result_save_folder = "ablation_results"
    if args.model_mode == "eval":
        if args.model_load_path is not None:
            model_producer_layers = get_producer_layers(model)
            producer_layer_weights = torch.load(args.model_load_path)
            for idx, producer_layer_weight in enumerate(producer_layer_weights):
                model_producer_layers[idx].load_state_dict(producer_layer_weight, strict=False)
            
        set_inference_mode(model, True)

        model.eval()

        torch.cuda.empty_cache()
        gc.collect()
        perplexity = 0
        if args.do_wikitext_eval:
            perplexity, eval_mask = evaluate_wikitext2(model=model, tokenizer=tokenizer, args=args, testenc=testenc_wk2, traintime_subset=False)
            print(f"Perplexity on Wikitext-2: {perplexity:.2f}")
        if args.do_downstream_eval:
            print("Evaluating on additional tasks...")
            task_results = run_lm_eval_zero_shot(model, tokenizer, task_list=args.task_list, limit=args.eval_subset, flash_attn=args.flash_attn, train_headpredictor=args.train_headpredictor)
        else:
            task_results = {}
        
        print(f"Perplexity on Wikitext-2: {perplexity:.2f}")
        print(f"Task evaluation results: {json.dumps(task_results, indent=4)}")
        if args.do_longbench_eval:
            longbench_results = run_long_bench_evaluation(model, tokenizer, args)
            print(f"LongBench evaluation results: {json.dumps(longbench_results, indent=4)}")
            longbench_scores = {}
            for dataset in longbench_results:
                longbench_scores[dataset] = longbench_results[dataset]
                print(f"Dataset: {dataset}, Score: {longbench_scores[dataset]}")

        torch.cuda.empty_cache()
        gc.collect()

        effective_sparsities = []
        for module in model.modules():
            if module.__class__.__name__.endswith("AttentionExperimental"):
                if module.effective_sparsity is not None:
                    effective_sparsities.append(module.effective_sparsity)
        
        print(f"Effective average token sparsity: {sum(effective_sparsities) / len(effective_sparsities)}")
        
        args.true_token_sparsity = sum(effective_sparsities) / len(effective_sparsities)
        possible_keys = ["acc,none", "em,none", "f1,none", "exact,none", "best_f1,none", "HasAns_exact,none", "NoAns_exact,none"]

        task_accuracies = {}
        for task in task_results:
            for key in possible_keys:
                if key in task_results[task]:
                    cleaned_key = key.replace(",none", "")  # Remove ',none'
                    task_accuracies[f"{task}_{cleaned_key}"] = task_results[task][key]
                    print(f"Task: {task}, Key: {cleaned_key}, Value: {task_results[task][key]}")

        if not os.path.exists(f"{result_save_folder}/"):
            os.makedirs(f"{result_save_folder}/")
        with open(f"{result_save_folder}/" + args.result_file, mode='a') as results_file:
            results_writer = csv.writer(results_file)
            args_dict = vars(args)
            header = list(args_dict.keys())
            header.append("perplexity")
            header.extend(task_accuracies.keys())  # Add all task-specific keys
            # header.extend([f"{task}_acc" for task in task_accuracies.keys()])
            if args.do_longbench_eval:
                header.extend([f"{dataset}_longbench_score" for dataset in longbench_scores.keys()])
            results_writer.writerow(header)
            row = list(args_dict.values())
            row.append(perplexity)
            row.extend([task_accuracies[task] for task in task_accuracies.keys()])
            if args.do_longbench_eval:
                row.extend([longbench_scores[dataset] for dataset in longbench_scores.keys()])
            results_writer.writerow(row)
    elif args.model_mode == "shadowllm":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    if dowandb:
        wandb.finish()