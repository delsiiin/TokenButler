import json
import tqdm
import torch
from torch import nn
from predictor import PredictorDynamicCache
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

from typing import Any, Dict, List, Optional, Tuple
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import lm_eval
from collections import Counter
import math
import json
import pprint
import csv
from torch.nn import CrossEntropyLoss

from transformers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import get_cosine_schedule_with_warmup

from huggingface_hub import login

from utils import FlattenedDataset, plot_thresholds, sanitize_filename, args_to_name

from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN
from datasets import load_from_disk

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

try:
    hftoken = os.getenv("HFTOKEN")
    login(token=hftoken)
except:
    print("Warning: HF-Login may be needed!")
    pass

torch.backends.cuda.enable_flash_sdp(True)


def save_model(args, model, note=None):
    if note is None:
        timestamp = True
    else:
        timestamp = False
    passargs = args
    passargs.model_resume_path = None

    folder_name, file_name = args_to_name(passargs, timestamp)
    folder_path = "expt_model/" + folder_name
    if not os.path.exists("expt_model"):
        os.makedirs("expt_model")
    os.makedirs(folder_path, exist_ok=True)
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
    device = next(model.parameters()).device
    model.eval()
    model_name = args.model_path.lower()
    model_type = args.model_path.split("/")[-1].split('_')[0].lower()
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
            data = load_dataset('THUDM/LongBench', dataset, split='test[:5%]')
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
        
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt")
        context_length = input_ids.input_ids.shape[-1]
        embed_device = model.model.embed_tokens.weight.device
        with autocast():
            if dataset == "samsum":
                output = model.generate(
                    **input_ids.to(embed_device),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0].to("cpu")
            else:
                output = model.generate(
                    **input_ids.to(embed_device),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0].to("cpu")
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj.get("all_classes", None), "length": json_obj.get("length", None)})
    return preds

def patched_prepare_cache_for_generation(
    self, generation_config, model_kwargs: Dict, *args, **kwargs
):
    # Normally, huggingface tries to do: model_kwargs["past_key_values"] = DynamicCache()
    # We override it with ours:
    if "past_key_values" not in model_kwargs or model_kwargs["past_key_values"] is None:
        model_kwargs["past_key_values"] = PredictorDynamicCache()
    return model_kwargs

def run_lm_eval_zero_shot(model, tokenizer, batch_size=1, max_length=512, task_list=["arc_easy", "hellaswag"], limit=None, flash_attn=False, train_headpredictor=False):

    for module in model.modules():
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
                limit=limit,
            )
    res = make_table(results)
    print(res)
    for module in model.modules():
        module.flash_attn = flash_attn
        if hasattr(module, 'is_head_predictor'):
            if train_headpredictor == False:
                module.flash_attn = False
    return results['results']

def evaluate_wikitext2(model, tokenizer, args, testenc=None, traintime_subset=False, config=None):
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
    try:
        if traintime_subset:
            dataset = dataset.select(range(1000))
        elif args.eval_subset is not None:
            dataset = dataset.select(range(args.eval_subset))
    except:
        pass

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

    for chunk in progress_bar:
        batch = input_chunks[chunk].unsqueeze(0).to(model.device)
        if batch.size(1) < 2:
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
            try:
                head_hit_rates = torch.tensor(head_hit_rates).mean().item()
                head_mean_rank_corr = torch.tensor(head_mean_rank_corr).mean().item()
                head_max_rank_corr = torch.tensor(head_max_rank_corr).mean().item()
            except:
                head_hit_rates = 0
                head_mean_rank_corr = 0
                head_max_rank_corr = 0
            avg_tok_hit_rate.append(tok_hit_rates)
            avg_head_hit_rate.append(head_hit_rates)
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexity evaluation completed: {perplexity.item()}")
    print(f"Average Token Hit Rate: {100*sum(avg_tok_hit_rate) / len(avg_tok_hit_rate)}%")
    print(f"Average Head Hit Rate: {100*sum(avg_head_hit_rate) / len(avg_head_hit_rate)}%")
    model_name = args.model_path.split("/")[-1]
    ### Threshold variance investigation
    if args.calibrate_thresholds:
        # Get number of layers in model
        nlayers = config.num_hidden_layers
        nheads = config.num_attention_heads
        threshold_tensor = torch.stack([x for x in threshold_mean if x.size(-1)==1024]).view(-1, nlayers - 1, nheads, 1024)
        true_threshold_tensor = torch.stack([x for x in true_threshmean if x.size(-1)==1024]).view(-1, nlayers - 1, nheads, 1024)
        mean_threshold_postattn, mean_threshold_predpresm = plot_thresholds(threshold_tensor, true_threshold_tensor, model_name, args.token_sparse_method)
        # We can save the mean_threshold_predpresm
        # The reference to the dict should be named as model name and sparsity target
        # check if a directory called threshold_calibs exists
        if not os.path.exists("threshold_calibs"):
            os.makedirs("threshold_calibs")
        # Now, make a dir with model name if it doesnt exist
        if not os.path.exists(f"threshold_calibs/{model_name}"):
            os.makedirs(f"threshold_calibs/{model_name}")
        # now, for the sparsity target (args.sparse_aggression), save a pkl file with mean_threshold_predpresm
        with open(f"threshold_calibs/{model_name}/{args.token_sparse_method}.pkl", "wb") as f:
            torch.save(mean_threshold_predpresm, f)
        print(mean_threshold_predpresm)
        print("-"*20, " Calibration Threshold Saved, Exiting... ", "-"*20)
        exit(0)

    if args.test_with_thresholds:
        effective_sparsity_list = torch.tensor(effective_sparsity_list)
        mean_sparsity = effective_sparsity_list.mean().item()
        stddev_sparsity = effective_sparsity_list.std().item()
        if not os.path.exists("calib_info.csv"):
            with open("calib_info.csv", "w") as f:
                f.write("model_name,token_sparse_method,mean_sparsity,stddev_sparsity,perplexity\n")
        with open("calib_info.csv", "a") as f:
            f.write(f"{model_name},{args.token_sparse_method},{mean_sparsity},{stddev_sparsity},{perplexity}\n")

        print("You tried to use calibrated values for testing expected sparsity.")
        print("Mean Sparsity: ", mean_sparsity)
        print("Stddev Sparsity: ", stddev_sparsity)

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
    # Base Arguments
    parser.add_argument('--proj_name', type=str, default="AllContextual", help="Name for wandb project")
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--model_mode', type=str, default="eval", choices=["eval", "finetune"])
    parser.add_argument('--model_load_path', type=str, default=None, help='Path to load model')
    parser.add_argument('--model_resume_path', type=str, default=None, help='Path to resume training (includes optimizer, scheduler, and step states).')
    parser.add_argument('--save_interval', type=int, default=200, help='Number of steps after which to save a checkpoint.')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral", "qwen", "glm", "phi3"])
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Selected model')
    parser.add_argument('--result_file', type=str, default="all_results.csv", help="Where to save results.")
    parser.add_argument('--wname', type=str, default=None, help="Name for wandb run")
    parser.add_argument('--model_parallelism', action='store_true', help='Enable model parallelism')
    parser.add_argument('--no_wandb', action='store_true', help='Enable or disable wandb logging')
    parser.add_argument('--evalgap', type=int, default=200, help='eval gap during training')
    parser.add_argument('--flash_attn', action='store_true', help='Use Flash Attention')
    
    # Train related arguments
    parser.add_argument('--finetune_dataset', type=str, default="wikitext", choices=["wikitext", "c4", "c4_realnewslike", "alpaca", "redpajama"], help='Dataset to use for fine-tuning')
    parser.add_argument('--train_seqlen', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--train_subset_fac', type=int, default=None)
    parser.add_argument('--max_norm', type=int, default=20, help='Max Norm')
    parser.add_argument('--pred_lr', type=float, default=1e-3, help='Predictor learning rate')

    # Evaluation Related Arguments
    parser.add_argument('--eval_wk2_seqlen', type=int, default=512)
    parser.add_argument('--num_tok_per_page', type=int, default=16, help='Number of tokens per page for Quest')
    parser.add_argument('--no_wikitext_eval', action='store_true', help='Whether to perform Wikitext evaluation.')
    parser.add_argument('--stream_llm_start_size', type=int, default=4, help='Num-sink tokens to keep for StreamingLLM')
    parser.add_argument('--eval_subset', type=int, default=None)
    parser.add_argument('--eval_llm_mode', type=str, default="TopSparse", help="oracle, lookahead_magnitude, lookahead_firstlayer_magnitude, predictor, h2o, streamingLLM")
    parser.add_argument('--token_sparse_method', type=str, default="fixed_10pc", help="LazyLLM, progressive_xpc, fixed_xpc...")
    parser.add_argument('--task_list', type=lambda s: [item for item in s.split(',')], default=["arc_easy", "hellaswag"], help='Comma-separated list of tasks for evaluation')
    parser.add_argument('--do_downstream_eval', action='store_true', help='Whether to perform downstream evaluation.')
    parser.add_argument('--do_longbench_eval', action='store_true', help='Whether to perform LongBench evaluation.')
    parser.add_argument('--longbench_datasets', type=lambda s: [item for item in s.split(',')], 
                        default=["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"],
                        help='Comma-separated list of datasets for LongBench evaluation')

    # Predictor Design Related Arguments
    parser.add_argument('--lookahead', type=int, default=0)
    parser.add_argument('--sliding_window', type=int, default=None, help='Sliding window at eval IF comparing to SnapKV, set it to 16: Very Important!!!!!')
    parser.add_argument('--randomize_init', action='store_true', help='Very Experimental! Tries to train predictor on RANDOMLY initialized transformer...')
    parser.add_argument('--train_headpredictor', action='store_true', help='Train Head Predictor')
    parser.add_argument('--min_sparse_index', type=int, default=4, help="Num of Sink Tokens")
    parser.add_argument('--attn_reduce_factor', type=int, default=8, help="reduce factor for token predictor attention")
    parser.add_argument('--head_attn_reduce_factor', type=int, default=2, help="reduce factor for head predictor attention")
    parser.add_argument('--dDash', type=int, default=16, help='Attn Red-dim')
    parser.add_argument('--intdim', type=int, default=512, help='Int-Proc Dim')

    # Model-Mode (Config, Calibrate) related arguments
    parser.add_argument('--calibrate_thresholds', action='store_true', help='Calibrate Per-Head Token Thresholding.')
    parser.add_argument('--test_with_thresholds', action='store_true', help='Test With Per-Head Token Thresholding, must have calibrated before!')
    args = parser.parse_args()

    assert args.model_mode != 'finetune', "Fine-tuning is in main.py"
    args.do_wikitext_eval = not args.no_wikitext_eval
    dowandb = not args.no_wandb
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    print("IF EVALUATING: To compare with SnapKV Fairly, please set --sliding_window to 16 for experiments.")

    if dowandb:
        if args.wname is not None:
            wandb.init(project=args.proj_name, name=args.wname, config=args)
        else:
            wandb.init(project=args.proj_name, config=args)

    model_path = args.model_path
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
    
    if args.model_parallelism:
        from transformers import AutoConfig
        device_cnt = torch.cuda.device_count()
        if device_cnt <= 2:
            extra_param_device = device_cnt - 1
            device_map = {"model.embed_tokens": extra_param_device, "model.rotary_emb": extra_param_device, "model.norm": extra_param_device, "lm_head": extra_param_device}
            device_num_layers = config.num_hidden_layers // device_cnt
            for i in range(config.num_hidden_layers):
                device_map[f"model.layers.{i}"] = i // device_num_layers 
        else:
            extra_param_device = 0
            device_map = {"model.embed_tokens": extra_param_device, "model.rotary_emb": extra_param_device, "model.norm": extra_param_device, "lm_head": extra_param_device}
            device_num_layers = config.num_hidden_layers // (device_cnt - 1)
            for i in range(config.num_hidden_layers):
                device_map[f"model.layers.{i}"] = (i % (device_cnt - 1)) + 1
            device_map[f'model.layers.{0}'] = 0
        dtype = torch.float16 if args.model_mode == "eval" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            offload_folder=None,
            trust_remote_code=True,
            use_auth_token=True,
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, offload_folder=None, trust_remote_code=True, use_auth_token=True, **kwargs).cuda()
    if args.randomize_init:
        model = AutoModelForCausalLM.from_config(config).cuda()

    if not hasattr(config, "num_hidden_layers"):
        args.producer_frequency = config.num_layers
    else:
        args.producer_frequency = config.num_hidden_layers

    ############### GENERATE BEFORE MODIFICATION ################
    question = "A $y$-intercept is a point on the graph that lies on the $y$-axis, so $x = 0$. Hence, the number $y$-intercepts corresponds to the number of real solutions of the quadratic equation $y^2 - 4y - 1 = 0$. The discriminant of this quadratic equation is $(-4)^2 + 4 \cdot 1 \cdot (-1) = 20$, which is positive, so the quadratic has two distinct real roots. Therefore, the number of $y$-intercepts is $\boxed{2}$. \n  \n [asy] \n size(150); \n real ticklen=3; \n real tickspace=2; \n  \n real ticklength=0.1cm; \n real axisarrowsize=0.14cm; \n pen axispen=black+1.3bp; \n real vectorarrowsize=0.2cm; \n real tickdown=-0.5; \n real tickdownlength=-0.15inch; \n real tickdownbase=0.3; \n real wholetickdown=tickdown; \n void rr_cartesian_axes(real xleft, real xright, real ybottom, real ytop, real xstep=1, real ystep=1, bool \n  \n useticks=false, bool complexplane=false, bool usegrid=true) { \n  \n import graph; \n  \n real i; \n  \n if(complexplane) { \n  \n label('$\textnormal{Re}$',(xright,0),SE); \n  \n label('$\textnormal{Im}$',(0,ytop),NW); \n  \n } else { \n  \n label('$x$',(xright+0.4,-0.5)); \n  \n label('$y$',(-0.5,ytop+0.2)); \n  \n } \n  \n ylimits(ybottom,ytop); \n  \n xlimits( xleft, xright); \n  \n real[] TicksArrx,TicksArry; \n  \n for(i=xleft+xstep; i<xright; i+=xstep) { \n  \n if(abs(i) >0.1) { \n  \n TicksArrx.push(i); \n  \n } \n  \n } \n  \n for(i=ybottom+ystep; i<ytop; i+=ystep) { \n  \n if(abs(i) >0.1) { \n  \n TicksArry.push(i); \n  \n } \n  \n } \n  \n if(usegrid) {"    
    embed_device = model.model.embed_tokens.weight.device
    premod_input_ids = tokenizer(question, return_tensors="pt").input_ids.to(embed_device)
    with torch.no_grad(), autocast():
        premod_generated_ids = model.generate(
            input_ids=premod_input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    premod_output_ids = premod_generated_ids[0][len(premod_input_ids[0]):]
    premod_answer = tokenizer.decode(premod_generated_ids[0][len(premod_input_ids[0]):], skip_special_tokens=True)
    ############### GENERATE BEFORE MODIFICATION ################


    if args.architecture == "llama" and "Yarn-Llama" not in model_path:
        print("Running module replacement")
        if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
            from modify_models.modify_llama import convert_kvcache_experimental
            from modify_models.modify_llama import LlamaAttentionExperimental
        else:
            from modify_models.modify_llama_baselines import convert_kvcache_experimental
            from modify_models.modify_llama_baselines import LlamaAttentionExperimental

        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "mistral":
        print("Running Mistral module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_mistral import convert_kvcache_experimental
        else:
            from modify_models.modify_mistral_baselines import convert_kvcache_experimental

        model = convert_kvcache_experimental(model, config, args.producer_frequency)
    elif args.architecture == "mixtral":
        print("Running Mixtral module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_mixtral import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Mixtral yet")

        model = convert_kvcache_experimental(model, config, args.producer_frequency)
    elif args.architecture == "phi3":
        print("Running Phi3 module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_phi3 import convert_kvcache_experimental
        else:
            from modify_models.modify_phi3_baselines import convert_kvcache_experimental

        model = convert_kvcache_experimental(model, config, args.producer_frequency)
    elif args.architecture == "glm":
        print("Running GLM module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_glm import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for GLM yet")

        model = convert_kvcache_experimental(model, config, args.producer_frequency)
    elif args.architecture == "qwen":
        print("Running Qwen module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_qwen import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Qwen yet")
    
        model = convert_kvcache_experimental(model, config, args.producer_frequency)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported")

    token_sparsity_list = []
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.eval_llm_mode = args.eval_llm_mode
            module.token_sparse_method = args.token_sparse_method
            module.set_token_sparsity()
            token_sparsity_list.append(1. - module.sparse_aggression)
            module.stream_llm_start_size = args.stream_llm_start_size
            module.num_tok_per_page = args.num_tok_per_page
            module.producer_frequency = args.producer_frequency
            module.dDash = args.dDash
            module.attn_reduce_factor = args.attn_reduce_factor
            module.head_attn_reduce_factor = args.head_attn_reduce_factor
            module.intdim = args.intdim
            module.flash_attn = args.flash_attn
            module.train_headpredictor = args.train_headpredictor
            module.min_sparse_index = args.min_sparse_index
            module.lookahead = args.lookahead
            module.num_layers_pred = module.producer_frequency
            module.sliding_window = args.sliding_window

            if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
                if module.layer_idx % args.producer_frequency == 0:
                    module.update_predictor()

    if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
        model._prepare_cache_for_generation = patched_prepare_cache_for_generation.__get__(
            model, model.__class__
        )
    else:
        print("WARNING: We are not patching the DynamicCache.")

    try:
        print("\n\n\n Expected Sparsity For Method: ", sum(token_sparsity_list) / len(token_sparsity_list), "\n\n\n")
    except:
        pass

    if token_sparsity_list:
        avg_token_sparsity = sum(token_sparsity_list) / len(token_sparsity_list)
        args.net_sparsity = avg_token_sparsity
    if dowandb:
        wandb.log({
            "avg_token_sparsity": avg_token_sparsity
        })
    
    
    set_inference_mode(model, True)
    
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(embed_device)

    with torch.no_grad(), autocast():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    output_ids = generated_ids[0][len(input_ids[0]):]
    answer = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    dem = "-"*20
    print(f"{dem}\n Original Dense Model Answer:\n{dem}\n{premod_answer}")
    print(f"{dem}\n Modified Model Answer:\n{dem}\n{answer}")
    if dowandb:
        wandb.finish()