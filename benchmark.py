import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import numpy as np
import os
import time
import gc
import wandb
from functools import partial


def get_producer_layers(model):
    """
    Traverses the model to find the producer layer (layer_idx=0).cc
    """
    producer_modules = []
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental") and module.layer_idx == 0:
            producer_modules.append(module)
    return producer_modules
    
def measure_latency(model, tokenizer, seq_len=128, batch_size=1, warmup_steps=5, measure_steps=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    for module in model.modules():
        module.seq_len_sim = seq_len
        if hasattr(module, 'predefine_attentionmask'):
            module.predefine_attentionmask(1)
        
    model = torch.compile(model)

    input_ids = torch.full(
        (batch_size, seq_len),
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids, attention_mask=attention_mask, use_cache=False)

    # Measure latency
    timings = []
    with torch.no_grad():
        for _ in range(measure_steps):
            start_time = time.time()
            _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
            timings.append(time.time() - start_time)

    torch.cuda.empty_cache()
    gc.collect()

    # Return both median and standard deviation
    return np.median(timings), np.std(timings)

def test_latency(models, tokenizer, seq_lens=[64, 128, 256, 512, 1024], batch_size=1):
    results = {}
    for model_name, model in models.items():
        print(f"\nTesting latency for {model_name}...")
        results[model_name] = {}
        for seq_len in seq_lens:
            latency = measure_latency(model, tokenizer, seq_len=seq_len, batch_size=batch_size)
            results[model_name][seq_len] = latency
            print(f"[{model_name} | Seq Len: {seq_len}] Median Latency: {latency:.4f} seconds")
    return results

def convert_model(model, config, args):
    """
    Converts the model based on architecture and specified arguments.
    """
    if args.architecture == "llama":
        from modify_llama_benchmark import convert_kvcache_experimental, convert_llama_channel_config_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    elif args.architecture == "mistral":
        from modify_mistral import convert_kvcache_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported")

    for module in model.modules():
        # If module's class name ends with AttentionExperimental
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.eval_llm_mode = args.eval_llm_mode
            module.gfac = args.gfac
            module.token_sparse_method = args.token_sparse_method
            module.set_token_sparsity()
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

            if module.layer_idx == 0:
                module.update_predictor()

    return model

def convert_model_equivalent(model, config, args):
    """
    Converts the model based on architecture and specified arguments.
    """
    if args.architecture == "llama":
        from modify_llama_nopred import convert_kvcache_experimental, convert_llama_channel_config_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    elif args.architecture == "mistral":
        from modify_mistral import convert_kvcache_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency, args.heavy_const, args.group_factor, args.q_bits)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported")

    for module in model.modules():
        # If module's class name ends with AttentionExperimental
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.eval_llm_mode = args.eval_llm_mode
            module.gfac = args.gfac
            module.token_sparse_method = args.token_sparse_method
            module.set_token_sparsity()
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

            if module.layer_idx == 0:
                module.update_predictor()

    return model

# Main
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')
    # add a seed
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Selected model')
    parser.add_argument('--task_list', type=lambda s: [item for item in s.split(',')], default=["arc_easy", "hellaswag"], help='Comma-separated list of tasks for evaluation')
    parser.add_argument('--offloading', action='store_true', help='Whether to use offloading')
    parser.add_argument('--architecture', type=str, default="llama", choices=["llama", "mistral", "mixtral"])
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
    parser.add_argument('--save_interval', type=int, default=2000, help='Number of steps after which to save a checkpoint.')

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

    parser.add_argument('--do_perf_eval', action='store_true', help='Whether to perform performance evaluation.')
    
    parser.add_argument('--prompt', type=str, default=None, help='Prompt to use for text generation.')
    parser.add_argument('--max_gen_tokens', type=int, default=50, help='Maximum number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for text generation.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling for text generation.')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search.')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_map = "cuda"
    model_path = args.model_path
    kwargs = {"torch_dtype": torch.float32, "device_map": device_map}
    args.seq_lens = [256, 512, 1024, 2048]
    args.batch_size = 1
    model = AutoModelForCausalLM.from_pretrained(model_path, offload_folder=None, trust_remote_code=True, use_auth_token=True, **kwargs).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, use_fast=True)
    config = AutoConfig.from_pretrained(model_path, use_auth_token=True)


    # If config doesn't have an attribute num_hidden_layers
    if not hasattr(config, "num_hidden_layers"):
        args.producer_frequency = config.num_layers
    else:
        args.producer_frequency = config.num_hidden_layers
    # Torch Compile
    # model = torch.compile(model)

    # Load Config
    config = AutoConfig.from_pretrained(args.model_path)

    # Evaluate Original Model Latency
    print("\nConverting Model...")
    original_model = convert_model_equivalent(model, config, args)
    original_model = model.to('cuda')  # Place on CUDA for evaluation
    original_results = {}
    for seq_len in args.seq_lens:
        median, std = measure_latency(original_model, tokenizer, seq_len=seq_len, batch_size=args.batch_size)
        original_results[seq_len] = {"median": median, "std": std}
        print(f"[Original | Seq Len: {seq_len}] Median Latency: {median:.4f}s | Std Dev: {std:.4f}")
    original_model.cpu()  # Offload after evaluation


    model = AutoModelForCausalLM.from_pretrained(model_path, offload_folder=None, trust_remote_code=True, use_auth_token=True, **kwargs).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, use_fast=True)
    config = AutoConfig.from_pretrained(model_path, use_auth_token=True)
    print("\nConverting Model...")
    converted_model = convert_model(model, config, args)
    converted_model.to('cuda')  # Place on CUDA for evaluation
    converted_results = {}
    for seq_len in args.seq_lens:
        median, std = measure_latency(converted_model, tokenizer, seq_len=seq_len, batch_size=args.batch_size)
        converted_results[seq_len] = {"median": median, "std": std}
        print(f"[Converted | Seq Len: {seq_len}] Median Latency: {median:.4f}s | Std Dev: {std:.4f}")

    num_tot_params = sum(p.numel() for p in converted_model.parameters())

    producer_layer = get_producer_layers(model)[0]

    tokpred_params = sum(p.numel() for p in producer_layer.sparse_token_predictor.parameters())
    head_pred_params = sum(p.numel() for p in producer_layer.sparse_head_predictor.parameters())
    total_params = tokpred_params + head_pred_params
    total_model_params = sum(p.numel() for p in model.parameters())
    spt_perc = tokpred_params / total_model_params * 100
    hpt_perc = head_pred_params / total_model_params * 100
    print("Total Model Parameters: ", total_model_params)
    print("Token Predictor Param Count: ", tokpred_params)
    print("Head Predictor Param Count: ", head_pred_params)
    print("Total Predictor Param Count: ", total_params)
    print("Percentage Of Model Params in Token Predictor: ", spt_perc)
    print("Percentage Of Model Params in Head Predictor: ", hpt_perc)
    # # Compare Latencies
    # for seq_len in args.seq_lens:
    #     orig_latency = original_results[seq_len]
    #     conv_latency = converted_results[seq_len]
    #     overhead = 100 * (conv_latency - orig_latency) / orig_latency
    #     print(f"[Seq Len: {seq_len}] Original: {orig_latency:.4f}s | Converted: {conv_latency:.4f}s | Overhead: {overhead:.2f}%")

    # Cleanup
    del original_model, converted_model
    torch.cuda.empty_cache()
    gc.collect()
    import csv

    # File path for the CSV
    csv_file_path = "pred_overhead_werr.csv"
    rows = []
    for seq_len in args.seq_lens:
        orig_latency = original_results[seq_len]["median"]
        orig_std = original_results[seq_len]["std"]
        conv_latency = converted_results[seq_len]["median"]
        conv_std = converted_results[seq_len]["std"]

        # Calculate overhead and propagated error
        overhead = 100 * (conv_latency - orig_latency) / orig_latency
        overhead_std = 100 * np.sqrt(
            (conv_std / orig_latency) ** 2 +
            (orig_std * (conv_latency - orig_latency) / (orig_latency ** 2)) ** 2
        )

        rows.append({
            "model_path": model_path,
            "architecture": args.architecture,
            "num_tot_params": num_tot_params / 1e6,  # in millions
            "predictor_params": tokpred_params / 1e6,    # in millions
            "param_ratio": spt_perc,
            "seq_len": seq_len,
            "orig_latency": orig_latency,
            "orig_latency_std": orig_std,
            "conv_latency": conv_latency,
            "conv_latency_std": conv_std,
            "overhead": overhead,
            "overhead_std": overhead_std  # Add the propagated error
        })

    # Write to CSV
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "model_path", "architecture", "num_tot_params", "predictor_params", "param_ratio",
            "seq_len", "orig_latency", "orig_latency_std", "conv_latency", "conv_latency_std", "overhead", "overhead_std"
        ])
        if not file_exists:
            writer.writeheader()  # Write headers if the file does not exist
        writer.writerows(rows)  # Append rows