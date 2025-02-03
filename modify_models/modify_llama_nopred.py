import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc
import time

import traceback
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from utils import LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv, sorted_index_to_mask
from utils import calculate_hit_metrics, calculate_effective_sparsity, threshold_to_mask
from transformers.cache_utils import DynamicCache
from predictor import TokenImportancePredictorAttentive, PredictorDynamicCache, HeadImportancePredictor, attention_mse_loss, attention
from threshold_calib_dict import *

from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss

# torch.backends.cuda.enable_flash_sdp(enabled=True)
# torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

class LlamaAttentionExperimental(nn.Module):
    def __init__(self, config: LlamaConfig, producer=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.inference_mode = False
        self.producer = producer
        self.layer_idx = layer_idx
        self.token_sparse_method = None
        self.sparse_aggression = None
        self.stream_llm_start_size = None
        self.dDash = None
        self.intdim = None
        self.attn_reduce_factor = None
        self.head_attn_reduce_factor = None
        self.effective_sparsity = None
        self.min_sparse_index = None
        self.pred_hid_size = self.hidden_size
        self.num_tok_per_page = None
        self.calc_hitrates = False
        self.flash_attn = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.seq_len_sim = None
        self.tok_calibration_set = threshold_model_dictionary.get(config._name_or_path, None)

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.headmseloss = MSELoss(reduction='none')
            self.headmsemagn_loss = None
        
        if self.producer is None:  # This is the producer layer
            self.q_importance = None  # Shared mask across layers during inference
            self.k_importance = None
            self.actmagn_masklist = {}
            self.available_tokens = {}

        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def predefine_attentionmask(self, bsz):
        self._pre_causal_mask = torch.ones((bsz, 1, self.seq_len_sim, self.seq_len_sim), dtype=torch.bool).cuda()
        self._pre_causal_mask = self._pre_causal_mask.triu(diagonal=1)
        self._attention_mask = torch.zeros_like(self._pre_causal_mask, dtype=torch.float32)
        self._attention_mask.masked_fill_(self._pre_causal_mask, float("-inf"))
        # create a tensor of shape self.seq_len_sim // 2, indices inside unique of range 0 to self.seq_len_sim
        self._attend_index = torch.randint(0, self.seq_len_sim, (bsz, self.seq_len_sim // 2)).cuda().unsqueeze(0).unsqueeze(-1).expand(-1, self.num_heads, self.seq_len_sim//2, self.head_dim)

            # causal_mask = causal_mask.triu(diagonal=1)  # Upper triangular part
            # attention_mask = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
            # attention_mask.masked_fill_(causal_mask, float("-inf"))
        
    def update_predictor(self):
        self.sparse_token_predictor = TokenImportancePredictorAttentive(
            self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = self.dDash, \
            intdim = self.intdim, attn_reduce_factor=self.attn_reduce_factor
        )

    def set_head_sparsity(self, head_sparsity_aggression, global_prune):
        self.head_sparsity_aggression = head_sparsity_aggression
        self.head_global_prune = global_prune

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        if self.token_sparse_method == "LazyLLM":
            if self.layer_idx <= 9:
                self.sparse_aggression = 1
            elif self.layer_idx <= 19:
                self.sparse_aggression = 0.7
            elif self.layer_idx <= 28:
                self.sparse_aggression = 0.4
            else:
                self.sparse_aggression = 0.1
        elif "fixed" in self.token_sparse_method:
            if self.layer_idx == 0:
                self.sparse_aggression = 1
            else:
                self.sparse_aggression = 1 - float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
        elif "progressive" in self.token_sparse_method:
            pc_drop = float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
            self.sparse_aggression = (1 - pc_drop) ** (self.layer_idx)  # (x% per layer, progressive_xpc style)
        else:
            raise ValueError(f"Unknown token sparsity method {self.token_sparse_method}")
            

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type") or self.config.rope_scaling.get("rope_type")
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear" or scaling_type == 'llama3':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[DynamicCache, PredictorDynamicCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PredictorDynamicCache]]:
        attention_mask = self._attention_mask
        bsz, q_len, _ = hidden_states.size()
        Ltrack = hidden_states.size(1)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # AHMED: Modified this to use the newer version.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        kv_seq_len = key_states.shape[-2]
        final_mask = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)

        assert self.head_dim % self.group_factor == 0
        min_sparse_index = self.min_sparse_index
        if self.layer_idx > 0:
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, is_causal=True
            )
            # attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # attn_weights = attn_weights + attention_mask

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)
        if self.layer_idx == 0:
            if self.effective_sparsity is None:
                self.effective_sparsity = 0.0

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, -1, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        # if self.producer is None:
        #     try:
        #         q_importance, k_importance, _ = self.sparse_token_predictor(
        #             hidden_states, 
        #             attention_mask=attention_mask, 
        #             position_ids=position_ids, 
        #             past_key_value=None, 
        #             use_cache=False
        #         )
        #         q_len = attn_output.size(1)
        #         k_len = k_importance.size(-1)
        #     except:
        #         print(traceback.format_exc())
        #         import pdb; pdb.set_trace()

        #     self.q_importance = q_importance
        #     self.k_importance = k_importance
        return attn_output, None, None

def convert_kvcache_experimental(model, config, producer_frequency, heavy_const=256, group_factor=8, label_bits=4):
    producer_layer = None
    layer_counter = {'idx': 0}

    def recurse_convert(parent_module):
        nonlocal producer_layer
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                if layer_counter['idx'] % producer_frequency == 0:
                    new_module = LlamaAttentionExperimental(config).to(dtype).to(device)
                    producer_layer = new_module
                else:
                    new_module = LlamaAttentionExperimental(
                        config,
                        producer=producer_layer,
                        layer_idx=layer_counter['idx']
                    ).to(dtype).to(device)
                new_module.load_state_dict(module.state_dict(), strict=False)
                new_module.heavy_const = heavy_const
                new_module.group_factor = group_factor
                new_module.label_bits = label_bits
                is_producer = layer_counter['idx'] % producer_frequency == 0
                if is_producer:
                    print(f"Converted Producer layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                else:
                    print(f"Converted layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                parent_module._modules[name] = new_module
                layer_counter['idx'] += 1
    recurse_convert(model)
    return model

def convert_llama_channel_config_experimental(model, channel_config, selected_channel="k"):
    selected_channel = "." + selected_channel + "_proj"

    for name, module in model.named_modules():
        if isinstance(module, LlamaAttentionExperimental):
            device = next(module.parameters()).device
            module.sorted_channel = torch.tensor(channel_config[name + selected_channel]).to(device)

    return model
