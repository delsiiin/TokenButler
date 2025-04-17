# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils import calculate_hit_metrics, calculate_effective_sparsity, threshold_to_mask, SlidingWindowCache, enforce_sliding_window, sorted_index_to_mask
from predictor import TokenImportancePredictorAttentive, HeadImportancePredictor, attention_mse_loss, attention

from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss
import math
import torch.nn.functional as F
from .cache_utils import Cache, DynamicCache, StaticCache, PredictorDynamicCache

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    q_importance: torch.Tensor = None,
    k_importance: torch.Tensor = None,
    **kwargs,
):
    kv_seq_len = key.shape[-2]
    final_mask = None

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    key_len = key_states.size(2)
    bsz, q_len = query.size(0), query.size(2)

    # if attention_mask is None:
    #     # We want a [q_len, kv_seq_len] boolean upper-triangular mask
    #     causal_mask_2d = torch.ones(q_len, kv_seq_len, 
    #                                 device=query.device, 
    #                                 dtype=torch.bool).triu(diagonal=1)
    #     # Then shape it to [bsz, 1, q_len, kv_seq_len]
    #     causal_mask_4d = causal_mask_2d.unsqueeze(0).expand(bsz, 1, q_len, kv_seq_len)
    #     # Now fill -inf where the mask is True
    #     attention_mask = torch.full_like(causal_mask_4d, 0, dtype=query.dtype)
    #     if q_len != 1:
    #         attention_mask = attention_mask.masked_fill(causal_mask_4d, float("-inf")

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_mask = causal_mask
    
    if module.inference_mode:
        min_sparse_index = module.min_sparse_index
        with torch.no_grad():
            if module.eval_llm_mode == "ExpPred":
                if module.layer_idx > 0:
                    q_importance_tensor = q_importance.float().to(query.device) # [BH, Lq, D']
                    k_importance_tensor = k_importance.float().to(key_states.device) # [BH, Lk, D']
                    importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(module.head_dim // module.attn_reduce_factor) # [BH, Lq, Lk]
                    importance_mask = importance_mask.view(bsz, module.num_heads, q_len, key_len) # [B, H, Lq, Lk]
                    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling
                    if module.calc_hitrates:
                        module.tok_hit_acc, module.tok_mean_rank_corr, module.tok_max_rank_corr = calculate_hit_metrics(
                            estimated_importance=nn.functional.softmax(importance_mask + attention_mask, dim=-1),
                            true_importance=nn.functional.softmax(attn_weights + attention_mask, dim=-1),
                            top_k_ratio=0.5
                        )
                    if module.calibrate_thresholds:
                        ### Threshold variance investigation
                        unadj_importance_mask = importance_mask.clone()
                        importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                        sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                        sorted_indices = sorted_indices[:, :, -q_len:, :]
                        sorted_values, sorted_ix = torch.sort(importance_mask, dim=-1)
                        sorted_true_values, _ = torch.sort(torch.gather(unadj_importance_mask, dim=-1, index=sorted_ix), dim=-1)
                        true_thresholds = sorted_true_values[:, :, :, int(importance_mask.size(-1) * module.sparse_aggression)]
                        thresholds = sorted_values[:, :, :, int(importance_mask.size(-1) * module.sparse_aggression)]
                        module.true_threshmean = true_thresholds
                        module.threshmean = thresholds
                    if module.test_with_thresholds:
                        unadj_importance_mask = importance_mask.clone()
                        perhead_thresholds = module.tok_calibration_set[module.layer_idx - 1].to(unadj_importance_mask.device) # 0 does not have calibration data.
                        mask_tensor = threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len)
                    else:
                        
                        importance_mask_pred = torch.softmax(importance_mask + attention_mask, dim=-1)
                        _, sorted_indices = importance_mask_pred.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                        sorted_indices = sorted_indices[:, :, -q_len:, :]
                        if q_len == 1:
                            # initialize tensor of zeros with shape like sorted_indices
                            mask_tensor = torch.ones_like(importance_mask_pred)
                            sorted_indices = sorted_indices[:, :, :, int(module.sparse_aggression * key_len):]
                            # scatter value float('-inf') at indexes in sorted_indices to mask_tensor
                            mask_tensor.scatter_(-1, sorted_indices, 0)
                            mask_tensor[:, :, :, :min_sparse_index] = 1
                            if module.sliding_window is not None:
                                mask_tensor[:, :, :, -module.sliding_window:] = 1
                            import pdb; pdb.set_trace()
                        else:
                            mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, module.sparse_aggression, module.sliding_window)
    
                    mask_tensor = mask_tensor.bool()
                    mask_tensor = torch.where(
                        mask_tensor,
                        torch.tensor(0.0, device=mask_tensor.device),
                        torch.tensor(float('-inf'), device=mask_tensor.device)
                    )
                    # ### Threshold variance investigation
                    # if self.sliding_window is not None:
                    #     if not hasattr(self, "window_cache"):
                    #         self.window_cache = SlidingWindowCache(max_seq_len=1024,
                    #                                             sliding_window=self.sliding_window,
                    #                                             device=mask_tensor.device)
                    #     window = self.window_cache.get_window(q_len, key_len)
                    #     mask_tensor = enforce_sliding_window(mask_tensor, window)
                    final_mask = mask_tensor

                    module.final_mask_investigate = final_mask
                    attn_weights = attn_weights + attention_mask
                    # if q_len == 1:
                    # During train-time, we want to keep this off, all our train-evals are 1 decode step focused
                    # not generation focused. So, we still want to assess prefill sparsity. 
                    # However, at inference time (generation), we should only use mask_tensor
                    # when q_len == 1
                    attn_weights = attn_weights + mask_tensor
                else:
                    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling
                    attn_weights = attn_weights + attention_mask
            else:
                raise ValueError(f"Unknown eval mode {module.eval_llm_mode}")
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    else:
        
        if module.flash_attn:
            if module.layer_idx > 0:
                # Token hit-rates cannot be calculated if using flash attention.
                module.tok_hit_acc = 0
                q_importance_tensor = q_importance.float().to(query.device) # [BH, Lq, D']
                k_importance_tensor = k_importance.float().to(key_states.device) # [BH, Lk, D']
                device_index = query.device.index
                assert module.lookahead == 0, "Lookahead not supported with flash attention yet. Please disable --flash_attn"
                with torch.cuda.device(device_index):
                    attn_output, mse_loss = attention_mse_loss(query.contiguous().to(torch.float16),
                                                                key_states.contiguous().to(torch.float16),
                                                                value_states.contiguous().to(torch.float16),
                                                                q_importance_tensor.contiguous().to(torch.float16),
                                                                k_importance_tensor.contiguous().to(torch.float16), 
                                                                True
                                                                )
                module.tok_hit_acc, module.tok_mean_rank_corr, module.tok_max_rank_corr = 0, 0, 0
                attn_output = attn_output.to(query.dtype)
                if not torch.isnan(mse_loss):
                    module.msemagn_loss = mse_loss
                else:
                    raise ValueError(f"NaN loss detected: {mse_loss}")
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(query, key_states, value_states, attn_mask=None, is_causal=True)
        else:
            min_sparse_index = module.min_sparse_index
            attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling
            if module.layer_idx > 0:
                q_importance_tensor = q_importance.float().to(query.device) # [BH, Lq, D']
                k_importance_tensor = k_importance.float().to(key_states.device) # [BH, Lk, D']
                importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(module.head_dim // module.attn_reduce_factor) # [BH, Lq, Lk]
                importance_mask = importance_mask.view(bsz, module.num_heads, q_len, key_len) # [B, H, Lq, Lk]

                if module.lookahead == 0:
                    if module.softmax_causal_loss_mse:
                        module.msemagn_loss = module.mseloss(
                            torch.softmax(attn_weights + attention_mask, dim=-1), 
                            torch.softmax(importance_mask + attention_mask, dim=-1)
                            )
                    elif module.softmax_causal_loss_ce:
                        target_dist = F.softmax(attn_weights + attention_mask, dim=-1).detach()
                        pred_dist = F.softmax(importance_mask + attention_mask, dim=-1)
                        ce = -(target_dist * (pred_dist + 1e-9).log()).sum(dim=-1)  
                        module.msemagn_loss = ce
                    else:
                        module.msemagn_loss = module.mseloss(attn_weights, importance_mask)
                else:
                    module.msemagn_loss = module.mseloss(attn_weights[:, :, module.lookahead:, :], importance_mask[:, :, :-module.lookahead, :])
                if module.late_context_upweight:
                    # Here, if we do seq_len_q with [1,1,seq_len_q,1], we focus on rewarding longer decodes more
                    # but,  if we do seq_len_k with [1,1,1,seq_len_k], we focus on rewarding correctness on more recent tokens more
                    # Since we want longer decode consistency, we will do seq_len_q
                    seq_len_q = module.msemagn_loss.shape[-2]  # Lk
                    weighting = torch.linspace(
                        start=0.1, 
                        end=1.0, 
                        steps=seq_len_q, 
                        device=module.msemagn_loss.device
                    )
                    weighting = weighting.view(1, 1, seq_len_q, 1)  # shape [1, 1, 1, Lk]
                    module.msemagn_loss = module.msemagn_loss * weighting
                    if module.softmax_causal_loss_mse:
                        module.msemagn_loss = module.msemagn_loss.sum(dim=-2).mean(dim=-1)  # shape [B, H]
                    else:
                        module.msemagn_loss = module.msemagn_loss.mean(dim=(-2, -1))  # shape [B, H]
                else:
                    if module.softmax_causal_loss_mse:
                        module.msemagn_loss = module.msemagn_loss.sum(dim=-2).mean(dim=-1)  # shape [B, H]
                    else:
                        module.msemagn_loss = module.msemagn_loss.mean(dim=(-1, -2))
                module.msemagn_loss = module.msemagn_loss.mean()

                if module.calc_hitrates:
                    module.tok_hit_acc, module.tok_mean_rank_corr, module.tok_max_rank_corr = calculate_hit_metrics(
                        estimated_importance=nn.functional.softmax(importance_mask + attention_mask, dim=-1),
                        true_importance=nn.functional.softmax(attn_weights + attention_mask, dim=-1),
                        top_k_ratio=0.5
                    )

                # merge attn
                importance_mask_pred = torch.softmax(importance_mask + attention_mask, dim=-1)
                _, sorted_indices = importance_mask_pred.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                sorted_indices = sorted_indices[:, :, -q_len:, :]
                if q_len == 1:
                    # initialize tensor of zeros with shape like sorted_indices
                    mask_tensor = torch.ones_like(importance_mask_pred)
                    sorted_indices = sorted_indices[:, :, :, int(module.sparse_aggression * key_len):]
                    # scatter value float('-inf') at indexes in sorted_indices to mask_tensor
                    mask_tensor.scatter_(-1, sorted_indices, 0)
                    mask_tensor[:, :, :, :min_sparse_index] = 1
                    if module.sliding_window is not None:
                        mask_tensor[:, :, :, -module.sliding_window:] = 1
                    import pdb; pdb.set_trace()
                else:
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, module.sparse_aggression, module.sliding_window)

                mask_tensor = mask_tensor.bool()

                attn_weights_comp = (attn_weights * mask_tensor) + ((~mask_tensor) * torch.finfo(attn_weights.dtype).min)
                
                # attn_weights_comp_lse = torch.logsumexp(attn_weights_comp, -1, keepdim=True)
                
                merge_mask = attention_mask.bool() * (~mask_tensor)

                merge_mask = merge_mask.bool()

                importance_mask = importance_mask * merge_mask
                
                # print(torch.min(importance_mask.sum(dim=-1, keepdim=True)), 111111111111)
                # print(torch.min(importance_mask), 22222222222222)

                # predictor_lse = torch.log(importance_mask.sum(dim=-1, keepdim=True) + 1e-6)
                # norm_factor_lse = torch.logaddexp(predictor_lse, attn_weights_comp_lse)
                # importance_mask = (torch.log(importance_mask + 1e-6) * merge_mask) + ((~merge_mask) * attn_weights_comp)
                # importance_mask = torch.exp(importance_mask - norm_factor_lse)

                importance_mask = importance_mask * merge_mask + attn_weights_comp * (~merge_mask)

                importance_mask = torch.softmax(importance_mask, dim=-1, dtype=torch.float32).to(value_states.dtype)

                # importance_mask = temperature_softmax(importance_mask, 1, dim=-1, dtype=torch.float32).to(value_states.dtype)

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

                merge_mseloss = module.merge_mseloss(importance_mask, attn_weights).sum(dim=-2).mean(dim=-1).mean()
                module.msemagn_loss += merge_mseloss
                
            else:

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

    # attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # attn_output = torch.matmul(attn_weights, value_states)

    checkeverytime = hasattr(module, 'test_with_thresholds')
    if checkeverytime:
        checkeverytime = module.test_with_thresholds
    if final_mask is not None:
        if module.effective_sparsity is None or checkeverytime:
        # if True:
            true_mask = final_mask + attention_mask
            num_deact = true_mask.bool().sum(dim=-1)                   # Number of tokens disabled.
            causally_deact = (attention_mask.bool()).sum(dim=-1).expand_as(num_deact)        # Number of tokens disabled causally anyway
            additional_deact = (num_deact - causally_deact)
            num_active = (~attention_mask.bool()).sum(dim=-1).expand_as(num_deact)    # Number of tokens active at this position if zero-sparsity
            effective_sparsity = 100 * (additional_deact.float() / num_active.float()).mean().item()
            module.effective_sparsity = effective_sparsity
            print(f"Layer {module.layer_idx}: Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)
    if module.layer_idx == 0:
        if module.effective_sparsity is None:
            module.effective_sparsity = 0.0

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.inference_mode = False
        self.sparse_aggression = None
        self.stream_llm_start_size = None
        self.effective_sparsity = None
        self.min_sparse_index = None
        self.num_tok_per_page = None
        self.calc_hitrates = False
        self.flash_attn = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.late_context_upweight = False
        self.softmax_causal_loss_mse = False
        self.softmax_causal_loss_ce = False

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.merge_mseloss = MSELoss(reduction='none')

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Union[DynamicCache, PredictorDynamicCache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        q_importance = None,
        k_importance = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PredictorDynamicCache]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(input_shape[0] * self.sparse_aggression)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            q_importance=q_importance,
            k_importance=k_importance,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        q_importance = None,
        k_importance = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_importance=q_importance,
            k_importance=k_importance,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config=config

        self.token_sparse_method = None
        self.dDash = None
        self.intdim = None
        self.attn_reduce_factor = None
        self.head_attn_reduce_factor = None
        self.pred_hid_size = self.config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def update_predictor(self):
        self.sparse_token_predictors = nn.ModuleList(
            [TokenImportancePredictorAttentive(
                    self.config, self.pred_hid_size, self.num_heads, self.num_hidden_layers, dropout=0.1, dDash = self.dDash, \
                    intdim = self.intdim, attn_reduce_factor=self.attn_reduce_factor
                ) for layer_idx in range(self.config.num_hidden_layers-1)
            ]
        )
        for idx in range(self.config.num_hidden_layers-1):
            self.sparse_token_predictors[idx] = self.sparse_token_predictors[idx].to(self.layers[idx].self_attn.q_proj.weight.device)
            self.sparse_token_predictors[idx].flash_attn = self.layers[idx].self_attn.flash_attn

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        if self.token_sparse_method is not None:
            try:
                mname = self.config._name_or_path.split("/")[-1]
                read_path = f"threshold_calibs/{mname}/{self.token_sparse_method}.pkl"
                threshold_model_dictionary = torch.load(read_path)
                self.tok_calibration_set = threshold_model_dictionary
            except:
                pass
        for idx in range(len(self.layers)):
            if self.token_sparse_method == "LazyLLM":
                if idx <= 9:
                    self.layers[idx].self_attn.sparse_aggression = 1
                elif idx <= 19:
                    self.layers[idx].self_attn.sparse_aggression = 0.7
                elif idx <= 28:
                    self.layers[idx].self_attn.sparse_aggression = 0.4
                else:
                    self.layers[idx].self_attn.sparse_aggression = 0.1
            elif "fixed" in self.token_sparse_method:
                if idx == 0:
                    self.layers[idx].self_attn.sparse_aggression = 1
                else:
                    self.layers[idx].self_attn.sparse_aggression = 1 - float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
            elif "progressive" in self.token_sparse_method:
                pc_drop = float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
                self.layers[idx].self_attn.sparse_aggression = (1 - pc_drop) ** (idx)  # (x% per layer, progressive_xpc style)
            else:
                raise ValueError(f"Unknown token sparsity method {self.token_sparse_method}")

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        q_importance = None
        k_importance = None
        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if idx == self.config.num_hidden_layers - 1:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        q_importance,
                        k_importance,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        q_importance,
                        k_importance,
                    )
                    q_importance, k_importance = self.sparse_token_predictors[idx](
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,  # the same single cache
                        use_cache=use_cache,
                        layer_idx=idx,       # or pass 0
                    )
                
            else:
                if idx == self.config.num_hidden_layers - 1:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        q_importance=q_importance,
                        k_importance=k_importance,
                        **flash_attn_kwargs,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        q_importance=q_importance,
                        k_importance=k_importance,
                        **flash_attn_kwargs,
                    )
                    q_importance, k_importance = self.sparse_token_predictors[idx](
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,  # the same single cache
                        use_cache=use_cache,
                        layer_idx=idx,       # or pass 0
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
