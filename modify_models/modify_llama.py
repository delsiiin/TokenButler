import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc

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
        self.train_headpredictor = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.tok_calibration_set = threshold_model_dictionary.get(config._name_or_path, None)

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.headmseloss = MSELoss(reduction='none')
            self.headmsemagn_loss = None
        
        if self.producer is None:  # This is the producer layer
            self.q_importance = None  # Shared mask across layers during inference
            self.k_importance = None
            self.head_importances = None
            self.actmagn_masklist = {}
            self.available_tokens = {}

        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        
    def update_predictor(self):
        self.sparse_token_predictor = TokenImportancePredictorAttentive(
            self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = self.dDash, \
            intdim = self.intdim, attn_reduce_factor=self.attn_reduce_factor
        ).to('cuda:0')
        self.sparse_token_predictor.flash_attn = self.flash_attn
        if self.train_headpredictor:
            self.sparse_head_predictor = HeadImportancePredictor(
                self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = self.dDash, \
                intdim = self.intdim, attn_reduce_factor=self.head_attn_reduce_factor
            ).to('cuda:0')
            self.sparse_head_predictor.flash_attn = self.flash_attn
        else:
            # initialize very small model to keep in-memory for seq-len 2048
            self.sparse_head_predictor = HeadImportancePredictor(
                self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = 4, \
                intdim = 16, attn_reduce_factor=(self.pred_hid_size // self.num_heads)
            ).to('cuda:0')
            # Dont use flash-attention if head-prediction is in pseudo mdoe.
            self.sparse_head_predictor.flash_attn = False

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
        bsz, q_len, _ = hidden_states.size()
        Ltrack = hidden_states.size(1)

        # Convert DynamicCache to PredictorDynamicCache if needed
        if past_key_value is not None and not isinstance(past_key_value, PredictorDynamicCache):
            if isinstance(past_key_value, DynamicCache):
                assert past_key_value.get_seq_length() == 0, "If past_key_value is DynamicCache, then it must be empty"
            past_key_value = PredictorDynamicCache()

        if q_len != 1:  # this is prefill stage for first token output, reset q-k importance tensors
            self.q_importance = None
            self.k_importance = None
            self.head_importances = None
        past_key_value_sp = None if past_key_value is None else past_key_value.get_predictor_cache()
        past_key_value_hp = None if past_key_value is None else past_key_value.get_head_predictor_cache()

        # if past_key_value_sp is not None:
        #     print("Past Key Value Sparsity:", past_key_value_sp)


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

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(q_len * self.sparse_aggression)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # AHMED: Modified this to use the newer version.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)

        kv_seq_len = key_states.shape[-2]
        final_mask = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)

        # Ahmed Modification. Always set an attention_mask
        # Create causal mask if attention_mask is None
        if attention_mask is None:
            # Create causal mask
            # [bsz, 1, q_len, kv_seq_len]
            # @Ahmed -- this should be corrected for decode.
            causal_mask = torch.ones((bsz, 1, q_len, kv_seq_len), device=hidden_states.device, dtype=torch.bool)
            causal_mask = causal_mask.triu(diagonal=1)  # Upper triangular part
            attention_mask = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
            attention_mask.masked_fill_(causal_mask, float("-inf"))
        assert self.head_dim % self.group_factor == 0
        if self.inference_mode:
            min_sparse_index = self.min_sparse_index
            with torch.no_grad():
                if evalmode == "ExpPred":
                    if self.layer_idx > 0:
                        q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                        k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                        importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash) # [BH, Lq, Lk]
                        importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len) # [B, H, Lq, Lk]
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        if self.calc_hitrates:
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=importance_mask,
                                true_importance=attn_weights,
                                top_k_ratio=0.5
                            )
                        if self.calibrate_thresholds:
                            ### Threshold variance investigation
                            unadj_importance_mask = importance_mask.clone()
                            importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                            sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                            sorted_indices = sorted_indices[:, :, -q_len:, :]
                            sorted_values, sorted_ix = torch.sort(importance_mask, dim=-1)
                            sorted_true_values, _ = torch.sort(torch.gather(unadj_importance_mask, dim=-1, index=sorted_ix), dim=-1)
                            true_thresholds = sorted_true_values[:, :, :, importance_mask.size(-1)//2]
                            thresholds = sorted_values[:, :, :, importance_mask.size(-1)//2]
                            self.true_threshmean = true_thresholds
                            self.threshmean = thresholds
                        if self.test_with_thresholds:
                            unadj_importance_mask = importance_mask.clone()
                            perhead_thresholds = self.tok_calibration_set[self.layer_idx - 1].to(unadj_importance_mask.device) # 0 does not have calibration data.
                            mask_tensor = threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len)
                        else:
                            importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                            sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                            sorted_indices = sorted_indices[:, :, -q_len:, :]
                            mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression)
                        ### Threshold variance investigation
                        final_mask = mask_tensor

                        self.final_mask_investigate = final_mask
                        attn_weights = attn_weights + mask_tensor + attention_mask
                    else:
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights = attn_weights + attention_mask
                elif evalmode == "ReplAttn":
                    if self.layer_idx > 0:
                        q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device)
                        k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device)
                        importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash)
                        importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len)
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        if self.calc_hitrates:
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=importance_mask,
                                true_importance=attn_weights,
                                top_k_ratio=0.5
                            )
                        num_heads_to_keep = int(self.num_heads * (1 - self.sparse_aggression))
                        head_importance_tensor = self.producer.head_importances[:, :, :, self.layer_idx % self.producer_frequency]
                        indices = torch.argsort(head_importance_tensor.mean(dim=1), dim=-1, descending=False)[:, :num_heads_to_keep].squeeze()
                        attn_weights[:, indices, :, :] = importance_mask[:, indices, :, :]
                        attn_weights = attn_weights + attention_mask
                    else:
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights = attn_weights + attention_mask

                else:
                    raise ValueError(f"Unknown eval mode {evalmode}")
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        else:
            if self.flash_attn:
                if self.layer_idx > 0:
                    # Token hit-rates cannot be calculated if using flash attention.
                    self.tok_hit_acc = 0
                    q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                    k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                    q_importance_tensor = q_importance_tensor.view(bsz, self.num_heads, q_len, self.dDash)
                    k_importance_tensor = k_importance_tensor.view(bsz, self.num_heads, key_len, self.dDash)
                    device_index = query_states.device.index
                    with torch.cuda.device(device_index):
                        attn_output, mse_loss = attention_mse_loss(query_states.contiguous().to(torch.float16),
                                                                    key_states.contiguous().to(torch.float16),
                                                                    value_states.contiguous().to(torch.float16),
                                                                    q_importance_tensor.contiguous().to(torch.float16),
                                                                    k_importance_tensor.contiguous().to(torch.float16), 
                                                                    True
                                                                    )
                    self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = 0, 0, 0
                    attn_output = attn_output.to(query_states.dtype)
                    if not torch.isnan(mse_loss):
                        self.msemagn_loss = mse_loss
                    else:
                        raise ValueError(f"NaN loss detected: {mse_loss}")
                else:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, is_causal=True)
                    # attn_output = attention(query_states.contiguous().to(torch.float16), 
                    #                         key_states.contiguous().to(torch.float16),
                    #                         value_states.contiguous().to(torch.float16), True, 1.0 / math.sqrt(self.head_dim))
                    # attn_output = attn_output.to(query_states.dtype)
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)   
                if self.layer_idx > 0:
                    q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                    k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                    importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash) # [BH, Lq, Lk]
                    importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len) # [B, H, Lq, Lk]

                    if self.lfunc == "MSE":
                        # # Here, save the attn_weights[0, 0] as a heatmap 
                        # import matplotlib.pyplot as plt
                        # import seaborn as sns
                        # # virdis colors
                        # sns.heatmap(attn_weights[0, 0].detach().cpu().numpy(), cmap='viridis')
                        # # save it as fig
                        # plt.savefig(f"attn_weights_{self.layer_idx}.png", dpi=600)
                        # exit()

                        self.msemagn_loss = self.mseloss(attn_weights, importance_mask)
                        self.msemagn_loss = (self.msemagn_loss).mean(dim=(-1, -2))
                        self.msemagn_loss = self.msemagn_loss.mean()

                        if self.calc_hitrates:
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=importance_mask,
                                true_importance=attn_weights,
                                top_k_ratio=0.5
                            )

                    else:
                        raise ValueError(f"Unknown loss function {self.lfunc}")
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

        if self.layer_idx > 0:
            head_importance_tensor = self.producer.head_importances[:, :, :, self.layer_idx % self.producer_frequency].float().to(attn_output.device)
            attn_head_weights = attn_output.mean(dim=-1).permute(0, 2, 1)
            self.headmsemagn_loss = self.headmseloss(attn_head_weights, head_importance_tensor).mean()

            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = calculate_hit_metrics(
                    estimated_importance=head_importance_tensor,
                    true_importance=attn_head_weights,
                    top_k_ratio=0.5
                )
            
        checkeverytime = hasattr(self, 'test_with_thresholds')
        if checkeverytime:
            checkeverytime = self.test_with_thresholds
        if final_mask is not None:
            if self.effective_sparsity is None or checkeverytime:
                true_mask = final_mask + attention_mask
                num_deact = true_mask.bool().sum(dim=-1)                   # Number of tokens disabled.
                causally_deact = (attention_mask.bool()).sum(dim=-1).expand_as(num_deact)        # Number of tokens disabled causally anyway
                additional_deact = (num_deact - causally_deact)
                num_active = (~attention_mask.bool()).sum(dim=-1).expand_as(num_deact)    # Number of tokens active at this position if zero-sparsity
                effective_sparsity = 100 * (additional_deact.float() / num_active.float()).mean().item()
                self.effective_sparsity = effective_sparsity
                print("Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)

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

        if self.producer is None:
            try:
                q_importance, k_importance, past_key_value_sp = self.sparse_token_predictor(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    past_key_value=past_key_value_sp, 
                    use_cache=use_cache
                )
                head_importances, past_key_value_hp = self.sparse_head_predictor(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value_hp,
                    use_cache=use_cache
                )
                head_importances = head_importances.view(bsz, q_len, self.num_heads, self.num_hidden_layers) # [B L H N]
                q_len = attn_output.size(1)
                k_len = k_importance.size(-1)
            except:
                print(traceback.format_exc())
                import pdb; pdb.set_trace()

            self.q_importance = q_importance
            self.k_importance = k_importance

            if self.head_importances is None:
                self.head_importances = head_importances
            else:
                self.head_importances = torch.cat([self.head_importances, head_importances], dim=1)


        if use_cache:
            if self.producer is None: # This is the producer layer
                past_key_value.update_predictors(past_key_value_sp, past_key_value_hp)

        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def convert_kvcache_experimental(model, config, producer_frequency, heavy_const=256, group_factor=8, label_bits=4):
    producer_layer = None
    producer_layer_device = None
    layer_counter = {'idx': 0}

    def recurse_convert(parent_module):
        nonlocal producer_layer
        nonlocal producer_layer_device
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                if layer_counter['idx'] % producer_frequency == 0:
                    new_module = LlamaAttentionExperimental(config).to(dtype).to(device)
                    producer_layer = new_module
                    producer_layer_device = device
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
    producer_layer = producer_layer.to(producer_layer_device)
    return model

def convert_llama_channel_config_experimental(model, channel_config, selected_channel="k"):
    selected_channel = "." + selected_channel + "_proj"
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttentionExperimental):
            device = next(module.parameters()).device
            module.sorted_channel = torch.tensor(channel_config[name + selected_channel]).to(device)

    return model