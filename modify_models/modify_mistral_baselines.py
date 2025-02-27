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

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, MistralConfig, MistralSdpaAttention, MistralRotaryEmbedding

from utils import repeat_kv, sorted_index_to_mask
from utils import calculate_hit_metrics
from transformers.cache_utils import DynamicCache

class BaselineDynamicCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.h2o_importance = None
    
    def update(self, key_states, value_states, layer_idx):
        # First update the base cache
        key_states, value_states = super().update(key_states, value_states, layer_idx)
        return key_states, value_states

    def update_h2o_importance(self, h2o_importance):
        self.h2o_importance = h2o_importance
    
    def get_h2o_importance(self):
        return self.h2o_importance


class MistralAttentionExperimental(nn.Module):
    def __init__(self, config: MistralConfig, producer=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.inference_mode = False
        self.producer = producer
        self.layer_idx = layer_idx
        self.token_sparse_method = None
        self.sparse_aggression = None
        self.pruneax = None
        self.init_token_importance = None
        self.predictor_type = None
        self.stream_llm_start_size = None
        self.phead_scale = None
        self.dDash = None
        self.intdim = None
        self.oproj = None
        self.ll_six = None
        self.olayer = None
        self.add_attn = None
        self.ilayer = None
        self.min_sparse_index = None
        self.no_pred_causal_mask = None
        self.effective_sparsity = None
        self.replace_attention = None
        self.post_proj_causal = None
        self.pred_hid_size = self.hidden_size
        self.num_tok_per_page = None
        if self.layer_idx > 0:
            # MSELoss
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.celoss = nn.CrossEntropyLoss(reduction='none')
            self.kldivloss = torch.nn.KLDivLoss(reduction='none')
        
        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MistralRotaryEmbedding(
            self.config
        )
        
    def update_predictor(self):
        pass
    
    def generate_ll_six(self, q_len):
        ll_six = []
        for curr_l in range(1, q_len+1):
            mt = [0,1,2,3] + list(range(4, curr_l, 1))[::-1]
            remmt = list(set(list(range(q_len))) - set(mt))
            mt = mt + remmt
            ll_six.append(mt)
        self.ll_six = torch.tensor(ll_six).to(self.q_proj.weight.device)

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
            

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[DynamicCache, BaselineDynamicCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[DynamicCache, BaselineDynamicCache]]]:
        bsz, q_len, _ = hidden_states.size()
        # convert hidden_states to same dtype as self.q_proj.weight
        # hidden_states = hidden_states.to(self.q_proj.weight.dtype)

        if past_key_value is not None and not isinstance(past_key_value, BaselineDynamicCache):
            if isinstance(past_key_value, DynamicCache):
                assert past_key_value.get_seq_length() == 0, "If past_key_value is DynamicCache, then it must be empty"
            past_key_value = BaselineDynamicCache()

        if q_len != 1: # this is prefill stage for first token output, reset self.token_mask
                       # further, this should guarantee that token_mask is always assigned, as its always prefill first.
            self.token_mask = None


        # if self.config.num_hidden_layers != 32:
        #     gc.collect()
        #     torch.cuda.empty_cache()

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except Exception as e:
            import pdb; pdb.set_trace()
        

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(q_len * self.sparse_aggression)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            h2o_importance_history = past_key_value.get_h2o_importance()
        else:
            h2o_importance_history = None
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # AHMED: Modified this to use the newer version.
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        kv_seq_len = key_states.shape[-2]

        final_mask = None
        # past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)

        # Ahmed Modification. Always set an attention_mask
        # Create causal mask if attention_mask is None
        if attention_mask is None:
            # Create causal mask
            # [bsz, 1, q_len, kv_seq_len]
            causal_mask = torch.ones((bsz, 1, q_len, kv_seq_len), device=hidden_states.device, dtype=torch.bool)
            causal_mask = causal_mask.triu(diagonal=1)  # Upper triangular part
            attention_mask = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
            attention_mask.masked_fill_(causal_mask, float("-inf"))


        assert self.head_dim % self.group_factor == 0
        attn_o_precalc = False
        min_sparse_index = self.min_sparse_index
        with torch.no_grad():
            if evalmode == "dense":
                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode in ["oracle", "random", "init_oracle", "lookahead_oracle"]:
                oracle_attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                oracle_attn_weights = oracle_attn_weights + attention_mask
                oracle_attn_weights = nn.functional.softmax(oracle_attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                importance_mask = oracle_attn_weights.detach().float()
                importance_mask = torch.softmax(importance_mask, dim=-1, dtype=torch.float32)
                if evalmode == "random":
                    importance_mask = torch.softmax(torch.rand_like(importance_mask) + attention_mask, dim=-1, dtype=torch.float32)
                if evalmode in ["init_oracle", "lookahead_oracle"]:
                    if self.layer_idx > 0:
                        importance_mask = self.producer.init_token_importance
                if evalmode == "oracle":
                    save_importance_mask = importance_mask.detach().float() # [B, H, L, L]
                    save_importance_mask = save_importance_mask.permute(0, 2, 3, 1) # [B, L, L, H]
                else:
                    save_importance_mask = importance_mask
                if self.layer_idx > 0:
                    if self.sparse_aggression < 1:
                        _, sorted_indices = importance_mask.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                        sorted_indices = sorted_indices[:, :, -q_len:, :]
                        mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression)
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        final_mask = mask_tensor
                        attn_wt_shape = attn_weights.shape
                        attn_weights = attn_weights + mask_tensor + attention_mask
                        if attn_weights.shape != attn_wt_shape:
                            import pdb; pdb.set_trace()
                        assert attn_weights.shape == attn_wt_shape, f"Shape mismatch {attn_weights.shape} {attn_wt_shape} due to MT {mask_tensor.shape}"
                    else:
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "streamingLLM":
                if self.layer_idx > 0:
                    # if self.ll_six is None or self.ll_six.size(-1) != q_len:
                    self.generate_ll_six(key_len)
                    ll_six = self.ll_six
                    # here, it should be q_len, key_len i think. -- init max size and then pick
                    sorted_indices = ll_six.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_heads, key_len, key_len).to(query_states.device)
                    sorted_indices = sorted_indices[:, :, -q_len:, :]
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression)
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    final_mask = mask_tensor
                    attn_weights = attn_weights + mask_tensor + attention_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "oracle_grouped":
                # **This buckets in the embedding dimension, quest buckets in the token dimension**
                if self.layer_idx > 0:
                    grouped_query = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
                    grouped_key = key_states.reshape(bsz, self.num_heads, kv_seq_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
                    # Materializes full [B H L L], but compute is reduced.
                    grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)
                    grouped_attn_weights = torch.softmax(grouped_attn_weights + attention_mask, dim=-1, dtype=torch.float32)
                    _, sorted_indices = grouped_attn_weights.sort(dim=-1, descending=True)
                    sorted_indices = sorted_indices[:, :, -q_len:, :]
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression)
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    final_mask = mask_tensor
                    attn_weights = attn_weights + mask_tensor + attention_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "snapkv_old":
                if self.layer_idx > 0:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()
                    combined_bh = bsz * num_heads
                    attn_weights_2d = attn_weights.view(combined_bh, q_len, kv_seq_len)
                    obs_size = 16
                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    final_mask_2d = final_mask.view(combined_bh, q_len, kv_seq_len)
                    # ---------------------------------------------------------------------
                    # Build one big causal_mask for entire seq_len x seq_len:
                    #    causal_mask[i, j] = 0 if j <= i, else -inf
                    # ---------------------------------------------------------------------
                    if hasattr(self, 'causal_mask'):
                        causal_mask = self.causal_mask
                    else:
                        causal_mask = torch.full((q_len, q_len), float('-inf'), device=attn_weights.device)
                        for r in range(q_len):
                            causal_mask[r, : r + 1] = 0.0  # lower-tri (including diagonal) = 0
                        self.causal_mask = causal_mask


                    for i in range(1, q_len):
                        max_budget = max(int((i+1) * self.sparse_aggression), min_sparse_index)
                        # Reset this row to -inf each iteration, so we only keep fresh selections
                        final_mask_2d[:, i, :] = float('-inf')

                        obs_start = max(0, i - obs_size + 1)
                        obs_length = i - obs_start + 1
                        prefix_length = obs_start  # everything before the observation window

                        if prefix_length > 0:
                            # Sum attention from obs window to prefix
                            attn_slice = attn_weights_2d[:, obs_start:(i+1), 0:obs_start]  # shape [CBH, obs_length, prefix_length]
                            temp_mask = causal_mask[obs_start : i + 1, : prefix_length]  # [obs_length, prefix_length]
                            attn_slice = attn_slice + temp_mask.unsqueeze(0)  # shape [CBH, obs_length, prefix_length]
                            attn_slice = F.softmax(attn_slice, dim=-1)
                            attn_agg = attn_slice.sum(dim=1)  # [CBH, prefix_length]

                            # Optional pooling
                            kernel_size = 5
                            pooled = F.avg_pool1d(
                                attn_agg.unsqueeze(1),
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                stride=1
                            ).squeeze(1)  # shape [CBH, prefix_length]

                            # Decide how many prefix tokens to keep
                            num_prefix_to_keep = max_budget - obs_length
                            num_prefix_to_keep = min(max(num_prefix_to_keep, 0), prefix_length)

                            if num_prefix_to_keep > 0:
                                topk_indices = pooled.topk(num_prefix_to_keep, dim=1).indices
                                # Unmask those top prefix positions
                                row_idx = torch.arange(combined_bh, device=attn_weights.device).unsqueeze(-1)
                                final_mask_2d[row_idx, i, topk_indices] = 0.0
                        final_mask_2d[:, i, obs_start:(i+1)] = 0.0
                        # Important code snippet to ensure consistency.
# for numel in [1, 16, 32, 64, 128, 256, 512, 768, 966, 968, 1000, 1020, 1023]: print("Seq-idx: ", numel, "Sparsity: ", 100*(1 - torch.logical_not(final_mask[0, 0, numel].bool()).sum()/(numel + 1)).item(), "%")
                    final_mask = final_mask_2d.view(bsz, num_heads, q_len, kv_seq_len)
                    final_mask[:, :, :, :min_sparse_index] = 0.0
                    attn_weights = attn_weights + final_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "snapkv":
                """
                Incremental SnapKV approach that mimics 'h2o_true':
                - We keep an active set of tokens of max size 'max_budget'.
                - Once a token is pruned, we never pick it again.
                - We use a SnapKV-like metric (aggregated attention from a local observation window)
                    to decide which tokens remain in the active set.
                """
                if self.layer_idx > 0:
                    # 1) Standard scaled-dot product attention
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()

                    # Flatten [batch, head] for simpler indexing
                    combined_bh = bsz * num_heads
                    attn_weights_2d = attn_weights.view(combined_bh, q_len, kv_seq_len)

                    # 2) Build big causal mask once (for the entire seq)
                    #    shape [q_len, kv_seq_len], or broadcastable to [CBH, q_len, kv_seq_len].
                    if not hasattr(self, "causal_mask") or self.causal_mask.shape[0] < q_len:
                        # Build new mask
                        big_mask = torch.full((q_len, kv_seq_len), float('-inf'), device=attn_weights.device)
                        for row in range(q_len):
                            # Unmask [0..row], mask [row+1.. end]
                            big_mask[row, :row+1] = 0.0
                        self.causal_mask = big_mask
                    else:
                        big_mask = self.causal_mask[:q_len, :kv_seq_len]  # slice if larger

                    # 3) Apply that mask (broadcast to [CBH, q_len, kv_seq_len]) and do one big softmax
                    #    This ensures all queries see only valid (non-future) tokens
                    attn_weights_2d = attn_weights_2d + big_mask.unsqueeze(0)  # shape [CBH, q_len, kv_seq_len]
                    attn_weights_2d = F.softmax(attn_weights_2d, dim=-1)       # shape [CBH, q_len, kv_seq_len]

                    # 4) Build prefix-sums along the query dimension => shape [CBH, q_len, kv_seq_len]
                    prefix_sums_2d = attn_weights_2d.cumsum(dim=1)

                    # 5) We'll keep a final_mask for each step i
                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    final_mask_2d = final_mask.view(combined_bh, q_len, kv_seq_len)

                    # 6) Active set arrays
                    #    We'll choose some initial "max_budget" based on full q_len, or adapt each step.
                    max_budget = max(int(q_len * self.sparse_aggression), min_sparse_index)
                    active_tokens = torch.full((combined_bh, max_budget), 0, dtype=torch.long, device=attn_weights.device)
                    active_counts = torch.ones(combined_bh, dtype=torch.long, device=attn_weights.device)

                    # 7) For i=0, unmask token 0
                    final_mask_2d[:, 0, 0] = 0.0

                    # 8) Define observation window
                    obs_size = 16

                    # 9) Main loop
                    for i in range(1, q_len):
                        # Re-compute the local step budget
                        step_budget = max(int((i + 1) * self.sparse_aggression), min_sparse_index)

                        # Identify obs_start
                        obs_start = max(0, i - obs_size + 1)
                        obs_length = i - obs_start + 1
                        prefix_length = obs_start

                        # 9A) aggregator for each token t in [0.. i] is sum_{q in [obs_start..i]} attn_weights_2d[row, q, t].
                        #     Using prefix sums, aggregator[row, t] = prefix_sums_2d[row, i, t] - prefix_sums_2d[row, obs_start-1, t].
                        # We'll write this into a buffer "aggregator"
                        aggregator = torch.zeros(combined_bh, i + 1, device=attn_weights.device)

                        if obs_start > 0:
                            aggregator[:, : (i + 1)] = prefix_sums_2d[:, i, : (i + 1)] - prefix_sums_2d[:, obs_start - 1, : (i + 1)]
                        else:
                            aggregator[:, : (i + 1)] = prefix_sums_2d[:, i, : (i + 1)]

                        # (Optional) apply SnapKV 1D pooling for clustering => aggregator pooling
                        # Let's do a simple example with avg_pool1d across tokens dimension => [CBH, i+1].
                        # We'll pool over dimension -1, so we need aggregator => shape [CBH, 1, i+1].
                        # Then pick from aggregator_pooled. This is just an example--adjust as needed.
                        kernel_size = 5
                        aggregator_reshaped = aggregator[:, : (i + 1)].unsqueeze(1)  # [CBH, 1, i+1]
                        aggregator_pooled = F.avg_pool1d(aggregator_reshaped, kernel_size=kernel_size,
                                                        stride=1, padding=kernel_size // 2)
                        aggregator_pooled = aggregator_pooled.squeeze(1)  # [CBH, i+1]

                        # 9B) aggregator for new token i => aggregator_pooled[row, i]
                        new_token_importance = aggregator_pooled[:, i].unsqueeze(-1)  # [CBH, 1]

                        # 9C) Insert or replace in the active set
                        can_add = active_counts < step_budget
                        add_indices = can_add.nonzero(as_tuple=False).squeeze(-1)
                        active_tokens[add_indices, active_counts[add_indices]] = i
                        active_counts[add_indices] += 1

                        cannot_add = ~can_add
                        if cannot_add.any():
                            replace_indices = cannot_add.nonzero(as_tuple=False).squeeze(-1)
                            current_active = active_tokens[replace_indices, :step_budget]
                            # aggregator for those tokens => aggregator_pooled[row, token]
                            # gather => shape [?, step_budget]
                            row_imps = aggregator_pooled[replace_indices].gather(1, current_active)
                            # find min
                            min_vals, min_idxs = torch.min(row_imps, dim=1, keepdim=True)
                            new_imps = new_token_importance[replace_indices]
                            should_replace = new_imps > min_vals
                            rows_to_replace = replace_indices[should_replace.squeeze(1)]
                            pos_to_replace = min_idxs[should_replace.squeeze(1)].squeeze(1)
                            active_tokens[rows_to_replace, pos_to_replace] = i

                        # 9D) Construct final_mask row i
                        final_mask_2d[:, i, :] = float('-inf')

                        positions = torch.arange(max_budget, device=attn_weights.device).unsqueeze(0)
                        valid_positions = positions < active_counts.unsqueeze(1)
                        valid_rows = valid_positions.nonzero(as_tuple=True)[0]
                        valid_token_positions = valid_positions.nonzero(as_tuple=True)[1]
                        valid_tokens = active_tokens[valid_rows, valid_token_positions]
                        final_mask_2d[valid_rows, i, valid_tokens] = 0.0

                        # Also unmask the obs window
                        # Only done in wikitext eval, otherwise we shouldn't do this.
                        # too low sparsity in downstream eval
                        # if q_len == 1024:
                        #     final_mask_2d[:, i, obs_start : i + 1] = 0.0

                    # 10) Reshape final_mask
                    final_mask = final_mask_2d.view(bsz, num_heads, q_len, kv_seq_len)
                    final_mask[:, :, :, :min_sparse_index] = 0.0
                    attn_weights = attn_weights + final_mask
                else:
                    # layer_idx == 0 => no pruning
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "h2o_true":
                if self.layer_idx > 0:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()
                    # Initialize final mask with -inf
                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    # Reshape for parallel processing over heads
                    combined_bh = bsz * num_heads
                    final_mask = final_mask.view(combined_bh, q_len, kv_seq_len)
                    # Determine maximum kv_cache_budget
                    max_budget = max(int(key_len * self.sparse_aggression), min_sparse_index)
                    
                    # If q_len == 1, this is a decode step (adding one token at a time)
                    if q_len == 1 and h2o_importance_history is not None:
                        # Load existing active_tokens and active_counts
                        prev_active_tokens, prev_active_counts = h2o_importance_history
                        # if prev_active_tokens.size(-1) is smaller than max_budget,
                        # make it bigger 
                        if prev_active_tokens.size(-1) <= max_budget:
                            prev_active_tokens = torch.cat(
                                [prev_active_tokens, torch.full_like(prev_active_tokens, 0)], dim=-1
                            )[..., :max_budget + 1]

                        # prev_active_tokens: [combined_bh, max_budget]
                        # prev_active_counts: [combined_bh]

                        # Let's say total_tokens_so_far = stored_key.size(2) + q_len
                        total_tokens_so_far = kv_seq_len
                        new_token_id = total_tokens_so_far - 1

                        # Current row weights up to new_token_id
                        # For q_len==1, i=0 is the new token. 
                        # Since q_len=1 means just one query position, attn_weights shape: [B,H,1,kv_seq_len]
                        # Let's flatten for easier indexing:
                        attn_weights_2d = attn_weights.view(combined_bh, 1, kv_seq_len)
                        row_weights = attn_weights_2d[:, 0, :new_token_id+1]  # [combined_bh, new_token_id+1]

                        kv_cache_budget = torch.full((combined_bh,),
                                                    max(min_sparse_index, int((new_token_id + 1) * self.sparse_aggression)),
                                                    device=attn_weights.device)

                        can_add = prev_active_counts < kv_cache_budget
                        add_indices = can_add.nonzero(as_tuple=False).squeeze(-1)

                        # if self.layer_idx == 1:
                        #     import pdb; pdb.set_trace()
                        prev_active_tokens[add_indices, prev_active_counts[add_indices] - 1] = new_token_id
                        prev_active_counts[add_indices] += 1

                        cannot_add = ~can_add
                        if cannot_add.any():
                            replace_indices = cannot_add.nonzero(as_tuple=False).squeeze(-1)
                            max_k = kv_cache_budget[replace_indices].max().item()
                            current_active = prev_active_tokens[replace_indices, :max_k]
                            active_importances = row_weights[replace_indices].gather(1, current_active)
                            min_vals, min_idxs = torch.min(active_importances, dim=1, keepdim=True)
                            new_importance = row_weights[replace_indices, new_token_id].unsqueeze(1)
                            should_replace = new_importance > min_vals
                            rows_to_replace = replace_indices[should_replace.squeeze(1)]
                            pos_to_replace = min_idxs[should_replace.squeeze(1)].squeeze(1)
                            prev_active_tokens[rows_to_replace, pos_to_replace] = new_token_id


                        # Construct final_mask for this single step
                        final_mask = torch.full_like(attn_weights, float('-inf'))
                        final_mask = final_mask.view(combined_bh, 1, kv_seq_len)
                        valid_positions = torch.arange(max_budget, device=attn_weights.device).unsqueeze(0) < prev_active_counts.unsqueeze(1)
                        valid_rows = valid_positions.nonzero(as_tuple=True)[0]
                        valid_token_positions = valid_positions.nonzero(as_tuple=True)[1]
                        valid_tokens = prev_active_tokens[valid_rows, valid_token_positions]
                        final_mask[valid_rows, 0, valid_tokens] = 0.0
                        final_mask = final_mask.view(bsz, num_heads, 1, kv_seq_len)

                        attn_weights = attn_weights + final_mask

                        # Update h2o_importance_history with the latest active_tokens and counts
                        h2o_importance_history = (prev_active_tokens, prev_active_counts)

                    else:
                        # This is the prefill scenario: we process the entire q_len at once.
                        # We'll run the original logic to figure out active_tokens for all steps.
                        final_mask = torch.full_like(attn_weights, float('-inf'))
                        final_mask_2d = final_mask.view(combined_bh, q_len, kv_seq_len)
                        attn_weights_2d = attn_weights.view(combined_bh, q_len, kv_seq_len)

                        active_tokens = torch.full((combined_bh, max_budget), 0, dtype=torch.long, device=attn_weights.device)
                        active_tokens[:, 0] = 0
                        active_counts = torch.ones(combined_bh, dtype=torch.long, device=attn_weights.device)

                        final_mask_2d[:, 0, 0] = 0.0

                        for i in range(1, q_len):
                            kv_cache_budget = torch.full((combined_bh,),
                                                        max(min_sparse_index, int((i + 1) * self.sparse_aggression)),
                                                        device=attn_weights.device)
                            row_weights = attn_weights_2d[:, i, :i + 1]
                            can_add = active_counts < kv_cache_budget
                            add_indices = can_add.nonzero(as_tuple=False).squeeze(-1)
                            active_tokens[add_indices, active_counts[add_indices]] = i
                            active_counts[add_indices] += 1

                            cannot_add = ~can_add
                            if cannot_add.any():
                                replace_indices = cannot_add.nonzero(as_tuple=False).squeeze(-1)
                                max_k = kv_cache_budget[replace_indices].max().item()
                                current_active = active_tokens[replace_indices, :max_k]
                                active_importances = row_weights[replace_indices].gather(1, current_active)
                                min_vals, min_idxs = torch.min(active_importances, dim=1, keepdim=True)
                                new_importance = row_weights[replace_indices, i].unsqueeze(1)
                                should_replace = new_importance > min_vals
                                rows_to_replace = replace_indices[should_replace.squeeze(1)]
                                pos_to_replace = min_idxs[should_replace.squeeze(1)].squeeze(1)
                                active_tokens[rows_to_replace, pos_to_replace] = i

                            valid_positions = torch.arange(max_budget, device=attn_weights.device).unsqueeze(0) < active_counts.unsqueeze(1)
                            valid_rows = valid_positions.nonzero(as_tuple=True)[0]
                            valid_token_positions = valid_positions.nonzero(as_tuple=True)[1]
                            valid_tokens = active_tokens[valid_rows, valid_token_positions]
                            final_mask_2d[valid_rows, i, valid_tokens] = 0.0

                        final_mask = final_mask_2d.view(bsz, num_heads, q_len, kv_seq_len)
                        attn_weights = attn_weights + final_mask
                        # After processing full q_len (prefill), store active_tokens and active_counts into h2o_importance_history
                        h2o_importance_history = (active_tokens, active_counts)
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

            elif evalmode == "quest":
                if self.layer_idx > 0:
                    num_tok_per_page = self.num_tok_per_page
                    num_full_pages = q_len // num_tok_per_page
                    if num_full_pages > 0:
                        remaining_tokens = q_len % num_tok_per_page
                        total_pages = num_full_pages + (1 if remaining_tokens > 0 else 0)
                        key_states_full = key_states[:, :, :num_full_pages * num_tok_per_page]
                        key_states_full = key_states_full.transpose(-2, -1).view(
                            bsz, self.num_heads, -1, num_full_pages, num_tok_per_page
                        )
                        key_states_full = key_states_full.amax(dim=-1)  # Take the maximum in each chunk
                        if remaining_tokens > 0:
                            key_states_partial = key_states[:, :, num_full_pages * num_tok_per_page:]
                            pad_size = num_tok_per_page - remaining_tokens
                            key_states_partial = F.pad(key_states_partial, (0, 0, 0, pad_size), value=torch.finfo(key_states.dtype).min)
                            key_states_partial = key_states_partial.transpose(-2, -1).view(
                                bsz, self.num_heads, -1, 1, num_tok_per_page
                            ).amax(dim=-1)  # Take the maximum in the partial page
                            key_states_to_page = torch.cat([key_states_full, key_states_partial], dim=-1)
                            num_pages = num_full_pages +  1
                        else:
                            key_states_to_page = key_states_full  # [B, H, key_len_new, num_full_pages, 2]
                            num_pages = num_full_pages

                        sign = (query_states > 0) + (~(query_states > 0)) * -1
                        key_states_signed = key_states * sign
                        query_states_signed = query_states * sign
                        key_states_reshaped = key_states_to_page.view(bsz, self.num_heads, -1, num_pages)  # Reshape for interaction
                        quest_page_weights = torch.matmul(query_states_signed, key_states_reshaped) / math.sqrt(self.head_dim)
                        quest_page_weights_repeated = quest_page_weights.repeat_interleave(
                            num_tok_per_page, dim=-1
                        )  # [B, H, q_len, key_len_new * num_tok_per_page]
                        quest_page_weights_repeated = quest_page_weights_repeated[..., :key_len]  # Trim excess padding
                        sorted_indices = torch.argsort(
                            quest_page_weights_repeated + attention_mask.view(1, 1, q_len, key_len).float(),
                            dim=-1,
                            descending=True,
                        )  # [B, H, q_len, key_len]
                    else:
                        # initialize random torch tensor [bsz, num_heads, q_len, key_len]
                        importance_mask = torch.softmax(torch.rand(bsz, self.num_heads, q_len, q_len).to(query_states.device) + attention_mask, dim=-1, dtype=torch.float32)
                        # No quest-token mask can exist, so drop tokens randomly.
                        _, sorted_indices = importance_mask.sort(dim=-1, descending=False)

                    sorted_indices = sorted_indices[:, :, -q_len:, :]
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression)
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    final_mask = mask_tensor
                    attn_weights = attn_weights + mask_tensor + attention_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            else:
                raise ValueError(f"Unknown eval mode {evalmode}")

        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if final_mask is not None:
            if self.effective_sparsity is None:
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

        if evalmode == "init_oracle":
            if self.layer_idx == 0:
                self.init_token_importance = torch.softmax(attn_weights.detach().float() + attention_mask, dim=-1, dtype=torch.float32)

        if evalmode == "lookahead_oracle":
            if self.layer_idx == 0:
                self.init_token_importance = torch.softmax(attn_weights.detach().float() + attention_mask, dim=-1, dtype=torch.float32)
            else:
                self.producer.init_token_importance = attn_weights.detach().float().sum(dim=2).unsqueeze(dim=2)

        if self.inference_mode:
            if "lookahead" in evalmode:
                if self.layer_idx == 0:
                    self.actmagn_masklist[self.layer_idx] = attn_weights.detach().float().sum(dim=2).unsqueeze(dim=2)
                else:
                    self.producer.actmagn_masklist[self.layer_idx] = attn_weights.detach().float().sum(dim=2).unsqueeze(dim=2)
                    
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, -1, self.hidden_size)


        attn_output = self.o_proj(attn_output)

        if use_cache:
            if evalmode == "h2o_true":
                past_key_value.update_h2o_importance(h2o_importance_history)
        else:
            past_key_value = None

        if not output_attentions:
            attn_weights = None

        # if self.config.num_hidden_layers != 32:
        #     indices, query_states, key_states, mask_t, importance_mask = None, None, None, None, None
        #     gc.collect()
        #     torch.cuda.empty_cache()
        
        return attn_output, attn_weights

def convert_kvcache_experimental(model, config, producer_frequency, heavy_const=256, group_factor=8, label_bits=4):
    producer_layer = None
    layer_counter = {'idx': 0}

    def recurse_convert(parent_module):
        nonlocal producer_layer
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, MistralSdpaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                if layer_counter['idx'] % producer_frequency == 0:
                    new_module = MistralAttentionExperimental(config).to(dtype).to(device)
                    producer_layer = new_module
                else:
                    new_module = MistralAttentionExperimental(
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
                    print(f"Converted Producer layer '{name}' to MistralAttentionExperimental at layer index {layer_counter['idx']}")
                else:
                    print(f"Converted layer '{name}' to MistralAttentionExperimental at layer index {layer_counter['idx']}")
                parent_module._modules[name] = new_module
                layer_counter['idx'] += 1
    recurse_convert(model)
    return model
