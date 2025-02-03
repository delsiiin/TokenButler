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
from transformers.cache_utils import DynamicCache

from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss

# torch.backends.cuda.enable_flash_sdp(enabled=True)
# torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

class PredictorDynamicCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.predictor_cache = None
        self.head_predictor_cache = None  # Add this for Head Importance Predictor
    
    def update(self, key_states, value_states, layer_idx):
        # First update the base cache
        key_states, value_states = super().update(key_states, value_states, layer_idx)
        return key_states, value_states

    def update_predictors(self, predictor_cache, head_predictor_cache):
        self.predictor_cache = predictor_cache
        self.head_predictor_cache = head_predictor_cache

    def get_predictor_cache(self):
        return self.predictor_cache
    
    def get_head_predictor_cache(self):
        return self.head_predictor_cache


class TokenImportancePredictorAttentive(nn.Module):
    def __init__(self, config, pred_hid_size, num_heads, num_hidden_layers, dDash, intdim, \
                 attn_reduce_factor, dropout=0.1):
        """
        Optimized Token Importance Predictor with parallel Q-K projections and simplified mapping.
        
        Args:
            config: Configuration object containing model parameters.
            pred_hid_size (int): Hidden size for the predictor's attention layer.
            num_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of transformer layers to predict.
            dropout (float): Dropout probability.
            q_downscale (int): Factor to downscale the Q dimension for efficiency.
            intermediate_dim (int): Intermediate dimension for non-linear transformations in projections.
        """
        super().__init__()
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4)
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False
        assert pred_hid_size % (num_heads * 4) == 0, "pred_hid_size must be divisible by num_heads * 4."

        # Reduce the hidden size for attention computations
        self.hidden_size_reduced = self.hidden_size // self.attn_reduce_factor  # For example, reduce to 1/4th
        assert self.hidden_size_reduced % self.num_heads == 0, "Reduced hidden size must be divisible by num_heads"
        self.attn_head_dim = self.hidden_size_reduced // self.num_heads

        # Input projection to reduce hidden size
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size_reduced, bias=False)

        # Query, Key, Value projections for attention
        self.q_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.k_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.v_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        # Output projection to restore hidden size
        # self.o_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)

        # LayerNorm and Feed-forward network
        self.norm1 = nn.LayerNorm(self.hidden_size_reduced)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.ffn_hidden_size = 2 * self.hidden_size_reduced  # Typical FFN hidden size
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size_reduced, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.hidden_size),
            nn.Dropout(self.dropout)
        )
        # Define Q and K projection layers for all layers in parallel with non-linearity[]
        # Output shape: [B, L, N * H * D']
        self.q_proj_importance = nn.Sequential(
            nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, num_hidden_layers * num_heads * self.dDash, bias=False)
        )
        self.k_proj_importance = nn.Sequential(
            nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, num_hidden_layers * num_heads * self.dDash, bias=False)
        )

        # Initialize rotary positional embeddings
        self._init_rope()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize in_proj_weight
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

                # Initialize out_proj
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def _init_rope(self):
        # Rotary embedding for attention layer
        self.rotary_emb_attn = LlamaRotaryEmbedding(
            self.attn_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        # Rotary embedding for importance projection
        self.rotary_emb_importance = LlamaRotaryEmbedding(
            self.dDash,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        """
        Forward pass for the Optimized Token Importance Predictor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, HQ].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, 1, L] or [B, 1, L, L].
            position_ids (torch.Tensor, optional): Position IDs.
            past_key_value (tuple, optional): Past key and value states.
            use_cache (bool, optional): Whether to use cache.
        
        Returns:
            torch.Tensor: Importance scores of shape [B, N, H, L, L].
        """
        B, L, E = hidden_states.size()
        if past_key_value is None:
            past_key_value = {}
        # if L == 1:
        #     import pdb; pdb.set_trace()
        past_primary = past_key_value.get('primary', None)
        # Reduce hidden size
        hidden_states_reduced = self.input_proj(hidden_states)  # [B, L, hidden_size_reduced]
        # Compute q, k, v for attention
        q = self.q_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        k = self.k_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        v = self.v_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        # Reshape q, k, v to [B, num_heads, L, attn_head_dim]
        q = q.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        k = k.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        v = v.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        # Compute kv_seq_len before concatenation
        if past_primary is not None:
            past_L = past_primary[0].shape[2]
            kv_seq_len = past_L + L
        else:
            kv_seq_len = L
        
        # Apply rotary positional embeddings based on kv_seq_len
        cos, sin = self.rotary_emb_attn(v, position_ids)
        
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(B, kv_seq_len)
        
        if past_primary is not None:
            # Concatenate past k and v
            k = torch.cat([past_primary[0], k], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
            v = torch.cat([past_primary[1], v], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
        
        # Apply rotary embeddings after concatenation
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Update cache if use_cache is True
        if use_cache:
            past_key_value['primary'] = (k.detach(), v.detach())

        if self.flash_attn:
            sm_scale = 1.0 / math.sqrt(self.attn_head_dim)
            attn_output = attention(q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16), True, sm_scale).to(q.dtype)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        # # attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        # sm_scale = 1.0 / math.sqrt(self.attn_head_dim)
        # attn_output = attention(q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16), True, sm_scale).to(q.dtype)
        attn_output = attn_output.to(q.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size_reduced)
        attn_output = self.norm1(attn_output)
        ffn_output = self.ffn(attn_output)
        hidden_states = hidden_states + ffn_output

        B, L, E = hidden_states.size()
        # Importance projections
        H = self.num_heads
        N = self.num_hidden_layers

        # Shape after projection: [B, L, N * H * D']
        q_importance = self.q_proj_importance(hidden_states)  # [B, L, N * H * D']
        k_importance = self.k_proj_importance(hidden_states)  # [B, L, N * H * D']

        # Reshape and permute to [B, N, H, L, D']
        q_importance = q_importance.view(B, L, N, H, self.dDash).permute(0, 2, 3, 1, 4).contiguous()  # [B, N, H, L, D']
        k_importance = k_importance.view(B, L, N, H, self.dDash).permute(0, 2, 3, 1, 4).contiguous()  # [B, N, H, L, D']

        # Flatten N and H for efficient computation
        # New shape: [B * N * H, L, D']
        q_importance = q_importance.view(B * N * H, L, self.dDash)  # [BNH, L, D']
        k_importance = k_importance.view(B * N * H, L, self.dDash)  # [BNH, L, D']

        # Retrieve past keys and values for importance attention
        past_importance = past_key_value.get('importance', None)

        # Concatenate past keys and values if cache is present
        if past_importance is not None:
            k_importance = torch.cat([past_importance['k'], k_importance], dim=1)  # [BNH, past_L + L, D']

        kv_seq_len = k_importance.size(1)

        # Apply rotary positional embeddings
        cos, sin = self.rotary_emb_importance(k_importance, position_ids)
        q_importance, k_importance = apply_rotary_pos_emb(q_importance, k_importance, cos, sin, position_ids)

        k_importance = k_importance.view(B * N * H, -1, self.dDash)  # [BNH, L, D']
        # Update cache if use_cache is True
        if use_cache:
            past_key_value['importance'] = {
                'k': k_importance.detach(),
            }
        q_importance = q_importance.view(B * H, N, L, self.dDash)  # [BH, N, L, D']
        k_importance = k_importance.view(B * H, N, kv_seq_len, self.dDash)  # [BH, N, L, D']
        return q_importance, k_importance, past_key_value



class HeadImportancePredictor(nn.Module):
    def __init__(self, config, pred_hid_size, num_heads, num_hidden_layers, dDash, intdim, \
                 attn_reduce_factor, dropout=0.1):
        """
        Optimized Token Importance Predictor with parallel Q-K projections and simplified mapping.
        
        Args:
            config: Configuration object containing model parameters.
            pred_hid_size (int): Hidden size for the predictor's attention layer.
            num_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of transformer layers to predict.
            dropout (float): Dropout probability.
            q_downscale (int): Factor to downscale the Q dimension for efficiency.
            intermediate_dim (int): Intermediate dimension for non-linear transformations in projections.
        """
        super().__init__()
        self.is_head_predictor = None
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4)
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False

        # Reduce the hidden size for attention computations
        self.hidden_size_reduced = self.hidden_size // self.attn_reduce_factor  # For example, reduce to 1/4th
        assert self.hidden_size_reduced % self.num_heads == 0, "Reduced hidden size must be divisible by num_heads"
        self.attn_head_dim = self.hidden_size_reduced // self.num_heads

        # Input projection to reduce hidden size
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size_reduced, bias=False)

        # Query, Key, Value projections for attention
        self.q_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.k_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.v_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        # Output projection to restore hidden size
        # self.o_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)

        # LayerNorm and Feed-forward network
        self.norm1 = nn.LayerNorm(self.hidden_size_reduced)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.ffn_hidden_size = 4 * self.hidden_size_reduced  # Typical FFN hidden size
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size_reduced, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.num_heads * self.num_hidden_layers),
        )

        # Initialize rotary positional embeddings
        self._init_rope()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize in_proj_weight
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

                # Initialize out_proj
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def _init_rope(self):
        # Rotary embedding for attention layer
        self.rotary_emb_attn = LlamaRotaryEmbedding(
            self.attn_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        # Rotary embedding for importance projection
        self.rotary_emb_importance = LlamaRotaryEmbedding(
            self.dDash,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        """
        Forward pass for the Optimized Token Importance Predictor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, HQ].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, 1, L] or [B, 1, L, L].
            position_ids (torch.Tensor, optional): Position IDs.
            past_key_value (tuple, optional): Past key and value states.
            use_cache (bool, optional): Whether to use cache.
        
        Returns:
            torch.Tensor: Importance scores of shape [B, N, H, L, L].
        """
        B, L, E = hidden_states.size()
        if past_key_value is None:
            past_key_value = {}
        # if L == 1:
        #     import pdb; pdb.set_trace()
        past_primary = past_key_value.get('primary', None)
        # Reduce hidden size
        hidden_states_reduced = self.input_proj(hidden_states)  # [B, L, hidden_size_reduced]
        # Compute q, k, v for attention
        q = self.q_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        k = self.k_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        v = self.v_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        # Reshape q, k, v to [B, num_heads, L, attn_head_dim]
        q = q.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        k = k.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        v = v.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        # Compute kv_seq_len before concatenation
        if past_primary is not None:
            past_L = past_primary[0].shape[2]
            kv_seq_len = past_L + L
        else:
            kv_seq_len = L
        
        # Apply rotary positional embeddings based on kv_seq_len
        cos, sin = self.rotary_emb_attn(v, position_ids)
        
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(B, kv_seq_len)
        
        if past_primary is not None:
            # Concatenate past k and v
            k = torch.cat([past_primary[0], k], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
            v = torch.cat([past_primary[1], v], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
        
        # Apply rotary embeddings after concatenation
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Update cache if use_cache is True
        if use_cache:
            past_key_value['primary'] = (k.detach(), v.detach())

        if self.flash_attn:
            sm_scale = 1.0 / math.sqrt(self.attn_head_dim)
            attn_output = attention(q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16), True, sm_scale).to(q.dtype)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_output = attn_output.to(q.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size_reduced)
        attn_output = self.norm1(attn_output)
        head_importances = self.ffn(attn_output)
        return head_importances, past_key_value
