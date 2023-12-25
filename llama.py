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

# 注释作者：YoungL

""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(padding_mask):      # YoungL：获取非pad数据   padding_mask维度：batch_size * seq_length（padding过的seq_length）
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)      # YoungL：获取batch中的每一条数据有多少token不是pad，就是对seq_length这个维度求和，因为pad的位置是0，原始文本位置是1
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()       # YoungL：将pad_mask打平并且返回非0元素的索引
    # YoungL：统计了每一条数据的长度之后，再打平。这样才能知道打平之后的indices怎么切分才能得到原来的每一条数据
    max_seqlen_in_batch = seqlens_in_batch.max().item()     # YoungL：获取当前batch中最长的数据
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):      # YoungL：RMS归一化
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))         # YoungL：初始化为全一矩阵
        self.variance_epsilon = eps

    def forward(self, hidden_states):                   # YoungL：batch_size * seq_length * hidden_size
        input_dtype = hidden_states.dtype                                       # YoungL：记录输入的数据类型
        hidden_states = hidden_states.to(torch.float32)                         # YoungL：将输入转换为float32用于后续计算
        variance = hidden_states.pow(2).mean(-1, keepdim=True)                  # YoungL：对所有元素求平方，在hidden_size维度上求均值
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)   # YoungL：均值之后再开方 与 输入的hidden_states对位相乘
        return self.weight * hidden_states.to(input_dtype)                      # YoungL：对位相乘


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):                  # YoungL：旋转位置编码
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim                                  # YoungL：位置向量的维度
        self.max_position_embeddings = max_position_embeddings      # YoungL：所需编码个数
        self.base = base                                            # YoungL：相位中的分母
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))            # YoungL：1 / (10000 ** (2*i / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)        # YoungL：将这个常数tensor保存到model.state_dict()中

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )   

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len                                                           # YoungL：记录position_ids的长度
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)         # YoungL：生成[0-max_seq_len_cached)的一维数组

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):              # YoungL：Llama decoder_layer中的前馈层
    def __init__(self, config):
        super().__init__()
        self.config = config                                                                # YoungL：参数配置
        self.hidden_size = config.hidden_size                                               # YoungL：attention层的输出
        self.intermediate_size = config.intermediate_size                                   # YoungL：第一层线性层的输出维度
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # YoungL：第一层线性层的gate层
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)      # YoungL：第一层线性变换层
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)    # YoungL：第二层线性变换层
        self.act_fn = ACT2FN[config.hidden_act]                                             # YoungL：激活层

    def forward(self, x):
        if self.config.pretraining_tp > 1:  # YoungL：如果pretraining_tp大于1则需要张量并行
            slice = self.intermediate_size // self.config.pretraining_tp    # YoungL：计算拆成pretraining_tp份之后，每一份的维度            
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)    # YoungL：将gate矩阵竖着切分
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)        # YoungL：将第一个线性变换矩阵竖着切分
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)    # YoungL：将第二个线性变换矩阵竖着切分

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )               
            # YoungL： batch_size * seq_len * hidden_size  * （intermediate_size/pretraining_tp * hidden_size）转置  =  batch_size*seq_len*（intermedidate_size/pretraining_tp）
            # YoungL： 经过concat变为batch_size * seq_len * intermediate_size
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            # YoungL： batch_size * seq_len * hidden_size  * （intermediate_size/pretraining_tp * hidden_size）转置 =  batch_size*seq_len*（intermedidate_size/pretraining_tp）
            # YoungL： 经过concat变为batch_size * seq_len * intermediate_size
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)    # YoungL：对位相乘并切分
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]   # YoungL：batch_size * seq_len * intermediate_size/pretraining_tp   *   （hidden_size * intermediate_size/pretraining_tp）转置
            down_proj = sum(down_proj)              # YoungL：对位相加
        else:                   # YoungL：没有并行处理的话，直接两层线性变换
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj    # YoungL：返回结果


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


class LlamaAttention(nn.Module):                                            # YoungL：多头注意力机制
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config                                                # YoungL：配置参数
        self.hidden_size = config.hidden_size                               # YoungL：隐层输出
        self.num_heads = config.num_attention_heads                         # YoungL：头数
        self.head_dim = self.hidden_size // self.num_heads                  # YoungL：多头切分后，每个头的维度
        self.num_key_value_heads = config.num_key_value_heads               # YoungL：K、V的头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # YoungL：K、V分组
        # YoungL：因为下文对输入线性变换计算Q、K、V矩阵的时候，Q的维度与K、V不同；并且Q的维度大，所以一个头的K、V对应着多头的Q，K、V的维度小是为了减少缓存的占用；
        self.max_position_embeddings = config.max_position_embeddings       # YoungL：最大文本长度
        self.rope_theta = config.rope_theta                                 # YoungL：旋转编码的相位

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )                                                               # YoungL：判断hidden_size是否可以被头数整除
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)               # YoungL：Q矩阵     
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # YoungL：K矩阵
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # YoungL：V矩阵
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)               # YoungL：attention的输出层变换矩阵

        self._init_rope()       # YoungL：初始化旋转位置编码

    def _init_rope(self):       # YoungL：初始化旋转位置编码
        if self.config.rope_scaling is None:    # YoungL：通过head_dim，最大文本长度和相位初始化
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:                   # YoungL：通过自定义类型加上head_dim，最大文本长度和相位初始化
            scaling_type = self.config.rope_scaling["type"]             # YoungL：旋转位置编码类型
            scaling_factor = self.config.rope_scaling["factor"]         # YoungL：旋转位置编码因子
            if scaling_type == "linear":        # YoungL：线性旋转矩阵
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":     # YoungL：动态旋转矩阵
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
        hidden_states: torch.Tensor,                                        # YoungL：decoder_layer的输入经过归一化之后的结果
        attention_mask: Optional[torch.Tensor] = None,                      # YoungL：attention的mask，对Q与K乘积的结果进行mask
        position_ids: Optional[torch.LongTensor] = None,                    # YoungL：position_ids
        past_key_value: Optional[Tuple[torch.Tensor]] = None,               # YoungL：上一轮的K、V
        output_attentions: bool = False,                                    # YoungL：是否输出Q与K计算的attention矩阵
        use_cache: bool = False,                                            # YoungL：是否使用缓存
        padding_mask: Optional[torch.LongTensor] = None,                    # YoungL：padding_mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()                                # YoungL：batch_size * seq_length * hidden_size

        if self.config.pretraining_tp > 1:                                  # YoungL：默认为1，当大于1时则并行训练
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp    # YoungL：计算线性变换矩阵切分pretraining_tp之后每一个slice的维度
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )                                                                                               # YoungL：将Q的变换矩阵切分
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)                                 # YoungL：将K的变换矩阵切分
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)                               # YoungL：将V的变换矩阵切分

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]    # YoungL：每一个slice都乘以hidden_states；相当于A*B，对B做了列切分的参数并行
            query_states = torch.cat(query_states, dim=-1)                                                          # YoungL：将所有的slice成绩结果cat到一起

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]        # YoungL：参数并行
            key_states = torch.cat(key_states, dim=-1)                                                              # YoungL：cat

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]    # YoungL：参数并行
            value_states = torch.cat(value_states, dim=-1)                                                          # YoungL：cat

        else:                                                               # YoungL：计算Q、K、V矩阵
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)         # YoungL：维度变换 batch_size * head_num * seq_length * head_dim
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # YoungL：维度变换 batch_size * KV_head_num * seq_length * head_dim
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)   # YoungL：维度变换 batch_size * KV_head_num * seq_length * head_dim

        kv_seq_len = key_states.shape[-2]           # YoungL：K、V的seq_length
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]   # YoungL：K、V的seq_length加上上一轮缓存的K、V的seq_length
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)       # YoungL：将旋转位置编码耦合到Q和K矩阵

        if past_key_value is not None: 
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)          # YoungL：K矩阵在seq_length维度上耦合了之前多轮的K矩阵
            value_states = torch.cat([past_key_value[1], value_states], dim=2)      # YoungL：V矩阵在seq_length维度上耦合了之前多轮的V矩阵
        # YoungL：Q*K转置为batch_size * head_num * seq_length * k_seq_length   再乘以V变成 batch_size * head_num * seq_length * head_dim

        past_key_value = (key_states, value_states) if use_cache else None          # YoungL：判断是否要保存K、V

        key_states = repeat_kv(key_states, self.num_key_value_groups)               # YoungL：K矩阵的头数进行广播，确保跟Q的头数维度相同，相当于是对应当前头k的不同的q可以同时与同一个k计算attention矩阵了
        value_states = repeat_kv(value_states, self.num_key_value_groups)           # YoungL：同上，头数这个维度不同的话无法与Q相乘

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)    # YoungL：Q * K转置 / 维度的开根号

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):         # YoungL：判断维度是否正确
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:                                              # YoungL：对attention矩阵做掩码
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # YoungL：注意力矩阵所softmax
        attn_output = torch.matmul(attn_weights, value_states)                                                  # YoungL：注意力矩阵 * V

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):                                   # YoungL：判断维度是否正确
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()              # YoungL：维度变换，并且保证存储空间连续
        # YoungL：将维度从batch_size * head_num * seq_length * head_dim 变为 batch_size * seq_length * head_num * head_dim
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)     # YoungL：合并最后两个维度得到 batch_size * seq_length * hidden_size

        if self.config.pretraining_tp > 1:              # YoungL：并行处理
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)      # YoungL：将attn_output矩阵拆分
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)     # YoungL：将O矩阵拆分
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])  # YoungL：参数并行的另一种方式：第一个矩阵竖着切分，第二个矩阵横着切分，计算结果对位相加
        else:
            attn_output = self.o_proj(attn_output)      # YoungL：不并行处理，直接做线性变换

        if not output_attentions:                           # YoungL：判断是否需要返回attention矩阵
            attn_weights = None

        return attn_output, attn_weights, past_key_value    # YoungL：返回结果


class LlamaFlashAttention2(LlamaAttention):                                 # YoungL：快速、内存高效的注意力
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,                                        # YoungL：attention的输入
        attention_mask: Optional[torch.Tensor] = None,                      # YoungL：注意力掩码矩阵
        position_ids: Optional[torch.LongTensor] = None,                    # YoungL：positiion_ids
        past_key_value: Optional[Tuple[torch.Tensor]] = None,               # YoungL：上一轮的K、V
        output_attentions: bool = False,                                    # YoungL：是否输出attention的矩阵
        use_cache: bool = False,                                            # YoungL：是否使用缓存
        padding_mask: Optional[torch.LongTensor] = None,                    # YoungL：padding的mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False                                           # YoungL：默认不输出attention矩阵

        bsz, q_len, _ = hidden_states.size()                                # YoungL：batch_size * seq_length * hidden_size

        query_states = self.q_proj(hidden_states)                           # YoungL：计算Q、K、V矩阵
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)                 # YoungL：Q、K、V矩阵多头切分
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]                       # YoungL：取出kv_seq_length
        if past_key_value is not None:                          # YoungL：长度加上一已经保存的前几轮的kv
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)       # YoungL：将旋转位置编码信息耦合

        if past_key_value is not None:                          # YoungL：如果保存了过去几轮的kv则将新的kv拼接都后面
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None      # YoungL：如果使用缓存，则对当前轮的kv进行缓存

        query_states = query_states.transpose(1, 2)             # YoungL：维度变换
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # YoungL：将维度从batch_size * head_num * seq_length * head_dim 变为 batch_size * seq_length * head_num * head_dim

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype        # YoungL：获取输入的数据类型
        if input_dtype == torch.float32:        # YoungL：将输入转为float16进行计算，即混合精度训练，用float32保存，float16计算
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)       # YoungL：将float32转换为float16
            key_states = key_states.to(torch.float16)           # YoungL：将float32转换为float16
            value_states = value_states.to(torch.float16)       # YoungL：将float32转换为float16

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate
        )           # YoungL：做注意力计算（为什么在注意力计算之前对多头矩阵的1、2两个维度做两次变换？只是为了与缓存的kv拼接吗？）

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()    # YoungL：shape转换为输入维度
        attn_output = self.o_proj(attn_output)                                          # YoungL：输出层线性变换

        if not output_attentions:                                                       # YoungL：如果不输出attention矩阵，则矩阵置空
            attn_weights = None

        return attn_output, attn_weights, past_key_value                                # YoungL：返回结果

    def _flash_attention_forward(
        self, query_states, key_states, value_states, padding_mask, query_length, dropout=0.0, softmax_scale=None
    ):                                                                                  # YoungL：flashattention的前馈计算
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:                                    # YoungL：如果有pad_maks，则先处理pad
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )                                                           # YoungL：对QKV的pad处理

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):     # YoungL：pad处理
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaDecoderLayer(nn.Module):             # YoungL：llama的单个decoder_layer实现，包括attention、MLP和RMS归一化
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size   # YoungL：decoder_layer的输入层维度
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )                                       # YoungL：attention层实现，LlamaAttention或者LlamaFlashAttention2二选一
        self.mlp = LlamaMLP(config)             # YoungL：定义MLP
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)    # YoungL：定义输入的归一化
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)   # YoungL：定义attention的归一化

    def forward(
        self,
        hidden_states: torch.Tensor,                                    # YoungL：上一层的输出保存在此变量中
        attention_mask: Optional[torch.Tensor] = None,                  # YoungL：掩码矩阵
        position_ids: Optional[torch.LongTensor] = None,                # YoungL：position_ids
        past_key_value: Optional[Tuple[torch.Tensor]] = None,           # YoungL：过去的K、V（上一轮前向计算的结果）
        output_attentions: Optional[bool] = False,                      # YoungL：attention的矩阵是否要输出
        use_cache: Optional[bool] = False,                              # YoungL：是否使用缓存      
        padding_mask: Optional[torch.LongTensor] = None,                # YoungL：padding的mask矩阵
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states        # YoungL：记录下当前decoder_layer的输入用于后续的attention残差链接

        hidden_states = self.input_layernorm(hidden_states)     # YoungL：首先对输入进行归一化

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )                                           # YoungL：将归一化后的输入计算attention
        hidden_states = residual + hidden_states    # YoungL：attention的残差链接

        # Fully Connected
        residual = hidden_states                    # YoungL：MLP的残差链接
        hidden_states = self.post_attention_layernorm(hidden_states)    # YoungL：对attention的结果进行归一化
        hidden_states = self.mlp(hidden_states)     # YoungL：归一化之后的结果送入MLP层
        hidden_states = residual + hidden_states    # YoungL：MLP的结果宇MLP的输入计算残差链接

        outputs = (hidden_states,)                  # YoungL：每一层输出的第一个元素为当前层的隐输出

        if output_attentions:                       # YoungL：第二个元素是attention的权重
            outputs += (self_attn_weights,)

        if use_cache:                               # YoungL：第三个元素是当前层的K、V
            outputs += (present_key_value,)

        return outputs                              # YoungL：返回当前decoder_layer的输出元组

# YoungL：模型初始化时的提示信息
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

# YoungL:在模型开始时，打印模型的相关信息
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig                  # YoungL：加载llama模型的config信息
    base_model_prefix = "model"                 # YoungL：
    supports_gradient_checkpointing = True      # YoungL：是否支持梯度检查点（优化内存）
    _no_split_modules = ["LlamaDecoderLayer"]           # YoungL：
    _skip_keys_device_placement = "past_key_values"     # YoungL：
    _supports_flash_attn_2 = True                   # YoungL：Attention的模式（llama中配备了LlamaAttention和LlamaFlashAttention）

    def _init_weights(self, module):            # YoungL：权重初始化
        std = self.config.initializer_range     # YoungL：获取配置文件中的初始化方差
        if isinstance(module, nn.Linear):       # YoungL：判断是否是线性层（attentioin和MLP都是线性层的组合）
            module.weight.data.normal_(mean=0.0, std=std)   # YoungL：根据均值和方差初始化参数
            if module.bias is not None:         # YoungL：如果线性层有偏置项则偏置项初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):  # YoungL：判断是否是embedding层
            module.weight.data.normal_(mean=0.0, std=std)   # YoungL：根据均值和方差初始化
            if module.padding_idx is not None:              # YoungL：判断是否存在pad_index，有的话初始化为0向量
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):     # YoungL：设置梯度检查点
        if isinstance(module, LlamaModel):      # YoungL：判断是否是LlamaModel，是的话则根据接收到的value值设置是否启用检查点
            module.gradient_checkpointing = value

# YoungL：模型初始化时的提示信息
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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
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
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
"""

# YoungL:在模型开始时，打印模型的相关信息
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):         # YoungL：包括embedding和多个decoder_layers
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id      # YoungL：pad_index
        self.vocab_size = config.vocab_size         # YoungL：词表大小

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)       # YoungL：embedding层构建   
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])   # YoungL：构建多层decoderlayer
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)       # YoungL：构建正则化

        self.gradient_checkpointing = False     # YoungL：不启用梯度检查点
        # Initialize weights and apply final processing
        self.post_init()    # YoungL：位置编码初始化

    def get_input_embeddings(self):     # YoungL：获取embedding的参数
        return self.embed_tokens

    def set_input_embeddings(self, value):  # YoungL：设置embedding的参数
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):  # YoungL：设置掩码
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)  # YoungL：打印模型前馈计算的提示信息
    def forward(
        self,
        input_ids: torch.LongTensor = None,                             # YoungL：输入id
        attention_mask: Optional[torch.Tensor] = None,                  # YoungL：attention_mask
        position_ids: Optional[torch.LongTensor] = None,                # YoungL：位置ID
        past_key_values: Optional[List[torch.FloatTensor]] = None,      # YoungL：之前的K、V（上一轮前向计算的结果）
        inputs_embeds: Optional[torch.FloatTensor] = None,              # YoungL：输入的embedding，跟input_ids二选一就可以
        use_cache: Optional[bool] = None,                               # YoungL：是否使用缓存
        output_attentions: Optional[bool] = None,                       # YoungL：输出的attention
        output_hidden_states: Optional[bool] = None,                    # YoungL：是否输出中间层结果
        return_dict: Optional[bool] = None,                             # YoungL：返回值形式为dict
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions   # YoungL：设置输出attention
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )                                                               # YoungL：设置需要输出的隐藏层
        use_cache = use_cache if use_cache is not None else self.config.use_cache       # YoungL：设置是否使用缓存

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict       # YoungL：设置返回的参数

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:         # YoungL：不能同时传递input_ids和input_embeds
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:                                     # YoungL：如果两个参数有一个不为空则取出batch_size和seq_length
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:                                                           # YoungL：若两个参数都为空，则报错
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length                   # YoungL：已经存在的序列部分长度，初始状态下就是输入序列的长度
        past_key_values_length = 0                          # YoungL：已经存在的K、V长度

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device        # YoungL：获取input_ids所在的device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )                                               # YoungL：根据输入序列的长度生成position_ids，放入input_ids的devices中
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)   # YoungL：增加batch_size的维度为1
        else:
            position_ids = position_ids.view(-1, seq_length).long()     # YoungL：设置position_ids的维度

        if inputs_embeds is None:       # YoungL：如果传入的是input_ids则需要做embedding处理，传入的是input_embeds则不需要处理
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:      # YoungL：如果没有attention_mask则需要构建。
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )   # YoungL：构建全1矩阵维度为batct_size*seq_length_with_past，即已经存在的所有字符在计算attention时，不需要mask
            padding_mask = None         # YoungL：输入序列没有pad，所以为None
        else:
            if 0 in attention_mask:     # YoungL：如果输入的attention_mask里面有0，则表示有padding的字符
                padding_mask = attention_mask
            else:                       # YoungL：如果没有0则时在输入的时候已经为input_ids构建好了attention_mask的矩阵，并且是所哟与的input_ids之间没有mask
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )           # YoungL：构建decoder的attention_mask

        hidden_states = inputs_embeds   # YoungL：设置初始的隐状态

        if self.gradient_checkpointing and self.training:   # YoungL：如果是训练模式，并且梯度检查点为true则不使用缓存。
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None    # YoungL：如果需要输出中间层结果，则初始化为元组，用于保存结果，否则为None
        all_self_attns = () if output_attentions else None  # YoungL：如果需要输出attention的结果则初始化为元组
        next_decoder_cache = () if use_cache else None      # YoungL：如果使用缓存则初始化为元组，否则为None

        for idx, decoder_layer in enumerate(self.layers):   # YoungL：遍历所有的block
            if output_hidden_states:                        # YoungL：隐藏层的结果需要输出则把隐状态加入元组
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None  # YoungL：如果之前保存了当前层KV，则加载当前层的KV

            if self.gradient_checkpointing and self.training:   # YoungL：如果需要梯度检查点并且为训练模式，则create_custom_forward

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:   # YoungL：如果不需要检查点则直接通过普通的decoder_layer进行前向计算
                layer_outputs = decoder_layer(
                    hidden_states,                      # YoungL：上一层的输出
                    attention_mask=attention_mask,      # YoungL：attention_mask
                    position_ids=position_ids,          # YoungL：位置编码
                    past_key_value=past_key_value,      # YoungL：之前保存的结果
                    output_attentions=output_attentions,    # YoungL：输出的attention_mask
                    use_cache=use_cache,                # YoungL：是否使用缓存
                    padding_mask=padding_mask,          # YoungL：padding的mask
                )

            hidden_states = layer_outputs[0]    # YoungL：输出的应该是一个结果元组，而第零个元素是当前层的输出

            if use_cache:   # YoungL：判断是否使用cache
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)  
                 # YoungL：layer_output[1]为output_attention； layer_output[2]为缓存结果，如果没有output_attention则向前补位 
            if output_attentions:       # YoungL：如果需要output_attention则保存第一个元素
                all_self_attns += (layer_outputs[1],)

        # YoungL：此处为所有block的最后一层的输出
        hidden_states = self.norm(hidden_states)    # YoungL：最后一层的出书hidden_states做归一化处理

        # add hidden states from the last decoder layer
        if output_hidden_states:                    # YoungL：将最后一层的hidden_states加入到all_hidden_states中
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None  # YoungL：将当前一个前馈计算的结果缓存
        if not return_dict:                                     # YoungL：如果返回的不是dict类型，则返回元组
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )       # YoungL：否则返回字典形式


class LlamaForCausalLM(LlamaPreTrainedModel):               # YoungL：普通生成式语言模型
    _tied_weights_keys = ["lm_head.weight"]                 # YoungL：TODO

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)                     # YoungL：初始化decoder层
        self.vocab_size = config.vocab_size                 # YoungL：记录词表尺寸
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)         # YoungL：在decoder层之上加一个线性层，将最后一层decoder的结果映射到词表维度进行解码

        # Initialize weights and apply final processing
        self.post_init()                                    # YoungL：初始化位置编码

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):                         # YoungL：加载decoder层的参数
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions   # YoungL：是否接收注意力矩阵
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )                                                                                                           # YoungL：是否接收隐层输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict                       # YoungL：返回值格式

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
        )                                                       # YoungL：获取多层decoder的结果

        hidden_states = outputs[0]                              # YoungL：取出最后一层decoder的隐输出
        if self.config.pretraining_tp > 1:                      # YoungL：输出层是否使用张量并行
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)    # YoungL：矩阵竖着切分
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]    # YoungL：计算乘积
            logits = torch.cat(logits, dim=-1)                                                                  # YoungL：最后一维拼接
        else:
            logits = self.lm_head(hidden_states)                # YoungL：不做并行处理，直接线性变换
        logits = logits.float()                                 # YoungL：TODO  原来是什么数据类型

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()     # YoungL：batch_size * seq_length * vocab_size => batch_size * seq_length - 1 * vocab_size
            shift_labels = labels[..., 1:].contiguous()         # YoungL：batch_size * seq_length => batch_size * seq_length - 1
            # YoungL：shift_logits去掉最后一个字符，shift_labels去掉第一个字符，然后所有位置的token都预测下一个位置，计算loss
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)    # YoungL：batch*(seq-1)  *  vocab_size
            shift_labels = shift_labels.view(-1)                            # YoungL：batch*(seq-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)             # YoungL：label迁移到计算设备上
            loss = loss_fct(shift_logits, shift_labels)                     # YoungL：计算loss

        if not return_dict:
            output = (logits,) + outputs[1:]                                # YoungL：如果不返回字典，则取回KV缓存、attention和中间隐层输出，保存为元组
            return (loss,) + output if loss is not None else output         # YoungL：将loss一起返回

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # YoungL：返回值格式

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )                                                                           # YoungL：最后一层decoder的输出
        hidden_states = transformer_outputs[0]                                      # YoungL：取出隐藏层
        logits = self.score(hidden_states)                                          # YoungL：经过线性层处理

        if input_ids is not None:                                                   # YoungL：获取batch_size
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:                    # YoungL：batch大于1的时候需要传入pad_token_id
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:                                        # YoungL：如果配置中没有pad_token_id，则sequence_length设置-1
            sequence_lengths = -1
        else:
            if input_ids is not None:                                               # YoungL：如果有pad_token_id，并且input式ids形式传入，则二者判断是否相等，可以得出每一条数据的长度
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )           # YoungL：eq判断相等为True不等为False，用long转换成1和0，argmax则找出第一个1的位置（即pad开始的位置），减1可能是seq的第一个字符是特殊字符
            else:
                sequence_lengths = -1                                               # YoungL：否则长度的为-1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]    
        # YoungL：切片操作logits[0,a,b] a是torch.arange(batch_size, device=logits.device)里面的， b是sequence_lengths里面的，a，b排列组合做切片，把pad的位置都去掉了

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":                                    # YoungL：回归
                loss_fct = MSELoss()
                if self.num_labels == 1:                                                    # YoungL：单值回归，只有一个位置需要回归计算
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:                                                                       # YoungL：多值回归，多个位置需要回归计算
                    loss = loss_fct(pooled_logits, labels)  
            elif self.config.problem_type == "single_label_classification":                 # YoungL：一分类
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":                  # YoungL：多分类
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
