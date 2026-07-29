# Adapted from DeQA-Score and mPLUG-Owl2, with portions derived from
# Hugging Face Transformers. See the repository NOTICE for attribution
# and license details.
import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.masking_utils import eager_mask, sdpa_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_mplug_owl2 import LlamaConfig


logger = logging.get_logger(__name__)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.max_seq_len_cached = 0
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _get_inv_freq(self, seq_len, device):
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        self.inv_freq = self._get_inv_freq(seq_len, device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if (
            self.cos_cached is None
            or seq_len > self.max_seq_len_cached
            or self.cos_cached.device != x.device
        ):
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_position_embeddings),
                device=x.device,
                dtype=torch.float32,
            )
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        self.inv_freq = self._get_inv_freq(seq_len, device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _get_inv_freq(self, seq_len, device):
        base = self.base
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
        return 1.0 / (
            base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        self.inv_freq = self._get_inv_freq(seq_len, device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, sequence_length, head_dim)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=2)
            down_proj = sum(
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            )
            return down_proj

        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MultiwayNetwork(nn.Module):
    def __init__(self, module_provider, num_multiway=2):
        super().__init__()
        self.multiway = nn.ModuleList([module_provider() for _ in range(num_multiway)])

    def forward(self, hidden_states, multiway_indices):
        if len(self.multiway) == 1:
            return self.multiway[0](hidden_states)

        output_hidden_states = None
        for idx, subway in enumerate(self.multiway):
            local_indices = multiway_indices.eq(idx).nonzero(as_tuple=True)
            hidden = hidden_states[local_indices].unsqueeze(1).contiguous()
            if hidden.numel():
                output = subway(hidden)
                if isinstance(output, tuple):
                    output = output[0]
                if output_hidden_states is None:
                    output_hidden_states = hidden_states.new_empty(
                        (*hidden_states.shape[:-1], output.shape[-1])
                    )
                output_hidden_states[local_indices] = output.squeeze(1)
        if output_hidden_states is None:
            raise ValueError("modality_indicators did not select any multiway branch")
        return output_hidden_states.contiguous()


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, "
                f"num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = MultiwayNetwork(
            module_provider=partial(
                nn.Linear,
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )
        self.v_proj = MultiwayNetwork(
            module_provider=partial(
                nn.Linear,
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        rope_scaling = self.config._deqa_rope_scaling
        if rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            return

        scaling_type = rope_scaling["type"]
        scaling_factor = rope_scaling["factor"]
        if scaling_type == "linear":
            rotary_embedding = LlamaLinearScalingRotaryEmbedding
        elif scaling_type == "dynamic":
            rotary_embedding = LlamaDynamicNTKScalingRotaryEmbedding
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        self.rotary_emb = rotary_embedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=scaling_factor,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indicators: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        batch_size, query_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states, modality_indicators)
        value_states = self.v_proj(hidden_states, modality_indicators)

        query_states = query_states.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_value_length = key_states.shape[-2]
        if past_key_value is not None:
            key_value_length += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=key_value_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        expected_shape = (batch_size, self.num_heads, query_length, key_value_length)
        if attention_weights.size() != expected_shape:
            raise ValueError(f"Attention weights should have shape {expected_shape}, got {attention_weights.size()}")
        if attention_mask is not None:
            expected_mask_shape = (batch_size, 1, query_length, key_value_length)
            if attention_mask.size() != expected_mask_shape:
                raise ValueError(
                    f"Attention mask should have shape {expected_mask_shape}, got {attention_mask.size()}"
                )
            attention_weights = attention_weights + attention_mask

        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, query_length, self.hidden_size)
        attention_output = self.o_proj(attention_output)

        if not output_attentions:
            attention_weights = None
        return attention_output, attention_weights, present_key_value


class LlamaSdpaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indicators: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        if output_attentions:
            logger.warning_once(
                "SDPA does not return attention weights; falling back to eager attention."
            )
            return super().forward(
                hidden_states=hidden_states,
                modality_indicators=modality_indicators,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        batch_size, query_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states, modality_indicators)
        value_states = self.v_proj(hidden_states, modality_indicators)

        query_states = query_states.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_value_length = key_states.shape[-2]
        if past_key_value is not None:
            key_value_length += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=key_value_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            expected_mask_shape = (
                batch_size,
                1,
                query_length,
                key_value_length,
            )
            if attention_mask.size() != expected_mask_shape:
                raise ValueError(
                    f"Attention mask should have shape {expected_mask_shape}, got {attention_mask.size()}"
                )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and query_length > 1,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            self.hidden_size,
        )
        attention_output = self.o_proj(attention_output)
        return attention_output, None, present_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        attention_classes = {
            "eager": LlamaAttention,
            "sdpa": LlamaSdpaAttention,
        }
        attention_class = attention_classes.get(config._attn_implementation)
        if attention_class is None:
            raise ValueError(
                f"Unsupported attention implementation: {config._attn_implementation}"
            )
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = MultiwayNetwork(
            module_provider=partial(
                LlamaRMSNorm,
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
        )
        self.post_attention_layernorm = MultiwayNetwork(
            module_provider=partial(
                LlamaRMSNorm,
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indicators: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, modality_indicators)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, modality_indicators)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = False
    _supports_sdpa = True

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


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        modality_indicators: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, sequence_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Specify input_ids or inputs_embeds")

        if modality_indicators is None:
            modality_indicators = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.long,
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
            )

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                sequence_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, sequence_length)
        else:
            position_ids = position_ids.view(-1, sequence_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        sequence_length_with_past = sequence_length + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            attention_mask = attention_mask.to(torch.bool)
        # With output_attentions=True, LlamaSdpaAttention falls back to eager attention,
        # which needs the additive mask rather than the boolean SDPA one.
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = sdpa_mask(
                batch_size=batch_size,
                q_length=sequence_length,
                kv_length=sequence_length_with_past,
                q_offset=past_key_values_length,
                attention_mask=attention_mask,
                device=inputs_embeds.device,
            )
        else:
            attention_mask = eager_mask(
                batch_size=batch_size,
                q_length=sequence_length,
                kv_length=sequence_length_with_past,
                q_offset=past_key_values_length,
                attention_mask=attention_mask,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("use_cache=True is incompatible with gradient checkpointing; disabling cache")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return decoder_layer(
                        *inputs,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=False,
                    )

                layer_outputs = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    modality_indicators,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    modality_indicators=modality_indicators,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                )
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
