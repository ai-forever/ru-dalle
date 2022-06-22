# -*- coding: utf-8 -*-
import math

import torch
from torch.nn import LayerNorm

from .utils import divide, split_tensor_along_last_dim
from .image_attention import get_conv_mask, get_row_mask, get_col_mask


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


@torch.jit.script
def gelu_jit(x):
    """OpenAI's gelu implementation."""
    return gelu(x)


class Layer(torch.nn.Module):
    """
    Helper class for gradient checkpointing.
    """

    def __init__(self, x, f, *args, **kwargs):
        super(Layer, self).__init__()
        # module to checkpoint
        self.x = x
        # post-processing function
        self.f = f
        # arguments to the module
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.f(self.x(x, *self.args, **self.kwargs))


class DalleTransformer(torch.nn.Module):
    """
    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
    """
    _mask_map = []

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 text_seq_length,
                 image_tokens_per_dim,
                 layernorm_epsilon=1.0e-5,
                 cogview_sandwich_layernorm=False,
                 cogview_pb_relax=False,
                 mlp_activation='gelu_jit',
                 is_bool_mask=False,
                 hf_version='v3'):
        super(DalleTransformer, self).__init__()

        self.num_layers = num_layers
        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_pb_relax = cogview_pb_relax
        # Additional stabilization tweak for large models
        self.hf_version = hf_version
        # Transformer layers.
        self.layers = torch.nn.ModuleList([
            DalleTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                cogview_sandwich_layernorm=cogview_sandwich_layernorm,
                cogview_pb_relax=cogview_pb_relax,
                mlp_activation=mlp_activation,
            ) for _ in range(num_layers)
        ])

        row_mask = get_row_mask(text_seq_length, image_tokens_per_dim, is_bool_mask=is_bool_mask)
        col_mask = get_col_mask(text_seq_length, image_tokens_per_dim, is_bool_mask=is_bool_mask)
        conv_mask = get_conv_mask(text_seq_length, image_tokens_per_dim, is_bool_mask=is_bool_mask,
                                  hf_version=self.hf_version)
        self.register_buffer('row_mask', row_mask)
        self.register_buffer('col_mask', col_mask)
        self.register_buffer('conv_mask', conv_mask)

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def _get_layer_mask(self, layer_id):
        if ((layer_id - 1) % 4 == 0):
            layer_mask = self.col_mask
        elif layer_id != self.num_layers - 1:
            layer_mask = self.row_mask
        else:
            layer_mask = self.conv_mask
        return layer_mask

    def forward(self, hidden_states, attention_mask, cache=None, use_cache=False, gradient_checkpointing=None):
        if cache is None:
            cache = {}
        # Immutable caching uses much more VRAM.
        # present_cache = {}

        if gradient_checkpointing:
            assert not use_cache
            layers = []

        for i, layer in enumerate(self.layers):
            mask = attention_mask
            layer_mask = self._get_layer_mask(i)[:mask.size(2), :mask.size(3)]
            mask = torch.mul(attention_mask, layer_mask)
            if gradient_checkpointing:
                layers.append(Layer(layer,
                                    # only get the embeddings, not present_has_cache
                                    lambda x: x[0],
                                    mask,
                                    use_cache=False, has_cache=False))
            else:
                hidden_states, layer_cache = layer(
                    hidden_states, mask, cache.get(i), mlp_cache=i == len(self.layers)-1, use_cache=use_cache)
                cache[i] = layer_cache
        if gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint_sequential(
                layers, gradient_checkpointing, hidden_states)

        output = self.final_layernorm(hidden_states)
        return output, cache


class DalleTransformerLayer(torch.nn.Module):
    """
    A single layer transformer.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 cogview_sandwich_layernorm=False,
                 cogview_pb_relax=False,
                 mlp_activation='gelu_jit'):
        super(DalleTransformerLayer, self).__init__()

        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_sandwich_layernorm = cogview_sandwich_layernorm
        self.cogview_pb_relax = cogview_pb_relax

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        if self.cogview_sandwich_layernorm:
            self.before_first_addition_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
            self.before_second_addition_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = DalleSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            cogview_pb_relax=cogview_pb_relax,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = DalleMLP(hidden_size, output_dropout_prob, activation=mlp_activation)

    def forward(self, hidden_states, ltor_mask, cache=None, use_cache=False, mlp_cache=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]
        # cache: [3/4, b, s, h] (query, key, output)

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, new_cache = self.attention(
            layernorm_output, ltor_mask,
            cache=cache[:3] if use_cache and cache is not None else cache, use_cache=use_cache)

        if self.cogview_sandwich_layernorm:
            attention_output = self.before_first_addition_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output

        # Update cache
        cached = 0 if cache is None else cache[0].shape[-2]

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        if use_cache and cached:
            # MLP caching for the last layer.
            mlp_output = torch.cat((cache[-1] if mlp_cache else layernorm_output[..., :cached, :],
                                    self.mlp(layernorm_output[..., cached:, :])), dim=-2)
            if mlp_cache:
                new_cache = new_cache + (mlp_output,)
        else:
            mlp_output = self.mlp(layernorm_output)

        if self.cogview_sandwich_layernorm:
            mlp_output = self.before_second_addition_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, new_cache


class DalleSelfAttention(torch.nn.Module):
    """
    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence length, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        output_dropout_prob: dropout probability for the output.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 cogview_pb_relax=False):
        super(DalleSelfAttention, self).__init__()

        # CogView stabilization of training features, see chapter 2.4 https://arxiv.org/pdf/2105.13290.pdf
        self.cogview_pb_relax = cogview_pb_relax
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """ Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with size [b, np, s, hn]. """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _calculate_attention_scores(self, query_layer, key_layer, ltor_mask):
        key_t = key_layer.transpose(-1, -2)
        mask_value = 10000.0
        if self.cogview_pb_relax:
            attention_scores = torch.matmul(
                query_layer / math.sqrt(self.hidden_size_per_attention_head),
                key_t
            )
        else:
            attention_scores = torch.matmul(query_layer, key_t) / math.sqrt(self.hidden_size_per_attention_head)
        ltor_mask = ltor_mask[:, :, -attention_scores.shape[-2]:]
        attention_scores = torch.mul(attention_scores, ltor_mask) - mask_value * (1.0 - ltor_mask)
        if self.cogview_pb_relax:
            # normalize attention scores. Should not affect resulting softmax value
            alpha = 32
            attention_scores_scaled = attention_scores / alpha
            attention_scores_scaled_maxes, _ = attention_scores_scaled.detach().view(
                [attention_scores.size(0), attention_scores.size(1), -1]
            ).max(dim=-1)  # max per head per sample
            attention_scores_scaled_maxes = attention_scores_scaled_maxes.unsqueeze(-1).unsqueeze(-1).expand(
                [-1, -1, attention_scores.size(2), attention_scores.size(3)]
            )  # expand to [b, np, s, s]
            attention_scores = (attention_scores_scaled - attention_scores_scaled_maxes) * alpha
        return attention_scores

    def forward(self, hidden_states, ltor_mask, use_cache=False, cache=None,):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]
        # cache: [3, b, s, h]  (key, value, output)
        # Attention heads. [b, s, hp]
        if use_cache and cache is not None:
            mixed_x_layer = self.query_key_value(hidden_states[:, cache[0].shape[-2]:, :])
        else:
            mixed_x_layer = self.query_key_value(hidden_states)

        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Can be simplified, but I didn't for readability's sake
        if use_cache and cache is not None:
            past_key, past_value, past_output = cache
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)
            attention_scores = self._calculate_attention_scores(
                query_layer=query_layer, key_layer=key_layer, ltor_mask=ltor_mask
            )
            extra_cache_size = hidden_states.shape[-2] - past_key.shape[-2]
            attention_scores = attention_scores[..., -extra_cache_size:, :]
        else:
            attention_scores = self._calculate_attention_scores(
                query_layer=query_layer, key_layer=key_layer, ltor_mask=ltor_mask
            )

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)

        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)

        if use_cache and cache is not None:
            output = torch.cat((past_output, output), dim=-2)

        if use_cache:
            cache = key_layer, value_layer, output
        # another option:
        # cache = torch.cat((key_layer, value_layer, output), dim=0)

        output = self.output_dropout(output)
        return output, cache


class DalleMLP(torch.nn.Module):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
    """

    def __init__(self, hidden_size, output_dropout_prob, activation='gelu_jit'):
        super(DalleMLP, self).__init__()
        self.activation = activation
        # Project to 4h.
        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        # Project back to h.
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        x = self.dense_h_to_4h(hidden_states)
        if self.activation == 'gelu_jit':
            x = gelu_jit(x)
        elif self.activation == 'gelu':
            x = gelu(x)
        else:
            raise NotImplementedError('Used MLP activation is not implemented.')

        # [b, s, h]
        x = self.dense_4h_to_h(x)

        output = self.dropout(x)
        return output
