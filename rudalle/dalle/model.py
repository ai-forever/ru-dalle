# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from einops import rearrange

from .utils import exists, is_empty, init_method_normal

from .transformer import DalleTransformer


class DalleModel(torch.nn.Module):
    def __init__(self,
                 device,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 text_seq_length=128,
                 image_tokens_per_dim=32,
                 image_vocab_size=16384,
                 loss_img_weight=7,
                 cogview_sandwich_layernorm=False,
                 cogview_pb_relax=False,
                 is_bool_mask=True,
                 mlp_activation='gelu_jit'):
        super(DalleModel, self).__init__()
        self.device = device
        self.image_tokens_per_dim = image_tokens_per_dim
        self.image_seq_length = image_tokens_per_dim ** 2
        self.text_seq_length = text_seq_length
        self.total_seq_length = self.text_seq_length + self.image_seq_length
        self.total_vocab_size = vocab_size + image_vocab_size
        self.vocab_size = vocab_size
        self.loss_img_weight = loss_img_weight

        init_method = init_method_normal(std=0.02)

        self.text_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.image_embeddings = torch.nn.Embedding(image_vocab_size, hidden_size)

        # Position embedding (serial).
        self.text_pos_embeddings = torch.nn.Embedding(text_seq_length + 1, hidden_size)
        self.image_row_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        self.image_col_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        init_method(self.text_pos_embeddings.weight)
        init_method(self.image_row_embeddings.weight)
        init_method(self.image_col_embeddings.weight)

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.total_vocab_size),
        )

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = DalleTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            text_seq_length=text_seq_length,
            image_tokens_per_dim=image_tokens_per_dim,
            cogview_sandwich_layernorm=cogview_sandwich_layernorm,
            cogview_pb_relax=cogview_pb_relax,
            mlp_activation=mlp_activation,
            is_bool_mask=is_bool_mask,
        )

    def get_param(self, item):
        return getattr(self, item)

    def get_image_pos_embeddings(self, image_input_ids, past_length=0):
        input_shape = image_input_ids.size()
        row_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) // self.image_tokens_per_dim
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) % self.image_tokens_per_dim
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)

    def forward(
            self,
            input_ids,
            attention_mask,
            return_loss=False,
            has_cache=False,
            use_cache=False,
    ):
        text = input_ids[:, :self.text_seq_length]
        text_range = torch.arange(self.text_seq_length)
        text_range += (self.vocab_size - self.text_seq_length)
        text_range = text_range.to(self.device)
        text = torch.where(text == 0, text_range, text)
        # some hardcode :)
        text = F.pad(text, (1, 0), value=2)
        text_embeddings = self.text_embeddings(text) + \
            self.text_pos_embeddings(torch.arange(text.shape[1], device=self.device))

        image_input_ids = input_ids[:, self.text_seq_length:]

        if exists(image_input_ids) and not is_empty(image_input_ids):
            image_embeddings = self.image_embeddings(image_input_ids) + \
                self.get_image_pos_embeddings(image_input_ids, past_length=0)
            embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        else:
            embeddings = text_embeddings
        # some hardcode :)
        if embeddings.shape[1] > self.total_seq_length:
            embeddings = embeddings[:, :-1]

        alpha = 0.1
        embeddings = embeddings * alpha + embeddings.detach() * (1-alpha)

        attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]
        transformer_output, present_has_cache = self.transformer(
            embeddings, attention_mask, has_cache=has_cache, use_cache=use_cache)

        logits = self.to_logits(transformer_output)
        if return_loss is False:
            return logits, present_has_cache

        labels = torch.cat((text[:, 1:], image_input_ids), dim=1).contiguous().long()
        logits = rearrange(logits, 'b n c -> b c n')

        text_logits = logits[:, :self.vocab_size, :self.text_seq_length].contiguous().float()
        image_logits = logits[:, self.vocab_size:, self.text_seq_length:].contiguous().float()

        loss_text = F.cross_entropy(
            text_logits,
            labels[:, :self.text_seq_length])
        loss_img = F.cross_entropy(
            image_logits,
            labels[:, self.text_seq_length:])

        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return loss, {'text': loss_text.data.detach().float(), 'image': loss_img.data.detach().float()}

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)
