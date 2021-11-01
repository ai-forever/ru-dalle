# -*- coding: utf-8 -*-
from os.path import join

import torch
import numpy as np
import youtokentome as yttm
from huggingface_hub import hf_hub_url, cached_download


def get_tokenizer(path=None, cache_dir='/tmp/rudalle'):
    # TODO docstring
    if path is None:
        repo_id = 'shonenkov/rudalle-utils'
        filename = 'bpe.model'
        cache_dir = join(cache_dir, 'tokenizer')
        config_file_url = hf_hub_url(repo_id=repo_id, filename=filename)
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)
        path = join(cache_dir, filename)
    tokenizer = YTTMTokenizerWrapper(yttm.BPE(model=path))
    print('tokenizer --> ready')
    return tokenizer


class YTTMTokenizerWrapper:
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.vocab_size()

    def get_pad_token_id(self):
        # TODO docstring
        return self.tokenizer.subword_to_id('<PAD>')

    def vocab_size(self):
        # TODO docstring
        return self.tokenizer.vocab_size()

    def encode_text(self, text, text_seq_length, bpe_dropout=0.0):
        # TODO docstring
        tokens = self.tokenizer.encode([text], output_type=yttm.OutputType.ID, dropout_prob=bpe_dropout)[0]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        return self.prepare_tokens(tokens, text_seq_length)

    def decode_text(self, encoded):
        # TODO docstring
        return self.tokenizer.decode(encoded.cpu().numpy().tolist(), ignore_ids=[
            self.eos_id, self.bos_id, self.unk_id, self.pad_id
        ])[0]

    @staticmethod
    def prepare_tokens(tokens, text_seq_length):
        # TODO docstring
        empty_positions = text_seq_length - len(tokens)
        if empty_positions > 0:
            tokens = np.hstack((tokens, np.zeros(empty_positions)))  # position tokens after text
        if len(tokens) > text_seq_length:
            tokens = tokens[:text_seq_length]
        return torch.tensor(tokens).long()
