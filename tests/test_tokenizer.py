# -*- coding: utf-8 -*-
import pytest


@pytest.mark.parametrize('text, text_seq_length, bpe_dropout', [
    ('hello, how are you?', 128, 0.1),
    ('hello, how are you?', 128, 0.5),
    ('hello, how are you?', 128, 1.0),
    ('hello ... how are you ?', 256, 1.0),
    ('a person standing at a table with bottles of win', 64, 0.5),
    ('привет как дела???', 76, 0.0),
    ('клип на русском языке :)', 76, 0.1),
])
def test_encode_decode_text_yttm(yttm_tokenizer, text, text_seq_length, bpe_dropout):
    tokens = yttm_tokenizer.encode_text(text, text_seq_length=text_seq_length, bpe_dropout=bpe_dropout)
    decoded_text = yttm_tokenizer.decode_text(tokens)
    assert text == decoded_text
