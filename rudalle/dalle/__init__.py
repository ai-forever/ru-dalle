# -*- coding: utf-8 -*-
import os

import torch
from huggingface_hub import hf_hub_url, cached_download

from .model import DalleModel
from .fp16 import FP16Module


MODELS = {
    'Malevich': dict(
        hf_version='v3',
        description='◼️ Malevich is 1.3 billion params model from the family GPT3-like, '
                    'that uses Russian language and text+image multi-modality.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='sberbank-ai/rudalle-Malevich',
        filename='pytorch_model_v3.bin',
        authors='SberAI, SberDevices, shonenkovAI',
        full_description='',  # TODO
    ),
    'Malevich_v2': dict(
        hf_version='v2',
        description='◼️ Malevich is 1.3 billion params model from the family GPT3-like, '
                    'that uses Russian language and text+image multi-modality.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='sberbank-ai/rudalle-Malevich',
        filename='pytorch_model_v2.bin',
        authors='SberAI, SberDevices, shonenkovAI',
        full_description='',  # TODO
    ),
    'Emojich': dict(
        hf_version='v2',
        description='😋 Emojich is 1.3 billion params model from the family GPT3-like, '
                    'it generates emoji-style images with the brain of ◾ Malevich.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='sberbank-ai/rudalle-Emojich',
        filename='pytorch_model.bin',
        authors='SberAI, SberDevices, shonenkovAI',
        full_description='',  # TODO
    ),
    'Surrealist_XL': dict(
        hf_version='v3',
        description='Surrealist is 1.3 billion params model from the family GPT3-like, '
                    'that was trained on surrealism and Russian.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='shonenkov-AI/rudalle-xl-surrealist',
        filename='pytorch_model.bin',
        authors='shonenkovAI',
        full_description='',
    ),
    'Kandinsky': dict(
        hf_version='v3',
        description='Kandinsky is large 12 billion params model from the family GPT3-like, '
                    'that uses Russian language and text+image multi-modality.',
        model_params=dict(
            num_layers=64,
            hidden_size=3840,
            num_attention_heads=60,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='shonenkov-AI/Kandinsky',
        filename='pytorch_model.bin',
        authors='SberAI, SberDevices, shonenkovAI',
        full_description='',  # TODO
    ),
    'dummy': dict(
        hf_version='v3',
        description='',
        model_params=dict(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='',
        filename='',
        full_description='',  # TODO
    ),
}


def get_rudalle_model(name, pretrained=True, fp16=False, device='cpu', use_auth_token=None,
                      cache_dir='/tmp/rudalle', **model_kwargs):
    assert name in MODELS

    if fp16 and device == 'cpu':
        print('Warning! Using both fp16 and cpu doesnt support. You can use cuda device or turn off fp16.')

    config = MODELS[name].copy()
    config['model_params'].update(model_kwargs)
    model = DalleModel(device=device, hf_version=config['hf_version'], **config['model_params'])
    if pretrained:
        cache_dir = os.path.join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'],
                        use_auth_token=use_auth_token)
        checkpoint = torch.load(os.path.join(cache_dir, config['filename']), map_location='cpu')
        model.load_state_dict(checkpoint)
    if fp16:
        model = FP16Module(model)
    model.eval()
    model = model.to(device)
    if config['description'] and pretrained:
        print(config['description'])
    return model
