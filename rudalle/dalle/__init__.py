# -*- coding: utf-8 -*-
import os
import gc

from tqdm.auto import tqdm
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

    if pretrained:
        def init_layer_func(x, prefix=None):
            if prefix:
                used_names = []
                tmp_checkpoint = {}
                for name in checkpoint.keys():
                    if name.startswith(prefix):
                        tmp_checkpoint[name[len(prefix):]] = checkpoint[name]
                        used_names.append(name)
                x.load_state_dict(tmp_checkpoint)
                for used_name in used_names:
                    weights = checkpoint.pop(used_name)
                    weights.to('cpu')
                    del weights
                pbar.update(len(used_names))
                gc.collect()

            if fp16:
                x = x.half()
            x = x.to(device)
            return x

        global checkpoint
        global pbar

        cache_dir = os.path.join(cache_dir, name)
        config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'],
                        use_auth_token=use_auth_token)
        checkpoint = torch.load(os.path.join(cache_dir, config['filename']), map_location=device)

        pbar = tqdm(total=len(checkpoint.keys()))
        pbar.set_description('Init model layer by layer')
    else:
        def init_layer_func(x, prefix=None):
            if fp16:
                x = x.half()
            x = x.to(device)
            return x

    model = DalleModel(device=device, init_layer_func=init_layer_func, hf_version=config['hf_version'],
                       **config['model_params'])

    if pretrained:
        pbar.update(len(checkpoint.keys()))
        del checkpoint
        gc.collect()

    if fp16:
        model = FP16Module(model)
    model.eval()
    model = model.to(device)
    if config['description'] and pretrained:
        print(config['description'])
    return model
