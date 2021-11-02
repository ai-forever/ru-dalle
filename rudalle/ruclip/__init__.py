# -*- coding: utf-8 -*-
import os

from transformers import CLIPModel
from huggingface_hub import hf_hub_url, cached_download

from .processor import RuCLIPProcessor

MODELS = {
    'ruclip-vit-base-patch32-v5': dict(
        repo_id='sberbank-ai/ru-clip',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
}


def get_ruclip(name, cache_dir='/tmp/rudalle'):
    assert name in MODELS
    config = MODELS[name]
    repo_id = config['repo_id']
    cache_dir = os.path.join(cache_dir, name)
    for filename in config['filenames']:
        config_file_url = hf_hub_url(repo_id=repo_id, filename=f'{name}/{filename}')
        cached_download(config_file_url, cache_dir=cache_dir, force_filename=filename)
    ruclip = CLIPModel.from_pretrained(cache_dir)
    ruclip_processor = RuCLIPProcessor.from_pretrained(cache_dir)
    print('ruclip --> ready')
    return ruclip, ruclip_processor
