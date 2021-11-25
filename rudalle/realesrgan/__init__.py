# -*- coding: utf-8 -*-
import os

from huggingface_hub import hf_hub_url, cached_download

from .model import RealESRGAN


MODELS = {
    'x2': dict(
        scale=2,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x2.pth',
    ),
    'x4': dict(
        scale=4,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x4.pth',
    ),
    'x8': dict(
        scale=8,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x8.pth',
    ),
}


def get_realesrgan(name, device='cpu', fp16=False, cache_dir='/tmp/rudalle'):
    assert name in MODELS
    config = MODELS[name]
    model = RealESRGAN(device, config['scale'], fp16=fp16)
    cache_dir = os.path.join(cache_dir, name)
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    model.load_weights(os.path.join(cache_dir, config['filename']))
    print(f'{name} --> ready')
    return model
