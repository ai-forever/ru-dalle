# -*- coding: utf-8 -*-
import os
import torch
import pytest

from .test_vae import preprocess


def test_forward_step_tensors(small_dalle):
    bs = 4
    text_seq_length = small_dalle.get_param('text_seq_length')
    total_seq_length = small_dalle.get_param('total_seq_length')
    device = small_dalle.get_param('device')
    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))
    with torch.no_grad():
        text_input_ids = torch.tensor([
            [*range(1000, 1000 + text_seq_length - 11), 2, *[0]*10] for _ in range(bs)
        ]).long()

        image_input_ids = torch.tensor([
            [*range(5000, 5000 + total_seq_length - text_seq_length)] for _ in range(bs)
        ]).long()

        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        loss, loss_values = small_dalle.forward(input_ids, attention_mask, return_loss=True)
        assert type(loss.data.detach().item()) == float
        assert type(loss_values) == dict


@pytest.mark.parametrize('text', [
    'мальчик играет с оленем',
])
def test_forward_step_and_criterion(text, sample_image, yttm_tokenizer, vae, small_dalle):
    bs = 4
    text_seq_length = small_dalle.get_param('text_seq_length')
    total_seq_length = small_dalle.get_param('total_seq_length')
    device = small_dalle.get_param('device')

    img = sample_image.copy()
    img = preprocess(img, target_image_size=256)
    images = img.repeat(bs, 1, 1, 1).to(device)

    text = text.lower().strip()
    text_input_ids = yttm_tokenizer.encode_text(text, text_seq_length=text_seq_length)
    text_input_ids = text_input_ids.unsqueeze(0).repeat(bs, 1).to(device)

    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))
    with torch.no_grad():
        image_input_ids = vae.get_codebook_indices(images)
        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        loss, loss_values = small_dalle.forward(input_ids, attention_mask, return_loss=True)
        assert type(loss.data.detach().item()) == float
        assert type(loss_values) == dict


@pytest.mark.skipif(os.getenv('PYTEST_RUN_SKIPPED') != '1', reason='Slow, for manual run only.')
@pytest.mark.parametrize('text', [
    'рыжий котик',
])
def test_xl_forward(text, sample_image_cat, yttm_tokenizer, pretrained_vae, xl_dalle):
    text_seq_length = xl_dalle.get_param('text_seq_length')
    total_seq_length = xl_dalle.get_param('total_seq_length')
    device = xl_dalle.get_param('device')

    text = text.lower().strip()
    text_input_ids = yttm_tokenizer.encode_text(text, text_seq_length=text_seq_length)
    text_input_ids = text_input_ids.unsqueeze(0).to(device)

    img = sample_image_cat.copy()
    images = preprocess(img, target_image_size=256)

    attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
    with torch.no_grad():
        image_input_ids = pretrained_vae.get_codebook_indices(images)
        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        loss, loss_values = xl_dalle.forward(input_ids, attention_mask, return_loss=True)
        loss = loss.data.detach().item()
        assert type(loss_values) == dict
        assert type(loss) == float
        assert 3.7 < loss < 3.9


@pytest.mark.skipif(os.getenv('PYTEST_RUN_SKIPPED') != '1', reason='Slow, for manual run only.')
@pytest.mark.parametrize('text', [
    'рыжий котик',
])
def test_xxl_forward(text, sample_image_cat, yttm_tokenizer, pretrained_vae, xxl_dalle):
    checkpoint = torch.load('../../rudalle_12b_v1.bin', map_location='cpu')
    xxl_dalle.load_state_dict(checkpoint)

    text_seq_length = xxl_dalle.get_param('text_seq_length')
    total_seq_length = xxl_dalle.get_param('total_seq_length')
    device = xxl_dalle.get_param('device')

    text = text.lower().strip()
    text_input_ids = yttm_tokenizer.encode_text(text, text_seq_length=text_seq_length)
    text_input_ids = text_input_ids.unsqueeze(0).to(device)

    img = sample_image_cat.copy()
    images = preprocess(img, target_image_size=256)

    attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
    with torch.no_grad():
        image_input_ids = pretrained_vae.get_codebook_indices(images)
        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        loss, loss_values = xxl_dalle.forward(input_ids, attention_mask, return_loss=True)
        loss = loss.data.detach().item()
        assert type(loss_values) == dict
        assert type(loss) == float
        assert 3.7 < loss < 4.0
