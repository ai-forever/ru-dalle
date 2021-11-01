# -*- coding: utf-8 -*-
import torch
import pytest

from .test_vae import preprocess


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
