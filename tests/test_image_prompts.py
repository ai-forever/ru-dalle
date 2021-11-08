# -*- coding: utf-8 -*-
import pytest

from rudalle.image_prompts import ImagePrompts


@pytest.mark.parametrize('borders, crop_first', [
    ({'up': 4, 'right': 0, 'left': 0, 'down': 0}, False),
    ({'up': 4, 'right': 0, 'left': 0, 'down': 0}, True),
    ({'up': 4, 'right': 3, 'left': 3, 'down': 3}, False)
])
def test_image_prompts(sample_image, vae, borders, crop_first):
    img = sample_image.copy()
    img = img.resize((256, 256))
    image_prompt = ImagePrompts(img, borders, vae, crop_first=crop_first)
    assert image_prompt.image_prompts.shape[1] == 32 * 32
    assert len(image_prompt.image_prompts_idx) == (borders['up'] + borders['down']) * 32 \
        + (borders['left'] + borders['right']) * (32 - borders['up'] - borders['down'])
