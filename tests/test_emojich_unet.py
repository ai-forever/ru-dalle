# -*- coding: utf-8 -*-
import numpy as np

from rudalle.pipelines import convert_emoji_to_rgba


def test_convert_emoji_to_rgba(sample_image, emojich_unet):
    img = sample_image.copy()
    img = img.resize((512, 512))
    rgba_images, runs = convert_emoji_to_rgba([img], emojich_unet, score_thr=0.99)
    assert len(runs) == len(rgba_images)
    rgba_img = rgba_images[0]
    assert rgba_img.size[0] == 512
    assert rgba_img.size[1] == 512
    assert np.array(rgba_img).shape[-1] == 4
    assert runs[0] in ['unet', 'classic']
