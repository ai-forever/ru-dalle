# -*- coding: utf-8 -*-
from rudalle.pipelines import super_resolution


def test_super_resolution(sample_image, realesrgan):
    img = sample_image.copy()
    img = img.resize((32, 32))
    sr_img = super_resolution([img], realesrgan)[0]
    assert sr_img.size[0] == 32*2
    assert sr_img.size[1] == 32*2
