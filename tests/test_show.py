# -*- coding: utf-8 -*-
from rudalle.pipelines import show


def test_show(sample_image):
    img = sample_image.copy()
    img = img.resize((256, 256))
    pil_images = [img]*5
    show(pil_images, nrow=2, save_dir='/tmp/pics', show=False)
