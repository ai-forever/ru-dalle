# -*- coding: utf-8 -*-
import PIL
import pytest
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


@pytest.mark.parametrize('target_image_size', [128, 192, 256])
def test_decode_vae(vae, sample_image, target_image_size):
    img = sample_image.copy()
    img = preprocess(img, target_image_size=target_image_size)
    with torch.no_grad():
        img_seq = vae.get_codebook_indices(img)
        out_img = vae.decode(img_seq)
    assert out_img.shape == (1, 3, target_image_size, target_image_size)


@pytest.mark.parametrize('target_image_size', [128, 192, 256])
def test_reconstruct_vae(vae, sample_image, target_image_size):
    img = sample_image.copy()
    with torch.no_grad():
        x_vqgan = preprocess(img, target_image_size=target_image_size)
        output = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), vae.model)
    assert output.shape == (1, 3, target_image_size, target_image_size)


@pytest.mark.parametrize('target_image_size', [256])
def test_reconstruct_dwt_vae(dwt_vae, sample_image, target_image_size):
    img = sample_image.copy()
    with torch.no_grad():
        x_vqgan = preprocess(img, target_image_size=target_image_size)
        output = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), dwt_vae.model)
    assert output.shape == (1, 3, target_image_size*2, target_image_size*2)


def preprocess(img, target_image_size=256):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x


def reconstruct_with_vqgan(x, model):
    z, _, [_, _, _] = model.encode(x)
    print(f'VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}')
    xrec = model.decode(z)
    return xrec
