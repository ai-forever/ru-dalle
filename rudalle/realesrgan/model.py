# -*- coding: utf-8 -*-
# Source: https://github.com/boomb0om/Real-ESRGAN-colab

import torch
import numpy as np
from PIL import Image

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image
from rudalle.dalle.fp16 import FP16Module


class RealESRGAN:
    def __init__(self, device, scale=4, fp16=False):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        self.fp16 = fp16

    def load_weights(self, model_path):
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        if self.fp16:
            self.model = FP16Module(self.model)
        self.model.to(self.device)

    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(lr_image, patch_size=patches_size,
                                                                padding_size=padding)
        if self.fp16:
            img = torch.HalfTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()
        else:
            img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i + batch_size])), 0)

        sr_image = res.permute((0, 2, 3, 1)).cpu().clamp_(0, 1)
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(np_sr_image, padded_image_shape=padded_size_scaled,
                                     target_shape=scaled_image_shape, padding_size=padding * scale)
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img
