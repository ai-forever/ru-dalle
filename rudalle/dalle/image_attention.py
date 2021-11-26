# -*- coding: utf-8 -*-

import torch


def _init_mask(text_tokens, image_tokens_per_dim, is_bool_mask=False):
    attn_size = text_tokens + image_tokens_per_dim**2
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool if is_bool_mask else torch.float32))
    return mask


def get_row_mask(text_tokens=256, image_tokens_per_dim=32, is_bool_mask=False):
    mask = _init_mask(text_tokens, image_tokens_per_dim, is_bool_mask=is_bool_mask)
    step = image_tokens_per_dim + 1
    for col in range(text_tokens, mask.size(1)):
        mask[col + step:, col] = False if is_bool_mask else 0.0
    return mask


def get_col_mask(text_tokens=256, image_tokens_per_dim=32, is_bool_mask=False):
    mask = _init_mask(text_tokens, image_tokens_per_dim, is_bool_mask=is_bool_mask)
    step = image_tokens_per_dim - 1
    for col in range(text_tokens, mask.size(1)):
        for i in range(1, mask.size(0), step+1):
            mask[col + i: col + i + step, col] = False if is_bool_mask else 0.0
    return mask


def get_conv_mask(text_tokens=256, image_tokens_per_dim=32, kernel=11, is_bool_mask=False):
    mask = _init_mask(text_tokens, image_tokens_per_dim, is_bool_mask=is_bool_mask)
    shift = kernel // 2
    for pos in range(text_tokens, mask.size(1)):
        mask[pos+1:, pos] = False if is_bool_mask else 0.0
        img = torch.zeros(image_tokens_per_dim, image_tokens_per_dim)
        pixel_id = pos - text_tokens
        row = pixel_id // image_tokens_per_dim
        col = pixel_id % image_tokens_per_dim
        for r in range(-shift, shift+1):
            for c in range(-shift, shift+1):
                c_abs = (c + col) % image_tokens_per_dim
                r_abs = (r + row) % image_tokens_per_dim
                img[r_abs, c_abs] = 0.2
                cell_id = r_abs * image_tokens_per_dim + c_abs
                if text_tokens + cell_id > pos:
                    mask[text_tokens + cell_id, pos] = True if is_bool_mask else 1.0

        img[row, col] = 1.0
    return mask
