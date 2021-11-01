# -*- coding: utf-8 -*-
import os
import random

import torch
import torchvision
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def torch_tensors_to_pil_list(input_images):
    out_images = []
    for in_image in input_images:
        in_image = in_image.cpu().detach()
        out_image = torchvision.transforms.functional.to_pil_image(in_image).convert('RGB')
        out_images.append(out_image)
    return out_images


def pil_list_to_torch_tensors(pil_images):
    result = []
    for pil_image in pil_images:
        image = np.array(pil_image, dtype=np.uint8)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        result.append(image)
    return torch.cat(result, dim=0)
