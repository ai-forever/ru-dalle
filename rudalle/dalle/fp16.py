# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)


class FP16Module(nn.Module):
    def __init__(self, module):
        super(FP16Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    def get_param(self, item):
        return self.module.get_param(item)

    def to(self, device, *args, **kwargs):
        self.module.to(device)
        return super().to(device, *args, **kwargs)
