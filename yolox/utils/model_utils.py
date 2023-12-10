#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
    "calc_sparsity",
    "dummy_prune_ckpt",
    "random_prune_layer",
    "dummy_prune_layer",
]


def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def calc_sparsity(model_dict, logger):
    weights_layers_num, total_weights, total_zeros = 0, 0, 0
    for k, v in model_dict.items():
        if k.startswith('backbone.') and k.endswith('weight'):
            weights_layers_num += 1
            total_weights += v.numel()
            total_zeros += (v.numel() - v.count_nonzero())
            zeros_ratio = (v.numel() - v.count_nonzero()) / v.numel() * 100.0
    overall_sparsity = 100 * total_zeros / total_weights
    logger.info(f"Overall Sparsity is roughly: {overall_sparsity:.1f}%")
    return overall_sparsity


def dummy_prune_ckpt(ckpt, logger, prune_ratio=0.5, random_prune=False):
    for k, v in ckpt.items():
        if k.startswith('backbone.') and k.endswith('.rbr_dense.conv.weight'):
            if random_prune:  # Sparsify layer randomly:
                v = random_prune_layer(v, prune_ratio)
            else:  # Sparsify layer according to magnitude:
                v = dummy_prune_layer(v, prune_ratio)
    calc_sparsity(ckpt, logger)
    return ckpt


def random_prune_layer(layer, prune_ratio=0.5):
    """
    Randomly prune (set to zero) a fraction of elements in a PyTorch tensor.

    Args:
        layer (torch.Tensor): Input tensor of shape [B, C, H, W].
        prune_ratio (float): Fraction of elements to set to zero.

    Returns:
        torch.Tensor: Pruned tensor with the same shape as the input.
    """
    # Determine the number of elements to prune
    num_elements = layer.numel()
    num_prune = int(prune_ratio * num_elements)

    # Create a mask with zeros and ones to select the elements to prune
    mask = torch.ones(num_elements)
    mask[:num_prune] = 0
    mask = mask[torch.randperm(num_elements)]  # Shuffle the mask randomly
    mask = mask.view(layer.shape)

    # Apply the mask to the input tensor to prune it
    layer *= mask
    return layer


def dummy_prune_layer(layer, prune_ratio=0.5):
    # Flatten the tensor
    flattened_layer = layer.flatten()
    # Get the absolute values
    abs_values = torch.abs(flattened_layer)
    # Get indices sorted by absolute values
    sorted_indices = torch.argsort(abs_values)
    # Determine the threshold index
    threshold_index = int(prune_ratio * len(sorted_indices))
    # Set values below the threshold to zero
    flattened_layer[sorted_indices[:threshold_index]] = 0
    # Reshape the tensor back to its original shape
    pruned_tensor = flattened_layer.reshape(layer.shape)

    return pruned_tensor
