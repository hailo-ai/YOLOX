#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from .repvgg import RepVGG
from .panrepvgg_neck import PANRepVGGNeck
from .network_blocks import BaseConv


class YOLOv6FPN(nn.Module):
    """
    YOLOv6FPN module.
    """

    def __init__(
        self,
        depth = 1.0,
        width = 1.0,
        act="silu",
    ):
        super().__init__()
        
        self.channels_list = [32, 64, 128, 256]  # , 64, 32, 32, 64, 64, 128]
        self.num_repeats = [3, 4, 6, 3]
        width_multiplier = [0.25] * len(self.channels_list)
        
        self.backbone = RepVGG(num_blocks=self.num_repeats, width_multiplier=width_multiplier)

        self.neck_num_repeats = [12, 12, 12, 12]
        self.neck_out_channels = [256, 128, 128, 256, 256, 512]
        self.neck = PANRepVGGNeck(num_repeats=self.neck_num_repeats, width_multiplier=width_multiplier, channels_list=self.neck_out_channels)


    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features.
        """
        # backbone
        backbone_features = self.backbone(inputs)
        # neck
        neck_features = self.neck(backbone_features)
        return neck_features
