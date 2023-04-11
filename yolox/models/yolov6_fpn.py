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
        depth=1.0,
        width=1.0,
        bb_channels_list=None,
        bb_num_repeats_list=None,
        neck_channels_list=None,
        neck_num_repeats_list=None
    ):
        super().__init__()
        assert bb_channels_list is not None
        assert bb_num_repeats_list is not None
        assert neck_channels_list is not None
        assert neck_num_repeats_list is not None

        width_multiplier = [width] * len(bb_channels_list)
        bb_num_repeats_final = [int(round(nr * depth)) for nr in bb_num_repeats_list]

        self.backbone = RepVGG(num_blocks=bb_num_repeats_final,
                               out_channels=bb_channels_list,
                               width_multiplier=width_multiplier)
        self.neck = PANRepVGGNeck(depth, width, neck_num_repeats_list,
                                  bb_channels_list, neck_channels_list)

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
