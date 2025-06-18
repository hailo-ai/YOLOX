#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from yolox.models.nanodet_repvgg import create_RepVGG_A12
from yacs.config import CfgNode

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.act = 'relu'
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.print_interval = 100
        self.eval_interval = 4

        self.basic_lr_per_img = 0.02 / 64.0
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        cfg = dict(
            name="PAN",
            in_channels=[128, 128, 256],
            out_channels=128,
            start_level=0,
            num_outs=3,
        )
        fpn_cfg = CfgNode(cfg)

        in_channels = [256, 256, 256] # input channels to the yolohead (this is the FPN output)
        backbone = create_RepVGG_A12(fpn_cfg) # PAN also
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = YOLOX(backbone, head)

        self.model.head.initialize_biases(1e-2)
        return self.model
