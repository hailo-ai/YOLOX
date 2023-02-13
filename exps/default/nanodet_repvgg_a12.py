#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from yolox.models.repvgg import create_RepVGG_A12
from yacs.config import CfgNode

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        #self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

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

        import ipdb; ipdb.set_trace()
        in_channels = [128, 128, 256]
        backbone = create_RepVGG_A12(fpn_cfg)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = YOLOX(backbone, head)

        self.model.head.initialize_biases(1e-2)
        return self.model
