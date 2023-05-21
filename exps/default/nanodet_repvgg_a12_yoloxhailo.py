#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from yolox.models.nanodet_repvgg import create_RepVGG_A12
from yacs.config import CfgNode
from yolox.models.effidehead import YoloxHailoHead, build_effidehead_layer


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.act = 'relu'
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.print_interval = 50
        self.eval_interval = 10
        self.max_epoch = 300
        self.data_num_workers = 4
        self.basic_lr_per_img = 0.02 / 64.0
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5

        self.input_size = (480, 640)  # (height, width)
        self.test_size = (480, 640)  # (height, width)
        self.output_dir = './nanodet_outputs_coco'
        self.data_dir = '/fastdata/users/COCO'
        # Loss
        self.iou_type = 'siou'

    def get_model(self):
        from yolox.models import YOLOX
        cfg = dict(
            name="PAN_conv1x1down",
            in_channels=[128, 128, 256],
            out_channels=128,
            start_level=0,
            num_outs=3,
        )
        fpn_cfg = CfgNode(cfg)
        backbone = create_RepVGG_A12(fpn_cfg)  # PAN also
        head_channels_list = [128, 128, 128]
        head_layers = build_effidehead_layer(head_channels_list, self.num_classes)
        head = YoloxHailoHead(self.num_classes, len(head_channels_list),
                          head_layers=head_layers, input_size=self.input_size, iou_type=self.iou_type)
        self.model = YOLOX(backbone, head)

        self.model.head.initialize_biases(1e-2)
        return self.model