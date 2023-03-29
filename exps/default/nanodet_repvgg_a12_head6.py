#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from yolox.models.nanodet_repvgg import create_RepVGG_A12
from yacs.config import CfgNode
from yolox.models.effidehead import Yolov6Head, build_effidehead_layer

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

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        cfg = dict(
            name="PAN_conv1x1down",
            in_channels=[128, 128, 256],
            out_channels=128,
            start_level=0,
            num_outs=3,
        )
        fpn_cfg = CfgNode(cfg)

        in_channels = [256, 256, 256] # input channels to the yolohead (this is the FPN output)
        backbone = create_RepVGG_A12(fpn_cfg) # PAN also
        # head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        num_anchors = 1
        # head_channels_list = [32, 64, 128]
        head_channels_list = [128, 128, 128]
        head_layers = build_effidehead_layer(head_channels_list, num_anchors, self.num_classes)
        head = Yolov6Head(self.num_classes, num_anchors, len(head_channels_list), head_layers=head_layers)
        self.model = YOLOX(backbone, head)

        self.model.head.initialize_biases(1e-2)
        return self.model
