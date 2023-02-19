#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hailo Inc.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.models.yolov6_fpn import YOLOv6FPN


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.test_size = (640, 640)
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.act = 'relu'
        self.output_dir = './yolov6n_outputs_nano'
        self.print_interval = 50
        self.eval_interval = 5
        # self.max_epoch = 24
        self.data_num_workers = 6
        self.basic_lr_per_img = 0.03 / 64.0

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOXHead
            in_channels = [64, 128, 256]
            width = 0.5
            backbone = YOLOv6FPN(self.depth, self.width, act=self.act)
            head = YOLOXHead(self.num_classes, width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
