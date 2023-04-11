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
        self.input_size = (480, 640)
        self.test_size = (480, 640)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        # --------------- transform config ----------------- #
        # self.degrees = 10.0  # default
        # self.translate = 0.1  # default
        # self.scale = (0.1, 2)
        # self.mosaic_scale = (0.8, 1.6)
        # self.shear = 2.0  # default
        # self.perspective = 0.0
        # self.enable_mixup = True  # default


        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.act = 'relu'
        self.output_dir = './yolov6n_outputs_coco'
        self.print_interval = 100
        self.eval_interval = 5
        self.max_epoch = 300
        self.data_num_workers = 6
        self.basic_lr_per_img = 0.03 / 64.0    # 0.03 / 64.0  # 0.0005 / 64
        # self.no_aug_epochs = 50
        # self.warmup_epochs = 0

        self.data_dir = '/fastdata/users/COCO'

        # backbone config
        self.bb_channels_list = [128, 256, 512, 1024]  # bb_channels * width = [32, 64, 128, 256]
        self.bb_num_repeats_list = [9, 15, 21, 12]  # bb_num_repeats * depth = [3, 5, 7, 4]
        # neck config
        self.neck_channels_list = [256, 128, 128, 256, 256, 512]  # neck_channels * width = [64, 32, 32, 64, 64, 128]
        self.neck_num_repeats_list = [9, 12, 12, 9]  # neck_num_repeats * depth = [3, 4, 4, 3]
        # head config
        self.head_width = 0.5
        self.head_channels_list = [64, 128, 256]

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOXHead
            backbone = YOLOv6FPN(self.depth, self.width,
                                 self.bb_channels_list, self.bb_num_repeats_list,
                                 self.neck_channels_list, self.neck_num_repeats_list)

            assert len(self.head_channels_list) == 3, "Number of head branches should be 3"
            head = YOLOXHead(self.num_classes, self.head_width, in_channels=self.head_channels_list, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
