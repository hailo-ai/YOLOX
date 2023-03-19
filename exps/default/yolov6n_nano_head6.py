#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hailo Inc.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.models.yolov6_fpn import YOLOv6FPN
from yolox.models.effidehead import Yolov6Head, build_effidehead_layer

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
        self.print_interval = 50
        self.eval_interval = 2
        self.max_epoch = 300
        self.data_num_workers = 6
        self.basic_lr_per_img = 0.03 / 64.0    # 0.03 / 64.0  # 0.0005 / 64 
        # self.no_aug_epochs = 50
        # self.warmup_epochs = 0

        # self.data_dir = '/fastdata/users/COCO'
        self.data_dir = '/fastdata/users/coco_pana'
        self.train_ann = "coco_pana_val.json"
        self.val_ann = "coco_pana_val.json"
        self.test_ann = "coco_pana_val.json"


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
            num_anchors = 1
            head_channels_list = [32, 64, 128]
            backbone = YOLOv6FPN(self.depth, self.width, act=self.act)
            head_layers = build_effidehead_layer(head_channels_list, num_anchors, self.num_classes)
            head = Yolov6Head(self.num_classes, num_anchors, len(head_channels_list), head_layers=head_layers)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
