#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Hailo Inc.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.models.yolox_hailo_fpn import YOLOxHailoFPN
from yolox.models.effidehead import YoloxHailoHead, build_effidehead_layer


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (736, 960)  # (height, width)
        self.test_size = (736, 960)  # (height, width)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.act = 'relu'
        self.output_dir = './yolox_hailo_outputs_4cls_prune'
        self.print_interval = 100
        self.eval_interval = 5
        self.max_epoch = 400
        self.data_num_workers = 8
        self.basic_lr_per_img = 0.02 / 64.0

        # Data
        self.num_classes = 4
        self.data_dir = '/fastdata/coco/coco_4classes/'
        self.train_ann = "instances_val2017_4cls.json"
        self.val_ann = "instances_val2017_4cls.json"
        self.test_ann = "instances_val2017_4cls.json"
        self.name = 'val2017_4cls/images/'
        self.rgb = True

        # Sparsity
        self.recipe = None
        self.recipe_args = None
        self.test_conf = 0.05
        self.warmup_epochs = 0

        # Loss
        self.iou_type = 'siou'
        # backbone config
        self.bb_channels_list = [128, 256, 512, 1024]  # bb_channels * width = [32, 64, 128, 256]
        self.bb_num_repeats_list = [9, 15, 21, 12]  # bb_num_repeats * depth = [3, 5, 7, 4]
        # neck config
        self.neck_channels_list = [256, 128, 128, 256, 256, 512]  # neck_channels * width = [64, 32, 32, 64, 64, 128]
        self.neck_num_repeats_list = [9, 12, 12, 9]  # neck_num_repeats * depth = [3, 4, 4, 3]
        # head config
        self.head_channels_list = [128, 256, 512]  # head_channels * width = [32, 64, 128]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX
            backbone = YOLOxHailoFPN(self.depth, self.width,
                                     self.bb_channels_list, self.bb_num_repeats_list,
                                     self.neck_channels_list, self.neck_num_repeats_list)

            assert len(self.head_channels_list) == 3, "Number of head branches should be 3"
            head_channels_list = [int(round(ch * self.width)) for ch in self.head_channels_list]
            head_layers = build_effidehead_layer(head_channels_list, self.num_classes)
            head = YoloxHailoHead(self.num_classes, len(head_channels_list),
                                  head_layers=head_layers, input_size=self.input_size, iou_type=self.iou_type)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
