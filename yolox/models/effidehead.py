import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from yolov6.layers.common import *
# from yolov6.assigners.anchor_generator import generate_anchors
# from yolov6.utils.general import dist2bbox
from .common import ConvBNAct, dist2bbox, generate_anchors

'''
Default configuration for head:
head=dict(
    type='EffiDeHead',
    in_channels=[128, 256, 512],
    num_layers=3,
    begin_indices=24,
    anchors=1,
    out_indices=[17, 20, 23],
    strides=[8, 16, 32],
    iou_type='siou',
    use_dfl=False,
    reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
'''

# TODO: Amit -- Remove use_dfl, reg_max, anchors(?), inplace(?)


class Yolov6Head(nn.Module):
    def __init__(self,
                 num_classes=80,
                 anchors=1,
                 num_layers=3,
                 inplace=True,
                 head_layers=None,
                 use_dfl=False,
                 reg_max=0):  # detection layer
    #     self,
    #     num_classes,
    #     width=1.0,
    #     strides=[8, 16, 32],
    #     in_channels=[256, 512, 1024],
    #     act="silu",
    #     depthwise=False,

        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2  # TODO: Amit -- Can be deleted
        self.inplace = inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])

    def initialize_biases(self, prior_prob):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x, labels=None, imgs=None):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            loss = torch.tensor([26.4065], requires_grad=True)
            iou_loss = torch.tensor([4.7327], requires_grad=True)
            conf_loss = torch.tensor([19.7182], requires_grad=True)
            cls_loss = torch.tensor([1.9557], requires_grad=True)
            l1_loss = torch.tensor([0.0], requires_grad=True)
            num_fg = 1.1578947368421053
            # loss
            # ipdb> tensor(23.2012, device='cuda:0', grad_fn=<AddBackward0>)
            # reg_weight
            # ipdb> 5.0
            # loss_iou
            # ipdb> tensor(0.9540, device='cuda:0', grad_fn=<DivBackward0>)
            # loss_obj
            # ipdb> tensor(16.7003, device='cuda:0', grad_fn=<DivBackward0>)
            # loss_cls
            # ipdb> tensor(1.7312, device='cuda:0', grad_fn=<DivBackward0>)
            # loss_l1
            # ipdb> 0.0
            # num_fg / max(num_gts, 1)
            # ipdb> 1.04
            # import ipdb; ipdb.set_trace()
            return loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg
            # return x, cls_score_list, reg_distri_list
            # In YOLOX we want to return: loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True)
            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=0):
    head_layers = nn.Sequential(
        # stem0
        ConvBNAct(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        ConvBNAct(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        ConvBNAct(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem1
        ConvBNAct(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        ConvBNAct(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        ConvBNAct(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        ConvBNAct(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        ConvBNAct(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        ConvBNAct(
            in_channels=channels_list[2],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )
    return head_layers