import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import ConvBNAct, dist2bbox, generate_anchors
from yolox.models.losses import CalculateLoss

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
    )
'''


class YoloxHailoHead(nn.Module):
    def __init__(self,
                 num_classes=80,
                 num_layers=3,
                 head_layers=None,
                 input_size=(640, 640),  # (height, width)
                 iou_type='siou'
                 ):

        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.proj_conv = nn.Conv2d(1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.compute_loss = CalculateLoss(num_classes=num_classes,
                                        input_size=input_size,
                                        iou_type=iou_type)

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
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

        self.proj = nn.Parameter(torch.linspace(0, 0, 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x, labels=None, imgs=None):
        '''
        Args: x - input images batch, after preprocess
              labels - labels batch, after preprocess

        output: if training, returns losses
                else (eval), returns evaluation
        '''
        x = list(x)
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

            cls_score = torch.cat(cls_score_list, axis=1)
            reg_distri = torch.cat(reg_distri_list, axis=1)
            preds = [x, cls_score, reg_distri]
            total_loss, loss_items = self.compute_loss(preds, labels)
            iou_loss = loss_items[0]
            cls_loss = loss_items[2]
            conf_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(iou_loss.device)
            l1_loss = 0.0
            num_fg = 1.0
            return total_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg
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


def build_effidehead_layer(channels_list, num_classes):
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
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=4,
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
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=4,
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
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=4,
            kernel_size=1
        )
    )
    return head_layers
