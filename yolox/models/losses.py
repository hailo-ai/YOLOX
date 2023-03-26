#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .tal_assigner import TaskAlignedAssigner
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import dist2bbox, generate_anchors, xywh2xyxy


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


############################################################
# Added by us (Amit) TODO: Think if to drop this header
############################################################

class IOUlossYolov6:
    """ Calculate IoU loss.
    """

    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
                torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ComputeLoss:
    '''Loss computation func.'''

    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 input_size=(640, 640),  # (height, width)
                 warmup_epoch=4,
                 reg_max=0,  # 16
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5}
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.input_size = input_size
        # self.warmup_epoch = warmup_epoch
        # self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(
            topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.iou_type).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,
        targets
        # epoch_num,
        # step_num
    ):

        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size,
                             self.grid_cell_offset, device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        if isinstance(self.input_size, int):
            gt_bboxes_scale = torch.full((1, 4), self.input_size).type_as(pred_scores)
        else:  # Tuple input_size
            WH_shape = tuple(reversed(self.input_size))
            gt_bboxes_scale = torch.tensor(WH_shape * 2).view(1, 4).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # Normalize targets bbox (xywh):
        # targets[:, :, 1:] = targets[:, :, 1:] / 640.0
        # targets[:, :, 1::2] = targets[:, :, 1::2] / self.input_size[1]  # normalize (xw) by W
        # targets[:, :, 2::2] = targets[:, :, 2::2] / self.input_size[0]  # normalize (yh) by H
        targets[:, :, 1:] /= gt_bboxes_scale
        # targets
        targets = self._preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy

        try:
            # if epoch_num < self.warmup_epoch:
            #     target_labels, target_bboxes, target_scores, fg_mask = \
            #         self.warmup_assigner(
            #             anchors,
            #             n_anchors_list,
            #             gt_labels,
            #             gt_bboxes,
            #             mask_gt,
            #             pred_bboxes.detach() * stride_tensor)
            # else:
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            # if epoch_num < self.warmup_epoch:
            #     _anchors = anchors.cpu().float()
            #     _n_anchors_list = n_anchors_list
            #     _gt_labels = gt_labels.cpu().float()
            #     _gt_bboxes = gt_bboxes.cpu().float()
            #     _mask_gt = mask_gt.cpu().float()
            #     _pred_bboxes = pred_bboxes.detach().cpu().float()
            #     _stride_tensor = stride_tensor.cpu().float()

            #     target_labels, target_bboxes, target_scores, fg_mask = \
            #         self.warmup_assigner(
            #             _anchors,
            #             _n_anchors_list,
            #             _gt_labels,
            #             _gt_bboxes,
            #             _mask_gt,
            #             _pred_bboxes * _stride_tensor)

            # else:
            _pred_scores = pred_scores.detach().cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _anchor_points = anchor_points.cpu().float()
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _stride_tensor = stride_tensor.cpu().float()

            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                    _pred_scores,
                    _pred_bboxes * _stride_tensor,
                    _anchor_points,
                    _gt_labels,
                    _gt_bboxes,
                    _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        # Dynamic release GPU memory
        # if step_num % 10 == 0:
        #     torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels,
                                    torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # print(f"target_scores shape = {target_scores.cpu().detach().numpy().shape}, target_scores_sum = {target_scores_sum.cpu().detach().numpy():.2f}")
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss = self.loss_weight['class'] * loss_cls + \
            self.loss_weight['iou'] * loss_iou + \
            self.loss_weight['dfl'] * loss_dfl

        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                       (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                       (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    def _preprocess(self, targets, batch_size, scale_tensor):
        '''
        '''
        # Prepare YOLOx labels [B, max_labels, 5] (5 = cls+bbox) to Yolov6 [valid_labels, 6] (6 = img_idx, cls, bbox)
        # nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # will result in [B], each with how many valid labels inside
        _max_labels = targets.shape[1]
        mask = ~(targets == 0).all(dim=-1)
        batch_index = torch.arange(batch_size).repeat(
            _max_labels, 1).transpose(0, 1).flatten().unsqueeze(1)
        targets_filtered = targets[mask]
        mask_flatten = mask.view(batch_size * _max_labels, 1)
        batch_index_filtered = batch_index[mask_flatten].unsqueeze(1).to(targets.device)
        targets = torch.cat([batch_index_filtered, targets_filtered.reshape(-1, 5)], dim=-1)

        # as yolov6
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(
            map(lambda l: l + [[-1, 0, 0, 0, 0]]*(max_len - len(l)), targets_list)))[:, 1:, :]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)  # Rescale back to HxW (unnormalize)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(),
                    gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUlossYolov6(box_format='xyxy', iou_type=iou_type, eps=1e-10)

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
            loss_dfl = torch.tensor(0.).to(pred_dist.device)
        else:
            loss_iou = torch.tensor(0.).to(pred_dist.device)
            loss_dfl = torch.tensor(0.).to(pred_dist.device)

        return loss_iou, loss_dfl
