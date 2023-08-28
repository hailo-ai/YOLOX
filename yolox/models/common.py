import torch.nn as nn
import torch

class ConvBNAct(nn.Module):
    '''Conv2d + BN + Activation (ReLU)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Transpose(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample_transpose = torch.nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # self.upsample_transpose = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample_transpose(x)


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''
    Convert distance (ltrb) to bbox (xywh or xyxy) according to wanted format
    Notice: xywh is (xcycwh)
    '''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    elif box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    else:
        assert False, f"Unknown bbox format {box_format}"
    return bbox


def xywh2xyxy(bboxes):
    '''
    Convert bbox from xcycwh format to xyxy format
    '''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5  # x_center to x1
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5  # y_center to y1
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]  # x2 = x1 + w
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]  # y2 = y1 + h
    return bboxes


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False):
    '''
    Generate anchors from features
    '''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
        stride_tensor = torch.cat(stride_tensor)
        anchor_points = torch.cat(anchor_points)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            anchor_points.append(anchor_point.reshape([-1, 2]))
            anchors.append(anchor.reshape([-1, 4]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor
