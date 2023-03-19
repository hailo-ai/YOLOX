
import torch.nn as nn
import torch
from .repvgg import RepVGGBlock
from .common import ConvBNAct, Transpose



def make_repvgg_stage(in_channels, out_channels, num_blocks, stride=1, deploy=False):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    for stride in strides:
        blocks.append(RepVGGBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=stride, padding=1, groups=1, deploy=deploy, use_se=False))
        in_channels = out_channels
    return nn.Sequential(*blocks)


class PANRepVGGNeck(nn.Module):
    def __init__(
        self,
        num_repeats=None,  # [12, 12, 12, 12]
        width_multiplier=None,  # [0.25, 0.25, 0.25, 0.25]
        channels_list=None
        ):

        super().__init__()
        neck_num_repeats = [int(r * w) for r, w in (zip(num_repeats, width_multiplier))]  # [3, 3, 3, 3]

        assert channels_list is not None
        assert num_repeats is not None

        self.neck_p4 = make_repvgg_stage(192, 64, neck_num_repeats[0])
        self.neck_p3 = make_repvgg_stage(96, 32, neck_num_repeats[1])
        self.neck_n3 = make_repvgg_stage(64, 64, neck_num_repeats[2])
        self.neck_n4 = make_repvgg_stage(128, 128, neck_num_repeats[3])

        self.align_channels0 = ConvBNAct(in_channels=256, out_channels=64, kernel_size=1)
        self.upsample0 = Transpose()
        self.align_channels1 = ConvBNAct(in_channels=64, out_channels=32, kernel_size=1)
        self.upsample1 = Transpose()
        self.downsample2 = nn.Sequential(
            ConvBNAct(in_channels=32, out_channels=32, kernel_size=3),  # Added to Tal's request
            ConvBNAct(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        )
        self.downsample1 = ConvBNAct(in_channels=64, out_channels=64, kernel_size=3, stride=2)

    def forward(self, input):
        """
        Args:
            input (Tuple[Tensor]): backbone output features:
                outputs[0] - /8 resolution feature map
                outputs[1] - /16 resolution feature map
                outputs[2] - /32 resolution feature map

        Returns:
            List[Tensor]: neck output features:
                outputs[0] - /8 resolution feature map
                outputs[1] - /16 resolution feature map
                outputs[2] - /32 resolution feature map
        """
        (x2, x1, x0) = input

        fpn_out0 = self.align_channels0(x0)
        upsample_features0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_features0, x1], 1)
        f_out0 = self.neck_p4(f_concat_layer0)

        fpn_out1 = self.align_channels1(f_out0)
        upsample_features1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_features1, x2], 1)
        pan_out2 = self.neck_p3(f_concat_layer1)

        down_features1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_features1, fpn_out1], 1)
        pan_out1 = self.neck_n3(p_concat_layer1)

        down_features0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_features0, fpn_out0], 1)
        pan_out0 = self.neck_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs