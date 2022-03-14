
import torch.nn as nn

from subnet_retraining.utils import BasicNetwork, MyGlobalAvgPool2d, make_divisible
from subnet_retraining.utils.layers import ConvLayer, IdentityLayer, LinearLayer, ResidualBlock, ResNetBottleneckBlock
from subnet_retraining.networks import ResNet50D

from icecream import ic
import torch
if __name__=='__main__':
        teacher_model = ResNet50D(
            n_classes=81,
            bn_param=(0.1, 0.1),
            dropout_rate=0,
            width_mult=1.0,
            depth_param=3,
            expand_ratio=0.35,
        )
        teacher_model=teacher_model.cuda()
        teacher_model.eval()

        x = torch.randn([10,3,128,128]).cuda()
        o = teacher_model(x)
        ic(o)
