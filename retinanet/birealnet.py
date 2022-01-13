import os

import math
from torchvision.ops import nms

import torch
import torch.nn as nn
import torch.nn.functional as F

#from retinanet.model import PyramidFeatures, RegressionModel, ClassificationModel
from retinanet.utils import BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.binary_units import BinaryActivation, HardBinaryConv, BinaryLinear


__all__ = ['birealnet18', 'birealnet34']

'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# may be I need to use this conv1x1 in the birealnet downsmaple opearation
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
'''


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_bin=False):
     if is_bin:
         return HardBinaryConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
     return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def activation(inplace=False, is_bin=False):
    if is_bin:
        return BinaryActivation()
    return nn.ReLU(inplace=inplace)

'''
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y
'''

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, is_bin=False):
        super(PyramidFeatures, self).__init__()

        #print('birealnet.py/FPN init')
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, is_bin=is_bin)
        self.act5 = activation(is_bin=is_bin)

        # add P5 elementwise to C4
        self.P4_1 = conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, is_bin=is_bin)
        self.act4 = activation(is_bin=is_bin)

        # add P4 elementwise to C3
        self.P3_1 = conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, is_bin=is_bin)
        self.act3 = activation(is_bin=is_bin)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1, is_bin=is_bin)
        self.act6 = activation(is_bin=is_bin)
        #self.avgPool6 = nn.AdaptiveAvgPool2d((10,13))
        self.P6_down = conv2d(C5_size, feature_size, kernel_size=1, stride=2, padding=0)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = activation(is_bin=is_bin)
        self.P7_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1, is_bin=is_bin)
        self.avgPool7 = nn.AdaptiveAvgPool2d((7,5))
        self.P7_down = conv2d(feature_size, feature_size, kernel_size=1, stride=2, padding=0)

    def forward1(self, inputs):
        #print('birealnet.py/FPN forward')
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_act = self.act5(P5_x)
        P5_x = self.P5_2(P5_act) + P5_x

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_act = self.act4(P4_x)
        P4_x = self.P4_2(P4_act) + P4_x

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_act = self.act3(P3_x)
        P3_x = self.P3_2(P3_act) + P3_x

        C5_act = self.act6(C5)
        x6 = self.P6(C5_act)
        #y6 = self.avgPool6(C5)
        y6 = self.P6_down(C5)
        #ic(C5.shape)
        #ic(x6.shape)
        #ic(y6.shape)
        #print('---')
        P6_x = x6 + y6

        P7_x = self.P7_1(P6_x)
        x7= self.P7_2(P7_x)
        y7 = self.P7_down(P6_x)
        #ic(P6_x.shape)
        #ic(x7.shape)
        #ic(y7.shape)
        #print('***')
        P7_x = x7 + y7


        return [P3_x, P4_x, P5_x, P6_x, P7_x]


    def forward2(self, inputs):
        #print('birealnet.py/FPN forward standard')
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

    def forward(self, inputs):
        #if match_forwards:
        #    return self.forward2(inputs)
        #return self.forward1(inputs)
        return self.forward2(inputs)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, is_bin=False):
        super(RegressionModel, self).__init__()

        #print('birealnet.py/Reg init')
        # not binarizing because treating this as the first layer dealing with input
        self.conv1 = conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = activation()

        self.conv2 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act2 = activation(is_bin=is_bin)

        self.conv3 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act3 = activation(is_bin=is_bin)

        self.conv4 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act4 = activation(is_bin=is_bin)

        self.output = conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward1(self, x):

        #print('birealnet.py/ Reg forward')
        out1 = self.conv1(x)
        #out = self.act1(out)

        out = self.act2(out1)
        out2 = self.conv2(out) + out1
        #out = self.act2(out)


        out = self.act3(out2)
        out3 = self.conv3(out) + out2
        #out = self.act3(out)

        out = self.act4(out3)
        out = self.conv4(out) + out3
        #out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

    def forward2(self, x):
        #print('birealnet.py/Reg forward standard')
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

    def forward(self, inputs):
        #if match_forwards:
        #    return self.forward2(inputs)
        #return self.forward1(inputs)
        return self.forward2(inputs)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, is_bin=False):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        #print('birealnet.py/Class init')
        self.conv1 = conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = activation()

        self.conv2 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act2 = activation(is_bin=is_bin)

        self.conv3 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act3 = activation(is_bin=is_bin)

        self.conv4 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act4 = activation(is_bin=is_bin)

        self.output = conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward1(self, x):

        #print('birealnet.py/class forward')
        out1 = self.conv1(x)
        #out = self.act1(out)

        out = self.act2(out1)
        out2 = self.conv2(out) + out1
        #out = self.act2(out)

        out = self.act3(out2)
        out3 = self.conv3(out) + out2
        #out = self.act3(out)

        out = self.act4(out3)
        out = self.conv4(out) + out3
        #out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


    def forward2(self, x):
        #print('birealnet.py/CLass forward standard')
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

    def forward(self, inputs):
        #if match_forwards:
        #    return self.forward2(inputs)
        #return self.forward1(inputs)
        return self.forward2(inputs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        self.out_channels = planes

    def forward(self, x):
        residual = x

        out = self.binary_activation(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if block == BasicBlock:
            # fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
            #              self.layer4[layers[3] - 1].conv2.out_channels]
            fpn_sizes = [self.layer2[layers[1] - 1].out_channels, self.layer3[layers[2] - 1].out_channels,
                         self.layer4[layers[3] - 1].out_channels]
        #elif block == Bottleneck:
        #    fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
        #                 self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError("Block type {} not understood".format(block))

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], is_bin=True)

        self.regressionModel = RegressionModel(256, is_bin=True)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes, is_bin=True)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        try:
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        except Exception as e:
            print('bias not in use in classification model')

        self.regressionModel.output.weight.data.fill_(0)

        try:
            self.regressionModel.output.bias.data.fill_(0)
        except Exception as e:
            print('bias not in use in regression model')

        #self.freeze_bn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print('------>>>block downsample stride = ', stride)
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                #conv1x1(self.inplanes, planes * block.expansion),
                conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=1, bias=False, is_bin=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        '''
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            classification_loss, regression_loss, all_positive_indices = self.focalLoss(classification, regression, anchors, annotations)
            return classification_loss, regression_loss, classification, regression, all_positive_indices,  features
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def birealnet18(checkpoint_path=None, **kwargs):
    """Constructs a BiRealNet-18 model. """
    print('Loading BiRealNet18')
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    if checkpoint_path:
        print('Loading pretrained model at {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model

