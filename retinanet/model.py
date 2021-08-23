import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

# adding the binarization units
from retinanet.binary_units import BinaryActivation, HardBinaryConv, BinaryLinear


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_bin=False):
     if is_bin:
         return HardBinaryConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
     return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def activation(inplace=False, is_bin=False):
    if is_bin:
        return BinaryActivation()
    return nn.ReLU(inplace=inplace)

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = activation()
        self.P7_2 = conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
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


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, is_bin=False):
        super(RegressionModel, self).__init__()

        # not binarizing because treating this as the first layer dealing with input
        self.is_bin = is_bin
        self.conv1 = conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = activation()

        self.conv2 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act2 = activation(is_bin=is_bin)

        self.se2 = SELayer(feature_size)

        self.conv3 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act3 = activation(is_bin=is_bin)

        self.se3 = SELayer(feature_size)

        self.conv4 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act4 = activation(is_bin=is_bin)

        self.se4 = SELayer(feature_size)

        self.output = conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward_regular(self, x):
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

    def forward_binary(self, x):

        out1 = self.conv1(x)
        #out = self.act1(out)

        out = self.act2(out1)
        out2 = self.conv2(out) + out1
        #out = self.act2(out)

        #out2 = self.se2(out2)

        out = self.act3(out2)
        out3 = self.conv3(out) + out2
        #out = self.act3(out)

        #out3 = self.se3(out3)

        out = self.act4(out3)
        out = self.conv4(out) + out3
        #out = self.act4(out)

        #out = self.se4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


    def forward(self, x, is_bin=False):
        if is_bin:
            #print('In Binary forward, Regression')
            return self.forward_binary(x)

        #print('In Regular forward, Regression')
        return self.forward_regular(x)



class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, is_bin=False):
        super(ClassificationModel, self).__init__()

        self.is_bin = is_bin
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = activation()

        self.conv2 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act2 = activation(is_bin=is_bin)

        self.se2 = SELayer(feature_size)

        self.conv3 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act3 = activation(is_bin=is_bin)

        self.se3 = SELayer(feature_size)

        self.conv4 = conv2d(feature_size, feature_size, kernel_size=3, padding=1, is_bin=is_bin)
        self.act4 = activation(is_bin=is_bin)

        self.se4 = SELayer(feature_size)

        self.output = conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward_regular(self, x):
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

    def forward_binary(self, x):
        out1 = self.conv1(x)
        #out = self.act1(out)

        out = self.act2(out1)
        out2 = self.conv2(out) + out1
        #out = self.act2(out)

        #out2 = self.se2(out2)

        out = self.act3(out2)
        out3 = self.conv3(out) + out2
        #out = self.act3(out)

        #out3 = self.se3(out3)

        out = self.act4(out3)
        out = self.conv4(out) + out3
        #out = self.act4(out)

        #out = self.se4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

    def forward(self, x, is_bin=False):
        if is_bin:
            #print('In Binary forward, Classification')
            return self.forward_binary(x)

        #print('In Regular forward, Classification')
        return self.forward_regular(x)



class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, is_bin=[False, False, False, False]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = activation(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_bin=is_bin[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_bin=is_bin[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_bin=is_bin[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_bin=is_bin[3])

        if block == BasicBlock:
            # fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
            #              self.layer4[layers[3] - 1].conv2.out_channels]
            fpn_sizes = [self.layer2[layers[1] - 1].out_channels, self.layer3[layers[2] - 1].out_channels,
                         self.layer4[layers[3] - 1].out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError("Block type {} not understood".format(block))

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

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

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1, is_bin=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False, is_bin=False),
                nn.BatchNorm2d(planes * block.expansion),
            ) # WE DO NOT BINARIZE THE DOWNSAMPLE LAYER

        layers = [block(self.inplanes, planes, stride, downsample, is_bin=is_bin)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_bin=is_bin))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
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
            classification_loss, regression_loss, all_positive_indices  = self.focalLoss(classification, regression, anchors, annotations)
            return classification_loss, regression_loss, classification, regression, all_positive_indices, features
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


def resnet18(arch, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if arch == 'full_precision':
        is_bin=[False, False, False, False]
    elif arch=='binary_net':
        is_bin=[True, True, True, True]
    elif arch=='layer1_binary':
        is_bin=[True, False, False, False]
    elif arch=='layer12_binary':
        is_bin=[True, True, False, False]
    elif arch=='layer123_binary':
        is_bin=[True, True, True, False]
    elif arch=='layer4_binary':
        is_bin=[False, False, False, True]
    elif arch=='layer43_binary':
        is_bin=[False, False, True, True]
    elif arch=='layer432_binary':
        is_bin=[False, True, True, True]
    else:
        raise(ValueError, 'arch not defined [full_precision, binary_net, layer1_binary, layer12_binary, layer123_binary, layer4_binary, layer43_binary, layer432_biinary]')
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], is_bin, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
