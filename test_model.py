import torch
from retinanet import model
from icecream import ic

retinanet = model.resnet50(num_classes=81, pretrained=False).cuda()
retinanet.eval()
x = torch.rand([10, 3, 128, 128]).cuda()

retinanet(x)
