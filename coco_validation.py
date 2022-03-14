import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory', default='/media/School/Datasets/coco')
    parser.add_argument('--model_path', help='Path to model', type=str, default='')

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    # Dis-712 sota resnet18
    parser.model_path = 'results2/new_teacher-Dis-644_teacher.evaltest_student-Dis-683_batchnorm_freeze_False_manual_same_forward_distillation_head_LambdaLR_lr0.0001_CDC{8}_RDC{8}_CLC{1}_RLC{1}'
    retinanet = torch.load('{}/coco_retinanet_11.pt'.format(parser.model_path))

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet, './eval_check_thresh0.0',threshold=0.0)


if __name__ == '__main__':
    main()
