import os
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
#from retinanet.birealnet import

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

import neptune
from icecream import ic

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

neptune.init('uzair789/Distillation')


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--exp_name', help='Path to folder for saving the model and log', type=str)
    parser.add_argument('--output_folder', help='Path to folder for saving all the experiments', type=str)

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100) # 100
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--caption', help='Any thing in particular about the experiment', type=str)
    parser.add_argument('--server', help='seerver name', type=str, default='ultron')
    parser.add_argument('--detector', help='detection algo', type=str, default='RetinaNet')
    parser.add_argument('--lrScheduler', help='LR Scheduler', type=str, default='Old')
    parser.add_argument('--cdc', help='Classification Distillation Coeff', type=float, default=1)
    parser.add_argument('--rdc', help='Regression Distillation Coeff', type=float, default=1)
    parser.add_argument('--fdc', help='Feature Distillation Coeff', type=float, default=1)

    parser = parser.parse_args(args)

    output_folder_path = os.path.join(parser.output_folder, parser.exp_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    PARAMS = {'dataset': parser.dataset,
              'exp_name': parser.exp_name,
              'depth': parser.depth,
              'epochs': parser.epochs,
              'batch_size': parser.batch_size,
              'lr': parser.lr,
              'caption': parser.caption,
              'server': parser.server,
              'LRScheduler': parser.lrScheduler,
              'classification_distill_coeff': parser.cdc,
              'regression_distill_coeff': parser.rdc,
              'feature_distill_coeff': parser.fdc


    }

    exp = neptune.create_experiment(name=parser.exp_name, params=PARAMS, tags=['resnet'+str(parser.depth),
                                                                                parser.caption,
                                                                                parser.detector,
                                                                                parser.dataset,
                                                                                parser.server])

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    distillation = True
    # Create the model
    if parser.depth == 18:
        #model_folder = 'BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch'
        #model_folder = 'BiRealNet18_backbone_plus_SE_attention_3_heads_with_shortcuts_LambdaLR'
        #model_folder = 'BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR'
        model_folder = 'BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_OldScheduler_binary_FPN'
        # retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, is_bin=True)
        #retinanet = torch.load('results/resnet18_layer123_binary_backbone_binary/coco_retinanet_11.pt')
        retinanet = torch.load('results/{}/coco_retinanet_11.pt'.format(model_folder))
        #retinanet.load_state_dict(checkpoint)
        print('student loaded!')
        print(retinanet)

        if distillation:
            #retinanet_teacher = model.resnet18(num_classes=dataset_train.num_classes(),
            #                                   pretrained=True,
            #                                   is_bin=False)
            #retinanet_teacher = torch.load('results/resnet18_layer1_binary_backbone_binary/coco_retinanet_11.pt')
            #retinanet_teacher = torch.load('results/resnet18_layer123_binary_backbone_distillation_head_teacher_layer12_cdc1_rdc1_fdc0/coco_retinanet_11.pt')
            retinanet_teacher = torch.load('results/resnet18_backbone_full_precision/coco_retinanet_11.pt')
            # retinanet_teacher.load_state_dict(checkpoint_teacher)
            print('teacher loaded!')
            print(retinanet_teacher)

    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            retinanet_teacher = retinanet_teacher.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        retinanet_teacher = torch.nn.DataParallel(retinanet_teacher).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    retinanet_teacher.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    if parser.lrScheduler == 'LambdaLR':
        print('LambdaLR')
        #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/parser.epochs), last_epoch=-1)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/12), last_epoch=-1)
    else:
        print('old scheduler')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    retinanet_teacher.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        exp.log_metric('Current lr', float(optimizer.param_groups[0]['lr']))
        exp.log_metric('Current epoch', int(epoch_num))

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):

            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss, class_output, reg_output, positive_indices, features = retinanet([data['img'].cuda().float(), data['annot']])
                    with torch.no_grad():
                        # deactivating grads on teacher to save memory
                        _, _, class_output_teacher, reg_output_teacher, positive_indices_teacher, features_teacher = retinanet_teacher([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                ## -->DISTILLATION ON FULL TENSOR
                # distillatioon losses with the loss coefficients
                class_loss_distill = parser.cdc * torch.norm((class_output_teacher - class_output))
                reg_loss_distill = parser.rdc * torch.norm((reg_output_teacher - reg_output))
                #import pdb
                #pdb.set_trace()
                #ic(class_output.shape)
                #ic(class_output_teacher.shape)
                #ic(reg_output.shape)
                #ic(reg_output_teacher.shape)
                #ic(len(positive_indices_teacher))
                ## <--


                ##-->> CHECKING FOR DISTILLATION ON THE POSITIVES ONLY
                """
                assert(len(class_output)==len(class_output_teacher))
                assert(len(reg_output)==len(reg_output_teacher))
                c = []
                r = []
                for i in range(parser.batch_size):
                    #print('{}/{}'.format(i, parser.batch_size), '----')

                    #ic(positive_indices[i].shape)
                    #ic(class_output_teacher.device)
                    #ic(positive_indices_teacher[i].shape)
                    #ic(positive_indices_teacher[i].device)
                    #continue
                    #ic(class_output_teacher[i, positive_indices_teacher[i], :].shape)
                    #ic(class_output[i, positive_indices_teacher[i], :].shape)
                    #ic(reg_output_teacher[i, positive_indices_teacher[i], :].shape)
                    #ic(reg_output[i, positive_indices_teacher[i], :].shape)


                    #c_loss = torch.norm(class_output_teacher[i, positive_indices_teacher[i], :] -
                    #           class_output[i, positive_indices_teacher[i], :]).cuda()
                    #r_loss = torch.norm(reg_output_teacher[i, positive_indices_teacher[i], :] -
                    #   reg_output[i, positive_indices_teacher[i], :]).cuda()
                    c_loss = (class_output_teacher[i, positive_indices_teacher[i], :] -
                              class_output[i, positive_indices_teacher[i], :]).cuda()
                    r_loss = (reg_output_teacher[i, positive_indices_teacher[i], :] -
                              reg_output[i, positive_indices_teacher[i], :]).cuda()
                    c.append(c_loss)
                    r.append(r_loss)
                #class_loss_distill = torch.tensor(c).mean()
                #reg_loss_distill = torch.tensor(r).mean()
                c = torch.vstack(c)
                r = torch.vstack(r)
                #ic(c.shape)
                #ic(r.shape)

                # also experimenting with sum in comparison to the mean
                class_loss_distill = parser.cdc * torch.norm(c)
                reg_loss_distill = parser.rdc * torch.norm(r)
                #ic(class_loss_distill)
                #ic(reg_loss_distill)
                """
                ## <<--

                # >>>> DIDNT WORK FOR POSITIVES ONLY DISTILLTION
                #class_loss_distill = parser.cdc * sum([torch.norm(class_output_teacher[i, positive_indices_teacher[i], :] - class_output[i, positive_indices_teacher[i], :])
                #                                       for i in range(parser.batch_size)])#/parser.batch_size
                #reg_loss_distill = parser.rdc *  sum([torch.norm(reg_output_teacher[i, positive_indices_teacher[i], :] - reg_output[i, positive_indices_teacher[i], :])
                #                                      for i in range(parser.batch_sie)])#/parser.batch_size
                ## <<<<



                features_loss_distill = parser.fdc * sum([torch.norm(features_teacher[i] - features[i]) for i in range(len(features)) ])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss + class_loss_distill + reg_loss_distill + features_loss_distill
                # loss = class_loss_distill + reg_loss_distill

                if bool(loss == 0):
                    print('loss=0 hence continue')
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | Class distill loss: {:1.5f} | Reg distill loss: {:1.5f} | Feat distill loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist), float(class_loss_distill), float(reg_loss_distill), float(features_loss_distill)))

                exp.log_metric('Training: Distill Classification loss', float(class_loss_distill))
                exp.log_metric('Training: Distill Regression loss', float(reg_loss_distill))
                exp.log_metric('Training: Distill Features loss', float(features_loss_distill))
                exp.log_metric('Training: Classification loss', float(classification_loss))
                exp.log_metric('Training: Regression loss', float(regression_loss))
                exp.log_metric('Training: Totalloss', float(loss))

                del classification_loss
                del regression_loss
                del class_loss_distill
                del reg_loss_distill
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet, exp=exp)

            #exp.log_metric('Validation: ap1', float(ap1))
            #exp.log_metric('Validation: IOU_0.5', float(iou_point_five))
            #exp.log_metric('Validation: IOU_0.75', float(iou_point_sevenfive))

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        if parser.lrScheduler == 'LambdaLR' and epoch_num < 12:
            print('step LambdaLR')
            scheduler.step()
        else:
            print('step oldScheduler')
            scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, os.path.join(output_folder_path, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))

    retinanet.eval()

    torch.save(retinanet, os.path.join(output_folder_path, 'model_final.pt'))


if __name__ == '__main__':
    main()
