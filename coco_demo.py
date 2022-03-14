import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
#from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def evaluate_coco(dataset, model, output_folder_pathi=None, threshold=0.05, exp=None):

    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):

            data = dataset[index]
            scale = data['scale']
            path = dataset.get_image_path(index)
            print(path)


            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :].numpy()

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    # image_result = {
                    #    'image_id'    : dataset.image_ids[index],
                    #    'category_id' : dataset.label_to_coco_label(label),
                    #    'score'       : float(score),
                    #    'bbox'        : box.tolist(),
                    #}

                    # append detection to results
                    #results.append(image_result)
                    print(box, ' | label = ', label, ' | score = ', score, ' | ',path)
            print('===========================')
            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')


        if not len(results):
            print('in return if not reuslts')
            return

        '''
        # write output
        json.dump(results, open('{}/{}_bbox_results.json'.format(output_folder_path, dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}/{}_bbox_results.json'.format(output_folder_path, dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        summary = coco_eval.stats
        print(summary, '<<--')
        print('below summary')

        exp.log_metric('Validation: ap1', float(summary[0]))
        exp.log_metric('Validation: IOU_0.5', float(summary[1]))
        exp.log_metric('Validation: IOU_0.75', float(summary[2]))

        model.train()
        '''
        return

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory', default='/media/School/Datasets/coco')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    teacher_path = 'results2/Resnet18_backbone_full_precision_pretrain_True_freezebatchnorm_False'
    retinanet = torch.load('{}/coco_retinanet_11.pt'.format(teacher_path))
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
