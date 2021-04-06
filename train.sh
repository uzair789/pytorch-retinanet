EXP_NAME='resnet18_binary'

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 18 --exp_name ${EXP_NAME} | tee ${EXP_NAME}/log.txt

