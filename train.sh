EXP_NAME='resnet50_binary'

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 50 --exp_name ${EXP_NAME} | tee ${EXP_NAME}/log.txt

