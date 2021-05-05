export CUDA_VISIBLE_DEVICES='4,5,6,7'

OUTPUT_FOLDER='./results'
EXP_NAME='resnet18_backbone_binary'

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} | tee ${EXP_NAME}/log.txt

