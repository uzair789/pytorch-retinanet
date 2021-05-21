export CUDA_VISIBLE_DEVICES='0,1,2,3'

OUTPUT_FOLDER='./results'
EXP_NAME='resnet18_backbone_full_precision'

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} | tee ${EXP_NAME}/log.txt

