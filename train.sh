export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results'
CDC=1
RDC=1
FDC=0
EXP_NAME="resnet18_backbone_binary_distillation_head_norm_per_batchTensor_mean_on_batch"
#EXP_NAME="resnet18_backbone_binary_distillation_head_on_positive_indices_normloss_2cards_cdc${CDC}_rdc${RDC}_fdc${FDC}"

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC}

