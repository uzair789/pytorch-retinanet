export CUDA_VISIBLE_DEVICES='6,7'

OUTPUT_FOLDER='./results'
CDC=1
RDC=1
FDC=0
#EXP_NAME="dummy_resnet18_backbone_binary_distillation_head_plus_featuresWithCoeff${FDC}"
#EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer1_cdc${CDC}_rdc${RDC}_fdc${FDC}"
EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer123_cdc${CDC}_rdc${RDC}_fdc${FDC}"

python train.py --dataset coco --caption progressive_distillation --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC}

