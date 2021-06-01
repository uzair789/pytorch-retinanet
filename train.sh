export CUDA_VISIBLE_DEVICES='4,5,6,7'

OUTPUT_FOLDER='./results'
FDC=0.01
EXP_NAME="resnet18_backbone_binary_distillation_head_plus_featuresWithCoeff${FDC}"

python train.py --dataset coco --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --fdc ${FDC}

