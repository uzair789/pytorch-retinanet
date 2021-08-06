export CUDA_VISIBLE_DEVICES='5'

OUTPUT_FOLDER='./results'
#EXP_NAME='BiRealNet18_backbone_binary_from_scratch'
DEPTH=18
EXP_NAME="without_if_condition_for_batchnormfreeze_Resnet${DEPTH}_backbone_full_precision_pretrain_True_freezebatchnorm_True"
ARCH='Resnet'
#ARCH='BiRealNet18'

python train.py --dataset coco --arch ${ARCH} --coco_path /media/School/Datasets/coco --depth $DEPTH --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --pretrain --freeze_batchnorm

