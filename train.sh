export CUDA_VISIBLE_DEVICES='4,5'

OUTPUT_FOLDER='./results3'
#EXP_NAME='BiRealNet18_backbone_binary_from_scratch'
DEPTH=50
#ARCH='ofa'
LR=0.001
EXP_NAME="debug_LR${LR}_Resnet${DEPTH}_backbone_full_precision_pretrain_False_freezebatchnorm_False"
ARCH='Resnet'
#ARCH='BiRealNet18'

python train.py --dataset coco --arch ${ARCH} --coco_path /media/School/Datasets/coco --depth $DEPTH --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr $LR --batch_size 8 --epochs 12 #--pretrain #--freeze_batchnorm

