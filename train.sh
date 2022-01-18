export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results2'
#EXP_NAME='BiRealNet18_backbone_binary_from_scratch'
DEPTH=34
EXP_NAME="Resnet${DEPTH}_backbone_full_precision_pretrain_True_freezebatchnorm_False"
ARCH='Resnet'
#ARCH='BiRealNet18'

python train.py --dataset coco --arch ${ARCH} --coco_path /media/School/Datasets/coco --depth $DEPTH --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --pretrain #--freeze_batchnorm

