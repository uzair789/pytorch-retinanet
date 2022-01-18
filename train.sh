export CUDA_VISIBLE_DEVICES='2,3'

OUTPUT_FOLDER='./results2'
#EXP_NAME='BiRealNet18_backbone_plus_heads_binary_Imagenet_pretrain'
#EXP_NAME='BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch'
#EXP_NAME='rerun_BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR_binary_FPN(DIS-375)_batchnorm_freeze_False'
EXP_NAME='BiRealNet34_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR_binary_FPN_batchnorm_freeze_False_load_same_binary_units'

python train.py --server ultron --dataset coco --arch 'BiRealNet34' --coco_path /media/School/Datasets/coco --depth 34 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 #--freeze_batchnorm #--pretrain

