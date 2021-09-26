export CUDA_VISIBLE_DEVICES='6,7'

OUTPUT_FOLDER='./results'
CDC=8
RDC=8
FDC=0
CLC=1
RLC=1
#LR=0.00000833
LR=0.0001
#EXP_NAME="dummy_resnet18_backbone_binary_distillation_head_plus_featuresWithCoeff${FDC}"
#EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer1_cdc${CDC}_rdc${RDC}_fdc${FDC}"
#EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer123_cdc${CDC}_rdc${RDC}_fdc${FDC}"
#EXP_NAME="BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR_distillation_head_LambdaLR_lr${LR}"
#EXP_NAME="BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_OldScheduler_binary_FPN_distillation_head_LambdaLR_lr${LR}"
#EXP_NAME="normalized_logit_map_teacher_changed_BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR_binary_FPN_distillation_head_LambdaLR_lr${LR}_CDC{$CDC}_RDC{$RDC}_CLC{$CLC}_RLC{$RLC}"
EXP_NAME="normalized_logit_map_teacher_changed_every_epoch_BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_OldScheduler_binary_FPN_distillation_head_LambdaLR_lr${LR}_CDC{$CDC}_RDC{$RDC}_CLC{$CLC}_RLC{$RLC}"
CAPTION='BiReal18_distillation'
python train.py --dataset coco --lrScheduler LambdaLR --caption ${CAPTION} --coco_path /media/ramdisk/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr ${LR} --batch_size 8 --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC} --clc ${CLC} --rlc ${RLC}

