export CUDA_VISIBLE_DEVICES='4,5'
 
OUTPUT_FOLDER='./results2'
CDC=0 #8
RDC=0 #8
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
#EXP_NAME="from_scratch_normalized_logit_map_teacher_changed_every_epoch_consistent_BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_OldScheduler_binary_FPN_distillation_head_LambdaLR_lr${LR}_CDC{$CDC}_RDC{$RDC}_CLC{$CLC}_RLC{$RLC}"
#EXP_NAME="Ablation_BiRealNet18_FFF_teacher_Dis-644_student_Dis-683_RYWYCNNY"
#EXP_NAME="FP_distillation_teacher_Dis-716_resnet34_student_Dis-644_resnet18_oldScheduler_CDC${CDC}_RDC${RDC}"
EXP_NAME="FP_distillation_teacher_Dis-643_resnet50_student_Dis-644_resnet18_LambdaLR_CDC${CDC}_RDC${RDC}"
#CAPTION='BiReal18_distillation'
CAPTION='FP_distillation'
python train.py --dataset coco --arch FP --lrScheduler LambdaLR --caption ${CAPTION} --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr ${LR} --batch_size 8 --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC} --clc ${CLC} --rlc ${RLC} --warmup --normalization --change_teacher #--freeze_batchnorm

