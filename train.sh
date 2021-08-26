export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results'
CDC=1
RDC=1
FDC=0
CLC=0.001
RLC=0.001
LR=0.0001
#EXP_NAME="dummy_resnet18_backbone_binary_distillation_head_plus_featuresWithCoeff${FDC}"
#EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer1_cdc${CDC}_rdc${RDC}_fdc${FDC}"
#EXP_NAME="resnet18_binary_backbone_distillation_head_teacher_layer123_cdc${CDC}_rdc${RDC}_fdc${FDC}"
#EXP_NAME="BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_distillation_head"
EXP_NAME="BiRealNet18_backbone_plus_SE_attention_3_heads_with_shortcuts_LambdaLR_distillation_head_clc${CLC}_rlc${RLC}"
CAPTION='BiReal18_distillation'
python train.py --dataset coco --caption ${CAPTION} --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr ${LR} --batch_size 8 --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC} --clc ${CLC} --rlc ${RLC}

