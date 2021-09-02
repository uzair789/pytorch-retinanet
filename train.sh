export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results'
#EXP_NAME='BiRealNet18_backbone_plus_heads_binary_Imagenet_pretrain'
#EXP_NAME='BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch'
EXP_NAME='BiRealNet18_backbone_plus_heads_shortcuts_binary_from_scratch_LambdaLR'

python train.py --dataset coco --arch 'BiRealNet18' --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 #--pretrain

