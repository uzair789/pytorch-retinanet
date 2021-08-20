export CUDA_VISIBLE_DEVICES='4,5'

OUTPUT_FOLDER='./results'
#EXP_NAME='BiRealNet18_backbone_plus_heads_binary_Imagenet_pretrain'
EXP_NAME='BiRealNet18_backbone_plus_SE_attention_3_heads_with_shortcuts_after_se'

python train.py --server 'Sierra' --dataset coco --arch 'BiRealNet18' --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 #--pretrain

