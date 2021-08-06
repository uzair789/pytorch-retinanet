export CUDA_VISIBLE_DEVICES='6,7'

OUTPUT_FOLDER='./results'
EXP_NAME='BiRealNet18_backbone_binary_from_scratch'

python train.py --dataset coco --arch 'BiRealNet18' --coco_path /media/School/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 #--pretrain

