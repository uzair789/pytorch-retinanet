export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results'
EXP_NAME='resnet18_backbone_binary_retrain_binarized_downsample'

python train.py --dataset coco --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --caption 'rework' --server 'sierra'

