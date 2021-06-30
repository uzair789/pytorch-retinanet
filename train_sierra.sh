

OUTPUT_FOLDER='./results'
CDC=1
RDC=1
FDC=(1 0.1 0.01)

batch_size=8
i=2
# for batch_size in ${BATCH_SIZES[@]}; do
for FDC in ${FDC[@]}; do

    EXP_NAME="resnet18_backbone_binary_distillation_head_layer_features_cdc${CDC}_rdc${RDC}_fdc${FDC}"
    #EXP_NAME="resnet18_backbone_binary_distillation_head_on_positive_indices_normloss_2cards_cdc${CDC}_rdc${RDC}_fdc${FDC}"
    echo "FDC=${FDC}_gpus=${i},$(($i+1))"
    export CUDA_VISIBLE_DEVICES="${i},$(($i+1))"
    screen -dmS "fdc_${FDC}_gpus_${i}_$(($i+1))" bash -c "workon retinanet; python train.py --dataset coco --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size ${batch_size} --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC} --caption layer_features_x1x2x3x4 --server sierra"
    i=$(($i+2))
done

