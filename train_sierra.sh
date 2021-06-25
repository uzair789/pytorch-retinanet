export CUDA_VISIBLE_DEVICES='4,5'

OUTPUT_FOLDER='./results'
CDC=1
RDC=1
FDC=0

BATCH_SIZES=(10)
i=6
for batch_size in ${BATCH_SIZES[@]}; do

    EXP_NAME="resnet18_backbone_binary_distillation_head_batchSize_${batch_size}"
    #EXP_NAME="resnet18_backbone_binary_distillation_head_on_positive_indices_normloss_2cards_cdc${CDC}_rdc${RDC}_fdc${FDC}"
    echo "batch_Size=${batch_size}_gpus=${i},$(($i+1))"
    export CUDA_VISIBLE_DEVICES="${i},$(($i+1))"
    screen -dmS "batch_size_${batch_size}_gpus_${i}_$(($i+1))" bash -c "workon retinanet; python train.py --dataset coco --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size ${batch_size} --epochs 12 --fdc ${FDC} --cdc ${CDC} --rdc ${RDC} --caption batch_size_effects --server sierra"
    i=$(($i+2))
done

