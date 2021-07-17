

OUTPUT_FOLDER='./results'
#FDC=(1 0.1 0.01)
LR=0.0001
batch_size=8
i=0
#NET_TYPE=('layer4_binary' 'layer43_binary' 'layer432_binary')
NET_TYPE=('layer1_binary' 'layer12_binary' 'layer123_binary')
# for batch_size in ${BATCH_SIZES[@]}; do

for net_type in ${NET_TYPE[@]}; do

    EXP_NAME="resnet18_${net_type}_backbone_binary_24epochs"
    #EXP_NAME="resnet18_backbone_binary_distillation_head_on_positive_indices_normloss_2cards_cdc${CDC}_rdc${RDC}_fdc${FDC}"
    echo "net_type=${net_type}_gpus=${i},$(($i+1))"
    export CUDA_VISIBLE_DEVICES="${i},$(($i+1))"
    screen -dmS "${net_type}_gpus_${i}_$(($i+1))" bash -c "workon retinanet; python train.py --dataset coco --coco_path /media/apple/Datasets/coco --depth 18 --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr ${LR} --batch_size ${batch_size} --epochs 24 --net_type ${net_type} --caption progressive_distillation_pretrains_24epochs --server sierra"
    i=$(($i+2))
done

