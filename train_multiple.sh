#export CUDA_VISIBLE_DEVICES='0,1'

OUTPUT_FOLDER='./results2'
#EXP_NAME='BiRealNet18_backbone_binary_from_scratch'
#DEPTH=101
ARCH='Resnet'
#ARCH='BiRealNet18'
gpu=0

depths=(101 50 18)
for DEPTH in ${depths[@]}; do
        EXP_NAME="Resnet${DEPTH}_backbone_full_precision_pretrain_True_freezebatchnorm_False"
	gpu2=$(($gpu+1))
	export CUDA_VISIBLE_DEVICES="${gpu},${gpu2}"
        echo "starting ${DEPTH} on gpu $gpu, $gpu2"
	screen -dms "${EXP_NAME}" bash -c "workon retinanet; python train.py --dataset coco --arch ${ARCH} --coco_path /media/School/Datasets/coco --depth $DEPTH --output_folder ${OUTPUT_FOLDER} --exp_name ${EXP_NAME} --lr 0.0001 --batch_size 8 --epochs 12 --pretrain" #--freeze_batchnorm"
	gpu=$(($gpu+2))

done
