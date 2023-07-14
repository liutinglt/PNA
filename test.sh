 
EXP='pna-deeplabv3_r50-d16_512x512_40k_cocostuff10k'
CONFIG_FILE='configs/pna/'${EXP}'.py'
CHECKPOINT_FILE='work_dirs/cocostuff10k_pna_deeplabv3_r50.pth' 
 

PORT=29501 CUDA_VISIBLE_DEVICES=0,1,2,3  bash tools/dist_test.sh \
 ${CONFIG_FILE} ${CHECKPOINT_FILE} 4 \
 --eval 'mIoU'



