EXP='pna_deeplabv3_r101-d16_512x512_20k_voc'
CONFIG_FILE='configs/pna/'${EXP}'.py'
WORK_DIR='work_dirs/voc/'${EXP}

PORT=29503 CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
                            ${CONFIG_FILE} \
                            2  \
                            --seed 0 \
                            --work-dir ${WORK_DIR} 
 