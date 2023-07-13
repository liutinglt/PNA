
EXP="pna_fpn_R_52_poly_8k_bs16"
DATASET="trans10k"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
--config-file configs/${DATASET}/${EXP}.yaml \
--num-gpus 4 \
OUTPUT_DIR output/${DATASET}/${EXP}/\
