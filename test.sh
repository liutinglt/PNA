
EXP="pna_fpn_R_52_poly_8k_bs16"
DATASET="trans10k"
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_net.py \
--num-gpu 4 \
--config-file configs/${DATASET}/${EXP}.yaml \
 --eval-only MODEL.WEIGHTS /home/ubuntu/code/detectron_new/projects/PNA/aaai_out/model_final.pth 