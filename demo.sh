
#  --eval-only MODEL.WEIGHTS output_msd/${EXP}/model_final.pth \
EXP=pna_fpn_R_52_poly_8k_bs16
DATASET="trans10k"
SAVE_DIR=output/trans10k/inference
MODEL_NAME=model_trans10k
mkdir $SAVE_DIR
CUDA_VISIBLE_DEVICES=2 python demo.py --config-file configs/${DATASET}/${EXP}.yaml \
  --input datasets/trans10k/test/images/*.jpg \
  --output ${SAVE_DIR} \
  --opts MODEL.WEIGHTS output/${MODEL_NAME}.pth \
  
