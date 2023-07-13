# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN

def add_pna_config(cfg):
    """
    Add config for PNA.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop. 
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.MODEL.FPN.RETURN_INPUT = False 

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1


    cfg.MODEL.SEM_SEG_HEAD.COARSE_SEM_SEG_HEAD_NAME = "DeepLabV3HEAD"
    cfg.MODEL.PNA_HEAD = CN()
    cfg.MODEL.PNA_HEAD.COMMON_STRIDE = 4 
    cfg.MODEL.PNA_HEAD.LOSS_WEIGHT = 0.1
    cfg.MODEL.PNA_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.PNA_HEAD.LOSS_TYPE = "hard_pixel_mining"
    cfg.MODEL.PNA_HEAD.NUM_CLASSES = 19
    cfg.MODEL.PNA_HEAD.EMBED_DIMS = [96, 192, 384, 768]
    cfg.MODEL.PNA_HEAD.WINDOW_SIZE = [9, 9, 9, 9]
    cfg.MODEL.PNA_HEAD.MLP_RATIO = 1
    cfg.MODEL.PNA_HEAD.UPSAMPLE_MODE = "nearest"
    cfg.MODEL.PNA_HEAD.USE_SA = True
    cfg.MODEL.PNA_HEAD.SIZE_DIVISIBILITY = -1
