# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ASPP, Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .loss import DeepLabCE

"""
It's same with DeeplabV3
Just add a function 'layers()' to obtain the initial predcition
"""
@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV3_PNA_Head(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        in_channels           = [input_shape[f].channels for f in self.in_features]
        aspp_channels         = cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        aspp_dilations        = cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # output stride
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.loss_type        = cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE
        train_crop_size       = cfg.INPUT.CROP.SIZE
        aspp_dropout          = cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT
        use_depthwise_separable_conv = cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
        # fmt: on

        assert len(self.in_features) == 1
        assert len(in_channels) == 1

        # ASPP module
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_crop_h, train_crop_w = train_crop_size
            if train_crop_h % self.common_stride or train_crop_w % self.common_stride:
                raise ValueError("Crop size need to be divisible by output stride.")
            pool_h = train_crop_h // self.common_stride
            pool_w = train_crop_w // self.common_stride
            pool_kernel_size = (pool_h, pool_w)
        else:
            pool_kernel_size = None
        self.aspp = ASPP(
            in_channels[0],
            aspp_channels,
            aspp_dilations,
            norm=norm,
            activation=F.relu,
            pool_kernel_size=pool_kernel_size,
            dropout=aspp_dropout,
            use_depthwise_separable_conv=use_depthwise_separable_conv,
        )

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)
    
    def layers(self, features):
        x = features[self.in_features[0]]
        x = self.aspp(x)
        x = self.predictor(x)
        return x
  
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """ 
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses