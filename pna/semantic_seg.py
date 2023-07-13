# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
    Backbone, 
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .loss import DeepLabCE
from .pna import PNAHead

@META_ARCH_REGISTRY.register()
class PNASemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module, 
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        size_divisibility:int=-1,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.size_divisibility = size_divisibility 
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "size_divisibility": cfg.MODEL.PNA_HEAD.SIZE_DIVISIBILITY,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        
        images = ImageList.from_tensors(
            images,
            size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        
        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets,
                size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses
        
        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1]) 
            r = sem_seg_postprocess(result, image_size, height, width) 
            processed_results.append({"sem_seg": r})
        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class PNASemSegHead(nn.Module): 

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.common_stride = cfg.MODEL.PNA_HEAD.COMMON_STRIDE

        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
            cfg.MODEL.SEM_SEG_HEAD.COARSE_SEM_SEG_HEAD_NAME
        )(cfg, input_shape)  

        self.pna_head = PNAHead(cfg, input_shape)
         
    def forward(self, features, targets=None): 
        
        coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features)

        pna_sem_seg_logits = self.pna_head.layers(coarse_sem_seg_logits, features)
        if self.training:
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets)
            losses_pna = self.pna_head.losses(pna_sem_seg_logits, targets)
            losses.update(losses_pna) 
            return None, losses
        else:
            sem_seg_logits = pna_sem_seg_logits[-1]
            sem_seg_logits = F.interpolate(sem_seg_logits, scale_factor=self.common_stride, mode='bilinear', align_corners=False) 
            return sem_seg_logits, {}
