# Copyright (c) Facebook, Inc. and its affiliates.
from audioop import bias
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, Conv2d, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .loss import DeepLabCE 
from natten import NeighborhoodCrossAttention
from .restomer import TransformerBlock


def BCHW_BHWC(x): 
    return x.permute(0, 2, 3, 1).contiguous()
    # return rearrange(x, 'b c h w -> b h w c').contiguous()

def BHWC_BCHW(x):
    return x.permute(0, 3, 1, 2).contiguous()
    # return rearrange(x, 'b h w c -> b c h w').contiguous()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
        weight_init.c2_msra_fill(self.fc1 )
        weight_init.c2_msra_fill(self.fc2 )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PNALayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, use_sa=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_sa = use_sa

        if self.use_sa:
            self.atten1 = TransformerBlock(dim, num_heads)
        else:
            self.atten1 = nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.attn = NeighborhoodCrossAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
         
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.out_norm = norm_layer(dim)

    def forward(self, x, v):
        shortcut = BCHW_BHWC(self.atten1(x))
        x = BCHW_BHWC(x) 
        v = BCHW_BHWC(v)
        x = self.norm1(x)
        v = self.norm1_2(v)
        x = self.attn(x, v)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x)) 
        x = self.out_norm(x)
        return BHWC_BCHW(x)  

class PNABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, depth=1, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, use_sa=True):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            PNALayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     norm_layer=norm_layer, use_sa=use_sa)
            for i in range(depth)])
 
    def forward(self, q, v):
        x = q
        for blk in self.blocks:
            x = blk(q, v) 
            v = x 
        
        return x 
  
@SEM_SEG_HEADS_REGISTRY.register()
class PNAHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """  
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
         
        in_features      = cfg.MODEL.PNA_HEAD.IN_FEATURES
        common_stride    = cfg.MODEL.PNA_HEAD.COMMON_STRIDE
        loss_weight      = cfg.MODEL.PNA_HEAD.LOSS_WEIGHT
        loss_type        = cfg.MODEL.PNA_HEAD.LOSS_TYPE
        embed_dims       = cfg.MODEL.PNA_HEAD.EMBED_DIMS
        window_size      = cfg.MODEL.PNA_HEAD.WINDOW_SIZE
        mlp_ratio        = cfg.MODEL.PNA_HEAD.MLP_RATIO
        upsample_mode    = cfg.MODEL.PNA_HEAD.UPSAMPLE_MODE  
        use_sa           = cfg.MODEL.PNA_HEAD.USE_SA  

        ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        # fmt: off
        self.in_features      = in_features  # starting from "res2" to "res5"
        in_channels           = [input_shape[f].channels for f in in_features]
        in_strides            = [input_shape[f].stride for f in in_features] 
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.num_classes      = num_classes  
        self.upsample_mode    = upsample_mode
        # fmt: on 
        assert (
            len(embed_dims) == len(self.in_features) 
        ), "Length of embed_dims is not equal to in_features {} vs. {}".format(
            len(embed_dims), len(self.in_features)
        )
        assert (
            len(window_size) == len(self.in_features) 
        ), "Length of window_size is not equal to in_features {} vs. {}".format(
            len(window_size), len(self.in_features)
        ) 
        
        self.embed_dims = embed_dims
        self.patch_embeds = nn.ModuleDict() 
        self.value_embeds = nn.ModuleDict() 
        self.stages = nn.ModuleDict()  
        self.conv_seg_layers = nn.ModuleDict() 

        for i, f in enumerate(self.in_features):   
            query_embed = nn.Conv2d(in_channels[i], embed_dims[i], kernel_size=1, bias=False)
            weight_init.c2_xavier_fill(query_embed)
            self.patch_embeds[f] = query_embed

            value_embed = nn.Conv2d(num_classes, embed_dims[i], kernel_size=1, bias=False) 
            weight_init.c2_xavier_fill(value_embed)
            self.value_embeds[f] = value_embed
            
            stage = PNABlock(
                dim=embed_dims[i], 
                num_heads=embed_dims[i]//32, 
                kernel_size=window_size[i],
                mlp_ratio = mlp_ratio,
                use_sa=use_sa)
            
            self.stages[f] = stage
     
            conv_seg_layer = nn.Conv2d(embed_dims[i], num_classes, kernel_size=1)
            nn.init.normal_(conv_seg_layer.weight, 0, 0.001)
            nn.init.constant_(conv_seg_layer.bias, 0)
            self.conv_seg_layers[f] = conv_seg_layer
          
        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)
 
    def forward(self, coarse_sem_seg_logits, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(coarse_sem_seg_logits, features) 

        if self.training:
            return None, self.losses(y, targets)
        else:
            y = y[-1]
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, coarse_sem_seg_logits, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        value = coarse_sem_seg_logits
        outs = []
        for f in self.in_features[::-1]:
            x = features[f]     
            if self.upsample_mode == 'bilinear':
                value = F.interpolate(value, size=x.shape[2:], mode='bilinear', align_corners=False) 
            else:
                value = F.interpolate(value, size=x.shape[2:], mode='nearest')
             
            query = self.patch_embeds[f] (x)  
            value = self.value_embeds[f](value)  

            out = self.stages[f](query, value)  
            pred = self.conv_seg_layers[f](out)
            outs.append(pred)

            value = pred   
 
        return outs

    def losses(self, predictions, targets):
        losses = {}
        for i, f in enumerate(self.in_features[::-1]):
            prediction = predictions[i]
            prediction = F.interpolate(
                prediction, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = self.loss(prediction, targets)
            losses["loss_sem_seg_pna_{}".format(f)] = loss * self.loss_weight
        return losses
 
