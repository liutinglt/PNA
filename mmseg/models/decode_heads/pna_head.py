# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_head/point_head.py  # noqa

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
  
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from ..losses import accuracy
from .cascade_decode_head import BaseCascadeDecodeHead

from natten import NeighborhoodCrossAttention
from ..utils.restomer import TransformerBlock


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
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.atten1 = TransformerBlock(dim, num_heads)
        
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
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            PNALayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     norm_layer=norm_layer)
            for i in range(depth)])
 
    def forward(self, q, v):
        x = q
        for blk in self.blocks:
            x = blk(q, v) 
            v = x 
        
        return x 

@HEADS.register_module()
class PNAHead(BaseCascadeDecodeHead):
      
    def __init__(self,
                 embed_dims=[96, 192, 384, 768],
                 window_size=[7, 7, 7, 7],
                 mlp_ratio=1,
                 out_index=[0,1,2,3],
                 upsample_mode='nearest',
                 channels=512,  # not used 
                 **kwargs):
        super(PNAHead, self).__init__(input_transform='multiple_select', channels=channels,
                                       **kwargs) 
         
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.upsample_mode = upsample_mode
        self.out_index = out_index

        self.patch_embeds = nn.ModuleDict() 
        self.value_embeds = nn.ModuleDict() 
        self.stages = nn.ModuleDict()  
        self.conv_seg_layers = nn.ModuleDict() 
         
        for i in range(len(self.in_index))[::-1]:  
            f = "pna{}".format(i)
            query_embed = nn.Conv2d(self.in_channels[i], embed_dims[i], kernel_size=1)
            self.patch_embeds[f] = query_embed

            value_embed = nn.Conv2d(self.num_classes, embed_dims[i], kernel_size=1) 
            self.value_embeds[f] = value_embed
            
            stage = PNABlock(
                dim=embed_dims[i], 
                num_heads=embed_dims[i]//32, 
                kernel_size=window_size[i],
                mlp_ratio = mlp_ratio)
            
            self.stages[f] = stage
     
            conv_seg_layer = nn.Conv2d(embed_dims[i], self.num_classes, kernel_size=1)
            self.conv_seg_layers[f] = conv_seg_layer 

        delattr(self, 'conv_seg')

    def init_weights(self):   
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs, prev_output):  
        inputs = self._transform_inputs(inputs)  
        value = prev_output
        outs = []
        for i in range(len(self.in_index))[::-1]: 
            x = inputs[i]   
            f = "pna{}".format(i)
            if self.upsample_mode == 'bilinear':
                value = resize(value, size=x.shape[2:], mode='bilinear', align_corners=False) 
            else:
                value = resize(value, size=x.shape[2:], mode='nearest')
             
            query = self.patch_embeds[f] (x)  
            value = self.value_embeds[f](value)  

            out = self.stages[f](query, value)  
            pred = self.conv_seg_layers[f](out)
            outs.append(pred)

            value = pred   

        outputs = [outs[i] for i in self.out_index]

        return outs  
        
    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        outs = self.forward(inputs, prev_output)
        return outs[-1]

    def losses(self, seg_logits, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        size = seg_label.shape[2:] 
        seg_label_4c = seg_label
        seg_label = seg_label.squeeze(1) 
        if not isinstance(seg_logits, list) and not isinstance(seg_logits, tuple):
            seg_logits = [seg_logits]
        for i, seg_logit in enumerate(seg_logits):
            seg_logit = resize(
                input=seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners)
            if self.sampler is not None: 
                seg_weight = self.sampler.sample(seg_logit, seg_label_4c)
            else:
                seg_weight = None

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
                
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

            loss['acc_seg_'+str(i)] = accuracy(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss 

 