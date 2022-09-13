# Modified by Sukmin Yun (sukmin.yun@kaist.ac.kr)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

import torch.distributed as dist

logger = logging.getLogger(__name__)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class HeadMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., cls_inclusive=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cls_inclusive = cls_inclusive

    
    def forward(self, x, mask=None):
        
        B, N, C = x.shape
        q = self.q(x[:, :1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        if self.cls_inclusive:
            x_kv = x
        else:
            x_kv = x[:, 1:]

        k = self.k(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            with torch.no_grad():
                mask_temp = torch.cat([torch.zeros(size=(B, 1), device=mask.device), mask], dim=1).unsqueeze(1).unsqueeze(1)
                attn += mask_temp

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls

class PatchAggregation(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Attention,
                 Mlp_block=Mlp, cls_inclusive=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, cls_inclusive=cls_inclusive)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_cls, mask):
        """
        x : L (H W) D
        x_cls : L 1 D
        mask : L (H W)
        """
        u = torch.cat((x_cls, x), dim=1)
        u_ = self.attn(self.norm1(u), mask=mask)

        x_cls = x_cls + self.drop_path(u_)
        x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))

        return x_cls

class PatchAggregationHead(nn.Module):
    def __init__(self, in_dim, num_heads, use_cls=True, cosim_p=1.0):
        super().__init__()

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

        self.cls_blocks = nn.ModuleList([
            PatchAggregation(
                dim=in_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, cls_inclusive=True)
            for i in range(2)])
        
        self.norm = partial(nn.LayerNorm, eps=1e-6)(in_dim)

        self.apply(self._init_weights)
        self.cosim_p = cosim_p
        self.embed_dim = in_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img, aggr_seed, batch_idx, patch_mask):
        """
        img: 'B (H W) D'
        aggr_seed: 'L D' e.g., text embedding
        batch_idx: 'L'
        patch_mask: 'L (H W)'
        """
        B, HW, D = img.size()
        L = batch_idx.size(0)

        ###############################################################################
        ##################### filter self.k_num masks using cosim #####################
        #  
        # with torch.no_grad():
        #     img_norm = nn.functional.normalize(img, dim=-1)
        #     img_norm = img_norm[batch_idx]

        #     aggr_norm = nn.functional.normalize(aggr_seed, dim=-1)

        #     sim_matrix = torch.einsum("ld,lpd->lp", aggr_norm, img_norm)
        #     sim_matrix += patch_mask

        #     import pdb; pdb.set_trace()

        #     _, top_idx = sim_matrix.topk(k=self.k_num, dim=-1)
            
        #     img_topk = img[word_batch_idx.unsqueeze(-1).expand(-1, self.k_num), top_idx]
        #
        ###############################################################################
        ###############################################################################
        img_repeat = img[batch_idx]
        
        if self.cls_token is not None:
            tokens = self.cls_token.expand(L, -1, -1)
        else:
            tokens = aggr_seed.unsqueeze(1)

        for i, blk in enumerate(self.cls_blocks):
            tokens = blk(img_repeat,
                         tokens,
                         mask=patch_mask)

        tokens = tokens.squeeze(1)

        return tokens

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLogit(nn.Module):
    def __init__(self, out_dim, warmup_temp,
                 warmup_temp_iters, temp=0.1,
                 center_momentum=0.9, name=None):
        super().__init__()
        self.temp = temp
        self.center_momentum = center_momentum
        self.name = name or ''
        
        self.register_buffer("center", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_temp,
                        temp, warmup_temp_iters),
        ))
        self.iter = 0
        logger.info(f"initializing dino logit {name}")
        logger.info(f"warmup temp from {warmup_temp} to {temp} for {warmup_temp_iters} iterations.")

    def forward(self, tokens, iter=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[min(self.iter, len(self.teacher_temp_schedule)-1)]
        logit = (tokens - self.center) / temp

        if self.training:
            self.update_center(tokens)

        if iter is not None:
            self.iter += 1

        return logit

    @torch.no_grad()
    def update_center(self, tokens):
        """
        Update centers
        """
        B = torch.tensor([tokens.size(0)], device=tokens.device)

        tokens_center = torch.sum(tokens, dim=0, keepdim=True)
        if is_dist_avail_and_initialized() and get_world_size() > 1:
            dist.all_reduce(tokens_center)
            dist.all_reduce(B)

        tokens_center = tokens_center / B

        # ema update
        maybe_nan = tokens_center.isnan().any()
        if maybe_nan:
            logger.info(f"Skip updating {self.name} centers due to NaN")

        else:
            self.center = self.center * self.center_momentum + tokens_center * (1 - self.center_momentum)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# class DINOHead(nn.Module):
#     def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         if nlayers == 1:
#             self.mlp = nn.Linear(in_dim, bottleneck_dim)
#         else:
#             layers = [nn.Linear(in_dim, hidden_dim)]
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.GELU())
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 if use_bn:
#                     layers.append(nn.BatchNorm1d(hidden_dim))
#                 layers.append(nn.GELU())
#             layers.append(nn.Linear(hidden_dim, bottleneck_dim))
#             self.mlp = nn.Sequential(*layers)
#         self.apply(self._init_weights)
#         self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
#         self.last_layer.weight_g.data.fill_(1)
#         if norm_last_layer:
#             self.last_layer.weight_g.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.mlp(x)
#         x = nn.functional.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#         return x

# class DINOLogit(nn.Module):
#     def __init__(self, out_dim, warmup_temp,
#                  warmup_temp_epochs, nepochs=10000, temp=0.1,
#                  center_momentum=0.9):
#         super().__init__()
#         self.temp = temp
#         self.center_momentum = center_momentum
        
#         self.register_buffer("img_global_center", torch.zeros(1, out_dim))
#         self.register_buffer("img_local_center", torch.zeros(1, out_dim))
#         self.register_buffer("word_global_center", torch.zeros(1, out_dim))
#         self.register_buffer("word_local_center", torch.zeros(1, out_dim))

#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         self.teacher_temp_schedule = np.concatenate((
#             np.linspace(warmup_temp,
#                         temp, warmup_temp_epochs),
#             np.ones(nepochs - warmup_temp_epochs) * temp
#         ))

#     def forward(self, img_glo, img_loc, word_glo, word_loc, epoch, word_batch_idx):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         bsz = word_glo.size(0)
        
#         # teacher centering and sharpening
#         temp = self.teacher_temp_schedule[epoch]

#         word_glo_logit = (word_glo - self.word_global_center) / temp
#         img_glo_logit = (img_glo - self.img_global_center) / temp

#         word_loc_logit = None
#         img_loc_logit = None
#         if img_loc is not None and word_loc is not None:
#             word_loc_logit = (word_loc - self.word_local_center) / temp
#             img_loc_logit = (img_loc - self.img_local_center) / temp

#             counts = torch.zeros(size=(bsz, ), device=word_loc_logit.device, dtype=torch.float)
#             counts.scatter_add_(dim=0, index=word_batch_idx, src=torch.ones_like(word_batch_idx, dtype=torch.float))

#             self.update_center(img_glo, img_loc, word_glo, word_loc, counts.sum())

#         return word_glo_logit, img_glo_logit, word_loc_logit, img_loc_logit

#     @torch.no_grad()
#     def update_center(self, img_glo, img_loc, word_glo, word_loc, num_words):
#         """
#         Update centers
#         """
#         bsz = word_glo.size(0)

#         img_glo_center = torch.sum(img_glo, dim=0, keepdim=True)
#         word_glo_center = torch.sum(word_glo, dim=0, keepdim=True)
        
#         img_loc_center = torch.sum(img_loc, dim=0, keepdim=True)
#         word_loc_center = torch.sum(word_loc, dim=0, keepdim=True)

#         if is_dist_avail_and_initialized() and get_world_size() > 1:
#             dist.all_reduce(img_glo_center)
#             dist.all_reduce(word_glo_center)

#             dist.all_reduce(img_loc_center)
#             dist.all_reduce(word_loc_center)

#             dist.all_reduce(num_words)

#         img_glo_center = img_glo_center / (bsz * get_world_size())
#         word_glo_center = word_glo_center / (bsz * get_world_size())

#         img_loc_center = img_loc_center / num_words
#         word_loc_center = word_loc_center / num_words

#         # ema update
#         img_glo_nan = img_glo_center.isnan().any()
#         word_glo_nan = word_glo_center.isnan().any()
#         img_loc_nan = img_loc_center.isnan().any()
#         word_loc_nan = word_loc_center.isnan().any()
#         if img_glo_nan or word_glo_nan or img_loc_nan or word_loc_nan:
#             logger.info("Skip updating centers due to NaN")
#             logger.info(f"img_glo_nan: {img_glo_nan}")
#             logger.info(f"word_glo_nan: {word_glo_nan}")
#             logger.info(f"img_loc_nan: {img_loc_nan}")
#             logger.info(f"word_loc_nan: {word_loc_nan}")

#         else:
#             self.img_global_center = self.img_global_center * self.center_momentum + img_glo_center * (1 - self.center_momentum)
#             self.word_global_center = self.word_global_center * self.center_momentum + word_glo_center * (1 - self.center_momentum)
            
#             self.img_local_center = self.img_local_center * self.center_momentum + img_loc_center * (1 - self.center_momentum)
#             self.word_local_center = self.word_local_center * self.center_momentum + word_loc_center * (1 - self.center_momentum)


# class Class_Attention(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to do CA 
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

    
#     def forward(self, x, attention=False, mask=None):
        
#         B, N, C = x.shape
#         q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         q = q * self.scale
#         v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         attn = (q @ k.transpose(-2, -1))
#         import pdb; pdb.set_trace()
#         if mask is not None:
#             mask_temp = torch.cat([torch.ones(B,1).bool().cuda(), mask],dim=1).unsqueeze(1).unsqueeze(1).expand(-1,self.num_heads,-1,-1)
#             attn = attn.masked_fill_(~mask_temp.bool(), float("-inf"))
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
#         x_cls = self.proj(x_cls)
#         x_cls = self.proj_drop(x_cls)
        
#         if attention:
#             return x_cls, attn
#         else:
#             return x_cls

# class LayerScale_Block_CA(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to add CA and LayerScale
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
#                  Mlp_block=Mlp):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention_block(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, x_cls, attention=False, mask=None):
#         u = torch.cat((x_cls,x),dim=1)
#         if attention:
#             u_, cls_attn = self.attn(self.norm1(u), attention=True)
#             return cls_attn
#         else:
#             u_ = self.attn(self.norm1(u), mask=mask)
#         x_cls = x_cls + self.drop_path(u_)
#         x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
#         return x_cls

# class Patch_Attention(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to do CA 
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
    
#     def forward(self, x):
#         B, N, C = x.shape
#         q = self.q(x[:,1:]).unsqueeze(1).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         q = q * self.scale
#         v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         attn = (q @ k.transpose(-2, -1)) 
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x_cls = (attn @ v).transpose(1, 2).reshape(B, N-1, C)
#         x_cls = self.proj(x_cls)
#         x_cls = self.proj_drop(x_cls)
        
#         return x_cls, attn

# class Patch_Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Patch_Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x_cls, x, return_attention=False):
#         u = torch.cat((x_cls,x),dim=1)
#         y, attn = self.attn(self.norm1(u))
#         if return_attention:
#             return attn
#         x = x + self.drop_path(y)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn


# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, return_attention=False):
#         y, attn = self.attn(self.norm1(x))
#         if return_attention:
#             return attn
#         x = x + self.drop_path(y)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x


# class VisionTransformer(nn.Module):
#     """ Vision Transformer """
#     def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim
#         self.num_heads = num_heads

#         self.patch_embed = PatchEmbed(
#             img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])

#         # Classifier head
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def interpolate_pos_encoding(self, x, w, h):
#         npatch = x.shape[1]
#         N = self.pos_embed.shape[1]
#         if npatch == N and w == h:
#             return self.pos_embed
#         patch_pos_embed = self.pos_embed
#         dim = x.shape[-1]
#         w0 = w // self.patch_embed.patch_size
#         h0 = h // self.patch_embed.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

#     def prepare_tokens(self, x):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)
#         # add positional encoding to each token
#         x = x + self.interpolate_pos_encoding(x, w, h)
#         return self.pos_drop(x)

#     def forward(self, x, loc=False):
#         x = self.prepare_tokens(x)
#         for blk in self.blocks:
#             x = blk(x)
#         return x

#     def get_intermediate_layers(self, x, n=1):
#         x = self.prepare_tokens(x)

#         # we return the output tokens from the `n` last blocks
#         output = []
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)
#             if len(self.blocks) - i <= n:
#                 output.append(x)
#         return output

# def vit_tiny(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_small(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_base(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model