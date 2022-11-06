import math
import random
from typing import Any, Dict, List, Optional, Tuple
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    SinusoidalPositionalEmbedding,
    GradMultiply
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from einops import rearrange, repeat

from .unify_transformer_layer import TransformerEncoderLayer
from .resnet import ResNet
from .frozen_bn import FrozenBatchNorm2d

import logging
logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def resolve_str_true_false(x):
    x = x.lower()
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        raise ValueError(f"Unable to recognize string bool input: {x}")

def maybe_no_grad(no_grad=True):

    if no_grad:
        return torch.no_grad()
    else:
        return contextlib.ExitStack()  # dummy contextmanager

def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(
        nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps)
    )

def make_token_bucket_position(bucket_size, max_position=DEFAULT_MAX_SOURCE_POSITIONS):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    # mid 는 bucket_size 의 절반, 즉 좌-우
    
    abs_pos = torch.where((relative_pos<mid) & (relative_pos > -mid), mid-1, torch.abs(relative_pos))
    # 기준 토큰으로 거리가 mid 까지는 mid-1 을 집어넣고 (mid 보다 작은 값이기만 하면되서 아무 값이나 넣은듯, bucket_pos 계산하는 맨 마지막 라인 참고)
    # 거리가 mid 넘어가면 실제 거리를 집어넣음 (바로 다음 라인에서 log 비율로 거리 결정하려고)
    
    log_pos = torch.ceil(torch.log(abs_pos/mid)/math.log((max_position-1)/mid) * (mid-1)) + mid
    # mid 를 넘어가는 거리로부터는 log(abs_pos/mid) / log((max_position-1)/mid) 비율로 거
    
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos*sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, seg_embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        embedding_modules=[] # used for tracking PE-related modules
  
        if getattr(args, "encoder_prompt", False):
            self.encoder_prompt_encoder = PromptEncoder(
                type=args.encoder_prompt_type,
                length=args.encoder_prompt_length,
                projection=args.encoder_prompt_projection,
                embed_dim=args.encoder_embed_dim,
                proj_dim=args.encoder_prompt_dim,
                layers=args.encoder_layers,
                vocab_size=args.vocab_size)
        self.encoder_dropout = nn.Dropout(p=0.2)
        
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.num_attention_heads = args.encoder_attention_heads

        self.embed_tokens = embed_tokens
        self.seg_embed_tokens = seg_embed_tokens
        
        self.embed_tokens_bag = nn.EmbeddingBag.from_pretrained(embed_tokens.weight, freeze=False)
        self.embed_tokens_bag.weight = embed_tokens.weight

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
            embedding_modules.append(self.layernorm_embedding)
        else:
            self.layernorm_embedding = None

        if getattr(args, "add_type_embedding", False):
            self.type_embedding = Embedding(2, embed_dim, padding_idx=None)
            embedding_modules.append(self.type_embedding)
        else:
            self.type_embedding = None

        freeze_entire_resnet = resolve_str_true_false(args.freeze_entire_resnet)
        freeze_resnet = True if freeze_entire_resnet else resolve_str_true_false(args.freeze_resnet)

        if getattr(args, "sync_bn", False):
            norm_layer = BatchNorm2d
        else:
            if freeze_resnet:
                logger.info("Freezing ResNet BN layers...")
                norm_layer = FrozenBatchNorm2d
            else:
                norm_layer = None

        if args.resnet_type == 'resnet101':
            self.embed_images = ResNet([3, 4, 23], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        elif args.resnet_type == 'resnet152':
            self.embed_images = ResNet([3, 8, 36], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        elif args.resnet_type == 'resnet50':
            self.embed_images = ResNet([3, 4, 6], norm_layer=norm_layer, drop_path_rate=args.resnet_drop_path_rate)
        else:
            raise NotImplementedError

        self.image_proj = Linear(1024, embed_dim)
        if getattr(args, "resnet_model_path", None):
            print("load resnet {}".format(args.resnet_model_path))
            resnet_state_dict = torch.load(self.args.resnet_model_path)
            self.embed_images.load_state_dict(resnet_state_dict)

        if freeze_entire_resnet:
            logger.info("Freezing all ResNet and image projection params...")
            for param in self.embed_images.parameters():
                param.requires_grad_(False)
            
            for param in self.image_proj.parameters():
                param.requires_grad_(False)

        if getattr(args, "patch_layernorm_embedding", False):
            self.patch_layernorm_embedding = LayerNorm(embed_dim)
            embedding_modules.append(self.patch_layernorm_embedding)
        else:
            self.patch_layernorm_embedding = None

        self.embed_positions = Embedding(args.max_source_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size ** 2 + 1, embed_dim)
        
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        
        self.pos_scaling = float(embed_dim / args.encoder_attention_heads * args.attn_scale_factor) ** -0.5
        self.pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_k_linear = nn.Linear(embed_dim, embed_dim)

        embedding_modules += [self.embed_positions, self.embed_image_positions, self.pos_ln, self.image_pos_ln, self.pos_q_linear, self.pos_k_linear]

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
            embedding_modules.append(self.quant_noise)
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.encoder_drop_path_rate, args.encoder_layers)]
        self.layers.extend(
            [self.build_encoder_layer(args, drop_path_rate=dpr[i]) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.encoder_layers)]
        )
        embedding_modules += [self.token_rel_pos_table_list, self.image_rel_pos_table_list]

        self.patch_image_size = args.patch_image_size
        self.orig_patch_image_size = args.orig_patch_image_size

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)

        self.entangle_position_embedding = args.entangle_position_embedding
        
        self.no_grad_image = resolve_str_true_false(args.no_grad_image)
        self.freeze_encoder_transformer = resolve_str_true_false(args.freeze_encoder_transformer)
        self.freeze_encoder_transformer_layers = args.freeze_encoder_transformer_layers

        if self.freeze_encoder_transformer:
            logger.info("Freezing all encoder transformer parameters...")
            self.freeze_encoder_transformer_layers = self.num_layers

        if self.freeze_encoder_transformer_layers > 0:
            assert self.freeze_encoder_transformer_layers <= self.num_layers and self.freeze_encoder_transformer_layers >= 0
            logger.info(f"Freezing positional/type embdding related parameters...")
            for module in embedding_modules:
                for param in module.parameters():
                    param.requires_grad_(False)
            
            logger.info(f"Freezing first {self.freeze_encoder_transformer_layers} encoder transformer layers...")
            for idx, layer in enumerate(self.layers):
                if idx == self.freeze_encoder_transformer_layers:
                    break

                for param in layer.parameters():
                    param.requires_grad_(False)
            
            if self.freeze_encoder_transformer_layers == self.num_layers and self.layer_norm is not None:
                logger.info(f"Freezing encoder output layer norm parameters...")
                for param in self.layer_norm.parameters():
                    param.requires_grad_(False)
                    
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                logger.info(f"encoder.{name} requires grad!")

    def build_encoder_layer(self, args, drop_path_rate=0.0):
        layer = TransformerEncoderLayer(args, drop_path_rate=drop_path_rate, \
            use_adapter=getattr(args, "adapter", False), adapter_dim=getattr(args, "adapter_dim", 200))
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = self.image_rp_bucket.unsqueeze(0).expand(
            bsz, rp_bucket_size, rp_bucket_size
        ).gather(1, image_position_ids[:, :, None].expand(bsz, seq_len, rp_bucket_size)
        ).gather(2, image_position_ids[:, None, :].expand(bsz, seq_len, seq_len))
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values

    def get_patch_images_info(self, patch_images, sample_patch_num, device):
        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]
        image_embed_shape = (h, w)
        image_num_patches = h * w
        image_padding_mask = patch_images.new_zeros((patch_images.size(0), image_num_patches)).bool()
        image_position_idx = torch.arange(w).unsqueeze(0).expand(h, w) + \
                             torch.arange(h).unsqueeze(1) * self.args.image_bucket_size + 1
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = [
                random.sample(range(image_num_patches), k=sample_patch_num)
                for _ in range(patch_images.size(0))
            ]
            patch_orders = torch.LongTensor(patch_orders).to(device)
            image_embed = image_embed.gather(
                1, patch_orders.unsqueeze(2).expand(-1, -1, image_embed.size(2))
            )
            image_num_patches = sample_patch_num
            image_padding_mask = image_padding_mask.gather(1, patch_orders)
            image_position_ids = image_position_ids.gather(1, patch_orders)
        orig_num_patches = (self.orig_patch_image_size // 16) ** 2
        orig_hw= self.orig_patch_image_size // 16
        if getattr(self.args, "interpolate_position", False) and image_num_patches > orig_num_patches:
            old_image_position_ids = torch.arange(orig_hw).unsqueeze(0).expand(orig_hw, orig_hw) + \
                                     torch.arange(orig_hw).unsqueeze(1) * self.args.image_bucket_size + 1
            old_image_position_ids = old_image_position_ids.to(device)
            old_image_pos_embed = self.embed_image_positions(old_image_position_ids)
            old_image_pos_embed = old_image_pos_embed.reshape(1, orig_hw, orig_hw, -1).permute(0, 3, 1, 2)
            image_pos_embed = F.interpolate(old_image_pos_embed, size=image_embed_shape, mode='bilinear')
            image_pos_embed = image_pos_embed.permute(0, 2, 3, 1).reshape(1, image_num_patches, -1)
            image_pos_embed = image_pos_embed.expand(patch_images.size(0), -1, -1)
        else:
            image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed, image_embed_shape

    def get_encoder_prompt(self, prompt_tokens):
        past_key_values = self.encoder_prompt_encoder(prompt_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            (self.args.encoder_layers) * 2,
            self.args.encoder_attention_heads,
            self.args.encoder_embed_dim // self.args.encoder_attention_heads,
        )
        past_key_values = self.encoder_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    def forward_embedding(
        self,
        src_tokens,
        image_embed: Optional[torch.Tensor] = None,
        image_embed_2: Optional[torch.Tensor] = None,
        token_embedding: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        image_pos_embed: Optional[torch.Tensor] = None,
        image_pos_embed_2: Optional[torch.Tensor] = None,
    ):
        # embed tokens and positions
        if token_embedding is None:
            # bsz, length = src_tokens.size()
            # src_tokens_flat = src_tokens.flatten()
            
            # offset = self.dictionary.index('<seg_0>')
            # mask = src_tokens_flat < offset
            
            # lang_tokens = src_tokens_flat[mask]
            # lang_embedding = self.embed_tokens(lang_tokens)
            
            # seg_tokens = src_tokens_flat[~mask] - offset
            # seg_embedding = self.seg_embed_tokens(seg_tokens)
            
            # token_embedding = torch.cat([lang_embedding, seg_embedding], dim=0)
            
            # all_idx = torch.arange(bsz*length)
            # lang_idx = all_idx[mask]
            # seg_idx = all_idx[~mask]
            # restore_idx = torch.cat([lang_idx, seg_idx], dim=0).argsort()
            
            # token_embedding = token_embedding[restore_idx]
            # token_embedding = token_embedding.reshape(bsz, length, -1)
            token_embedding = self.embed_tokens(src_tokens)

        x = embed = self.embed_scale * token_embedding
        if self.entangle_position_embedding and pos_embed is not None:
            x += pos_embed
        if self.type_embedding is not None:
            x += self.type_embedding(src_tokens.new_zeros(x.size()[:2]))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # embed raw images
        image_embed_before_scale = None
        if image_embed is not None:
            image_embed_before_scale = self.image_proj(image_embed)            
            image_x = image_embed = self.embed_scale * image_embed_before_scale
            if self.entangle_position_embedding and image_pos_embed is not None:
                image_x += image_pos_embed
            if self.type_embedding is not None:
                image_x += self.type_embedding(src_tokens.new_ones(image_x.size()[:2]))
            if self.patch_layernorm_embedding is not None:
                image_x = self.patch_layernorm_embedding(image_x)
            image_x = self.dropout_module(image_x)
            if self.quant_noise is not None:
                image_x = self.quant_noise(image_x)
            x = torch.cat([image_x, x], dim=1)
            embed = torch.cat([image_embed, embed], dim=1)

        if image_embed_2 is not None:
            assert self.type_embedding is not None
            image_embed_2 = self.image_proj(image_embed_2)
            image_x_2 = image_embed_2 = self.embed_scale * image_embed_2
            if self.entangle_position_embedding and image_pos_embed_2 is not None:
                image_x_2 += image_pos_embed_2
            if self.type_embedding is not None:
                image_x_2 += self.type_embedding(src_tokens.new_full(image_x_2.size()[:2], fill_value=2))
            if self.patch_layernorm_embedding is not None:
                image_x_2 = self.patch_layernorm_embedding(image_x_2)
            image_x_2 = self.dropout_module(image_x_2)
            if self.quant_noise is not None:
                image_x_2 = self.quant_noise(image_x_2)
            x = torch.cat([image_x_2, x], dim=1)
            embed = torch.cat([image_embed_2, embed], dim=1)

        return x, embed, image_embed_before_scale

    def forward(
        self,
        src_tokens,
        src_lengths,
        patch_images: Optional[torch.Tensor] = None,
        fake_image_tokens: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        fake_image_token_offsets: Optional[torch.Tensor] = None,
        mix_mask: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
    ):
        
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # if masked_forward:
        forward_scriptable = self.encode
        
        # else:
        #     forward_scriptable = self.encode
        
        return forward_scriptable(src_tokens,
                                  src_lengths,
                                  patch_images,
                                  fake_image_tokens,
                                  patch_masks,
                                  fake_image_token_offsets,
                                  mix_mask,
                                  return_all_hiddens,
                                  token_embeddings,
                                  sample_patch_num)

    def encode(
        self,
        src_tokens,
        src_lengths,
        patch_images: Optional[torch.Tensor] = None,
        fake_image_tokens: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        fake_image_token_offsets: Optional[torch.Tensor] = None,
        mix_mask: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        prompt_tokens = None
        prompt_padding_mask = None
        prompt_kv_list = None
        if self.args.encoder_prompt:
            bsz, seq_len = src_tokens.shape[0], src_tokens.shape[1]
            if self.args.encoder_prompt_type in ("prefix"):
                prompt_tokens = torch.arange(
                    0, self.args.encoder_prompt_length).to(
                    src_tokens.device)
                prompt_tokens = prompt_tokens.unsqueeze(0).expand(bsz, -1)
                prompt_padding_mask = torch.zeros_like(prompt_tokens).to(prompt_tokens.device)
            prompt_kv_list = self.get_encoder_prompt(prompt_tokens)
        image_embed = None
        image_embed_2 = None
        image_pos_embed = None
        image_pos_embed_2 = None
        if patch_images is not None:
            image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed, image_embed_shape = \
                self.get_patch_images_info(patch_images, sample_patch_num, src_tokens.device)
            image_padding_mask[~patch_masks] = True            
            
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if patch_images is not None:
            encoder_padding_mask = torch.cat([image_padding_mask, encoder_padding_mask], dim=1)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        pos_embed = self.embed_positions(utils.new_arange(src_tokens))
        
        ######
        ## forward embedding
        # embed tokens and positions
        if token_embeddings is None:
            token_embeddings = self.embed_tokens(src_tokens)

        x = embed = self.embed_scale * token_embeddings
        if self.entangle_position_embedding and pos_embed is not None:
            x += pos_embed
        if self.type_embedding is not None:
            x += self.type_embedding(src_tokens.new_zeros(x.size()[:2]))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # embed raw images
        image_embed_before_scale = self.image_proj(image_embed)
        
        if mix_mask is not None:
            # fake_image_tokens are input_ids for 1024 text classes
            # fake_image_token_offsets are offsets for embedding bag
            bsz = fake_image_tokens.size(0)
            fake_image_tokens = fake_image_tokens[fake_image_tokens != self.padding_idx]
            
            fake_image_token_offsets = rearrange(fake_image_token_offsets, '(b l) -> b l', b=bsz)
            fake_image_token_offsets = torch.cat([fake_image_token_offsets.new_zeros(size=(bsz, 1)), fake_image_token_offsets], dim=1)
            offset = torch.cat([fake_image_token_offsets.new_zeros(size=(1, )), fake_image_token_offsets[:-1,-1]]).cumsum(0)
            fake_image_token_offsets = fake_image_token_offsets + offset.unsqueeze(1)
            fake_image_token_offsets = fake_image_token_offsets[:, :-1].flatten()

            fake_image_embed = self.embed_tokens_bag(fake_image_tokens, offsets=fake_image_token_offsets)
            D = fake_image_embed.size(-1)
            fake_image_embed = rearrange(fake_image_embed, '(b l) d -> b l d', b=bsz)
            
            mixed_image_embed_before_scale = torch.where(mix_mask.unsqueeze(-1).expand(-1, -1, D), fake_image_embed, image_embed_before_scale)
        else:
            mixed_image_embed_before_scale = image_embed_before_scale

        image_x = image_embed = self.embed_scale * mixed_image_embed_before_scale
        if self.entangle_position_embedding and image_pos_embed is not None:
            image_x += image_pos_embed
        if self.type_embedding is not None:
            image_x += self.type_embedding(src_tokens.new_ones(image_x.size()[:2]))
        if self.patch_layernorm_embedding is not None:
            image_x = self.patch_layernorm_embedding(image_x)
        image_x = self.dropout_module(image_x)
        if self.quant_noise is not None:
            image_x = self.quant_noise(image_x)
        x = torch.cat([image_x, x], dim=1)
        embed = torch.cat([image_embed, embed], dim=1)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        pos_embed = self.pos_ln(pos_embed)
        if patch_images is not None:
            image_pos_embed = self.image_pos_ln(image_pos_embed)
            pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)
        
        pos_q = self.pos_q_linear(pos_embed).view(
            pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
        ).transpose(1, 2) * self.pos_scaling
        pos_k = self.pos_k_linear(pos_embed).view(
            pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
        ).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        if prompt_padding_mask is not None:
            encoder_padding_mask = torch.cat([prompt_padding_mask, encoder_padding_mask], dim=1)
        # encoder layers
        for idx, layer in enumerate(self.layers):
            self_attn_bias = abs_pos_bias.clone()
            self_attn_bias[:, :, -src_tokens.size(1):, -src_tokens.size(1):] += self.get_rel_pos_bias(src_tokens, idx)
            if patch_images is not None:
                self_attn_bias[:, :, :x.size(0) - src_tokens.size(1), :x.size(0) - src_tokens.size(1)] += \
                    self.get_image_rel_pos_bias(image_position_ids, idx)
            
            self_attn_bias = self_attn_bias.reshape(-1, self_attn_bias.size(2), self_attn_bias.size(2))
            if self.args.encoder_prompt:
                if self.args.encoder_prompt_type != "prompt":
                    prompt_kv = prompt_kv_list[idx]
                else:
                    if idx == 0:
                        prompt_kv = prompt_kv_list[idx]
                    else:
                        prompt_kv = None
            else:
                prompt_kv = None 
            x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, \
                    self_attn_bias=self_attn_bias, prompt_kv=prompt_kv)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.args.encoder_prompt:
            encoder_padding_mask = encoder_padding_mask[:, prompt_tokens.size(1):]
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.

        encoder_embedding = self.embed_tokens.weight.clone()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "position_embeddings": [pos_embed],  # B x T x C
            "patch_images": [patch_images],
            "image_embed_before_scale": [image_embed_before_scale],
            "image_embed_shape": [image_embed_shape],
            "image_embed_before_proj": [image_embed]
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            new_src_tokens = []
        else:
            new_src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            new_src_lengths = []
        else:
            new_src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        if len(encoder_out["position_embeddings"]) == 0:
            new_position_embeddings = []
        else:
            new_position_embeddings = [(encoder_out["position_embeddings"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        if len(encoder_out["patch_images"]) == 0:
            new_patch_images = []
        else:
            new_patch_images = [
                encoder_out["patch_images"][0].index_select(0, new_order)
            ]
            
        if len(encoder_out["image_embed_before_scale"]) == 0:
            new_image_embed_before_scale = []
        else:
            new_image_embed_before_scale = [
                encoder_out["image_embed_before_scale"][0].index_select(0, new_order)
            ]
        
        if len(encoder_out["image_embed_before_proj"]) == 0:
            new_image_embed_before_proj = []
        else:
            new_image_embed_before_proj = [
                encoder_out["image_embed_before_proj"][0].index_select(0, new_order)
            ]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": new_src_tokens,  # B x T
            "src_lengths": new_src_lengths,  # B x 1
            "position_embeddings": new_position_embeddings,  # B x T x C
            "patch_images": new_patch_images,
            "image_embed_before_scale": new_image_embed_before_scale,
            "image_embed_before_proj": new_image_embed_before_proj,
            "image_embed_shape": encoder_out["image_embed_shape"]
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        # version_key = "{}.version".format(name)
        # if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
        #     # earlier checkpoints did not normalize after the stack of layers
        #     self.layer_norm = None
        #     self.normalize = False
        #     state_dict[version_key] = torch.Tensor([1])

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["encoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["encoder.embed_image_positions.weight"])
            embed_dim = state_dict["encoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["encoder.embed_image_positions.weight"].dtype,
            )
            state_dict["encoder.embed_image_positions.weight"] = torch.cat(
                [state_dict["encoder.embed_image_positions.weight"], new_pos_embed_to_add]
            )
        return state_dict

class PromptEncoder(torch.nn.Module):
    r"""
    Prompt encoder to generate prompts, including prompt, prefix, instance and instruction
    """

    def __init__(
            self,
            type,
            length,
            projection,
            embed_dim,
            proj_dim,
            layers,
            vocab_size):
        super().__init__()
        self.prefix_projection = projection

        if type == "prefix":
            layers = layers
            prompt_vocab_size = length

        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(prompt_vocab_size, embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, proj_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_dim, layers * 2 * embed_dim)
            )
        else:
            if type == "prefix":
                self.embedding = torch.nn.Embedding(
                    prompt_vocab_size, layers * 2 * embed_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m