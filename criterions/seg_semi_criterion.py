# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
import pickle as pkl
import math
from dataclasses import dataclass, field
from re import S, X
from typing import Optional
from collections import OrderedDict

import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from mmseg.ops import resize

from timm.models.layers import trunc_normal_

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import logging
logger = logging.getLogger(__name__)


CLASSES_ADE = np.array([
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag'])

CLASSES_COCOF = np.array([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
    'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
    'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
    'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
    'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
    'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
    'window-blind', 'window-other', 'wood'])

CLASSES_COCOC = np.array([
    'electronic', 'appliance', 'food-things', 'furniture-things', 'indoor', 
    'kitchen', 'accessory', 'animal', 'outdoor', 'person', 
    'sports', 'vehicle', 'ceiling', 'floor', 'food-stuff', 
    'furniture-stuff', 'raw material', 'textile', 'wall', 'window', 
    'building', 'ground', 'plant', 'sky', 'solid', 
    'structural', 'water'])

@dataclass
class SegSemiCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=True,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    upscale_lprobs: str = field(
        default='true',
        metadata={"help": "true | fasle"}
    )
    unsupervised_segmentation: str = field(
        default='true',
        metadata={"help": "true | fasle"}
    )
    unlabeled_threshold: float = field(
        default=0.0, metadata={"help": "labeled loss weight"}
    )
    teacher_temperature: float = field(
        default=0.1, metadata={"help": "teacher temperature for loss"}
    )
    student_temperature: float = field(
        default=0.1, metadata={"help": "teacher temperature for loss"}
    )
    alpha: float = field(
        default=0.5, metadata={"help": "alpha weight"}
    )
    use_centering: str = field(
        default='true', metadata={"help": "whether to apply logit centering"}
    )
    criterion_update_freq: int = field(
        default=1, metadata={"help": "update frequency used in this criterion (e.g., for updating logit center)."}
    )
    hard_rampup_iter: int = field(
        default=0, metadata={
            "help": "Hard rampup (imediately turn on alpha coefficient) at this iteration. ``effective iteration'' (1 iter per N update-freq) is used."}
    )
    freeze_embedding_iter: int = field(
        default=-1, metadata={
            "help": "Freeze the token embedding after this iteration (ignored if -1). ``effective iteration'' (1 iter per N update-freq) is used."}
    )
    full_context_alignment: str = field(
        default='false', metadata={"help": "whether to apply full attention in decoder"}
    )
    unlabeled_target: str = field(
        default='self', metadata={"help": "self | gt | cosine"}
    )
    unlabeled_head_type: str = field(
        default='shared', metadata={"help": "shared | mlp"}
    )
    init_seg_with_text: str = field(
        default='true', metadata={"help": "whether to lazy initialize the segmentation with text embedding bags."}
    )
    mask_cosine_criterion: str = field(
        default='all', metadata={"help": "same | different | all"}
    )
    mask_threshold_criterion: str = field(
        default='all', metadata={"help": "or | and | all"}
    )
    use_alignment: str = field(
        default='false', metadata={"help": "whether to use distribution alignment."}
    )
    unlabeled_loss_type: str = field(
        default='ce', metadata={"help": "ce | focal"}
    )
    
def resolve_str_true_false(x):
    x = x.lower()
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        raise ValueError(f"Unable to recognize string bool input: {x}")

def construct_rdrop_sample(x):
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim()-1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(
        lprobs, target, epsilon, update_num, reduce=True,
        drop_worst_ratio=0.0, drop_worst_after=0, use_rdrop=False, reg_alpha=1.0,
        constraint_masks=None, constraint_start=None, constraint_end=None
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(loss[:true_batch_size], k=int(true_batch_size * (1 - drop_worst_ratio)), largest=False)
            loss = torch.cat([loss[indices], loss[indices+true_batch_size]])
            nll_loss = torch.cat([nll_loss[indices], nll_loss[indices+true_batch_size]])
            lprobs = torch.cat([lprobs[indices], lprobs[indices+true_batch_size]])
        else:
            loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = 1
    nll_loss = nll_loss.mean()
    loss = loss.mean()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens

@register_criterion(
    "seg_semi_criterion", dataclass=SegSemiCriterionConfig
)
class SegSemiCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=True,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None,
        upscale_lprobs='true',
        unsupervised_segmentation='true',
        teacher_temperature=0.1,
        student_temperature=0.1,
        alpha=0.5,
        unlabeled_threshold=0.0,
        use_centering='true',
        criterion_update_freq=1,
        hard_rampup_iter=0,
        freeze_embedding_iter=-1,
        full_context_alignment='false',
        unlabeled_target='self',
        unlabeled_head_type='shared',
        init_seg_with_text='true',
        mask_cosine_criterion='all',
        mask_threshold_criterion='all',
        use_alignment='false',
        unlabeled_loss_type='ce'
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.effective_alpha = self.alpha = alpha
        self.unlabeled_threshold = unlabeled_threshold
        self.criterion_update_freq = criterion_update_freq
        
        self.hard_rampup_iter = hard_rampup_iter
        self.freeze_embedding_iter = freeze_embedding_iter
        
        self.iter = -1
        self.effective_iter = -1 # effective_iter = iter // criterion_update_freq
        
        self.use_centering = resolve_str_true_false(use_centering)
        self.upscale_lprobs = resolve_str_true_false(upscale_lprobs)
        self.unsupervised_segmentation = resolve_str_true_false(unsupervised_segmentation)
        self.full_context_alignment = resolve_str_true_false(full_context_alignment)
        self.init_seg_with_text = resolve_str_true_false(init_seg_with_text)
        self.use_alignment = resolve_str_true_false(use_alignment)
        
        self.mask_cosine_criterion = mask_cosine_criterion
        self.mask_threshold_criterion = mask_threshold_criterion
        
        self.unlabeled_target = unlabeled_target
        self.unlabeled_head_type = unlabeled_head_type
        self.unlabeled_loss_type = unlabeled_loss_type
        
        self.seg_id_offset = task.target_dictionary.index("<seg_0>")
        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)
        
        if self.unlabeled_head_type == 'shared':
            # assert self.student_temperature == 1.0
            self.output_classes = task.cfg.num_seg_tokens
        elif self.unlabeled_head_type == 'separate':
            self.output_classes = 8192
            self.unlabeled_head = DINOHead(in_dim=256, out_dim=self.output_classes, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256)
        else:
            raise NotImplementedError
        
        if self.use_alignment:
            labeled_num_samples = task.cfg.labeled_num_samples
            with open(f"label_avg_{labeled_num_samples}.pkl", "rb") as f:
                label_avg = torch.tensor(pkl.load(f)).unsqueeze(0)
                self.register_buffer("label_avg", label_avg, persistent=False) # self.label_avg
            self.unlabel_avg = []

        if self.use_centering:
            self.register_buffer("center", torch.zeros(size=(1, self.output_classes))) # self.center
            self.register_buffer("center_accumulation", torch.zeros(size=(1, self.output_classes)), persistent=False) # tmp buffer for accumulated updates. # self.center_accumulation
            self.center_momentum = 0.9
            self.register_buffer("center_batch_size", torch.zeros(size=(1, )), persistent=False) # tmp buffer for accumulated updates. # self.center_batch_size

    def forward(self, model, sample, update_num=0, reduce=True, ema_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.iter == -1 and update_num != 0:
            self.iter = self.criterion_update_freq * update_num - 1
            logger.info(f"Restored iteration counts {self.iter}")
        
        if self.init_seg_with_text and self.iter == -1:
            # Do lazy initializations;
            def encode_text(text):
                line = [self.task.bpe.encode(' {}'.format(word.strip())) for word in text.strip().split()]
                line = ' '.join(line)
                
                s = self.task.tgt_dict.encode_line(
                    line=line,
                    add_if_not_exist=False,
                    append_eos=False
                ).long()
                return s
            
            if self.output_classes == 150:
                CLASSES = CLASSES_ADE
            elif self.output_classes == 171:
                CLASSES = CLASSES_COCOF
            elif self.output_classes == 27:
                CLASSES = CLASSES_COCOC
            else:
                raise NotImplementedError

            with torch.no_grad():
                id2text = [encode_text(f" {x}") for x in CLASSES]
                id2text_tokens = torch.cat(id2text).cuda()
                
                text_length = torch.tensor([len(x) for x in id2text])
                start_offset = torch.cat([text_length.new_zeros(size=(1, ), dtype=torch.long), text_length.cumsum(dim=0)[:-1]], dim=0).cuda()
                
                avg_embedding = model.encoder.embed_tokens_bag(id2text_tokens, offsets=start_offset)
                model.encoder.seg_embed_tokens.weight.data = avg_embedding.data
                model.decoder.seg_embed_tokens.weight.data = avg_embedding.data
                if ema_model is not None:
                    ema_model.encoder.seg_embed_tokens.weight.data = avg_embedding.data
                    ema_model.decoder.seg_embed_tokens.weight.data = avg_embedding.data
                
                if not model.decoder.tie_seg_projection:
                    model.decoder.seg_projection.weight.data = avg_embedding.data
                    if ema_model is not None:
                        ema_model.decoder.seg_projection.weight.data = avg_embedding.data
                    
            logger.info("Initialized seg tokens with embedding bag.")

        # 20221006 수정사항
        # count effective iterations given iter and criterion_update_freq
        if model.training:
            self.iter += 1
            self.effective_iter = self.iter // self.criterion_update_freq
        
        # 20221006 수정사항
        # hard ramp-up 적용
        if self.effective_iter < self.hard_rampup_iter:
            if (self.iter == 0) and (self.hard_rampup_iter != 0):
                logger.info(f"Set effective_alpha == 0.0 until hard_rampup iterations {self.hard_rampup_iter}")
            self.effective_alpha = 0.0
        else:
            if (self.effective_iter == self.hard_rampup_iter) and (self.iter % self.criterion_update_freq == 0) and (self.hard_rampup_iter != 0):
                logger.info(f"Hard ramp-up effective_alpha == {self.alpha} as effective_iter reached {self.hard_rampup_iter} (raw iter == {self.iter})")
            self.effective_alpha = self.alpha
        
        # 20221006 수정사항
        # embedding freeze iteration 기능 적용
        if (self.freeze_embedding_iter == -1) or (self.effective_iter < self.freeze_embedding_iter):
            pass
        else:
            if (self.effective_iter == self.freeze_embedding_iter) and (self.iter % self.criterion_update_freq == 0):
                logger.info(f"Freezing embeddings as effective_iter reached {self.freeze_embedding_iter} (raw iter == {self.iter})")
            model.encoder.embed_tokens.weight.requires_grad_(False)
            model.decoder.embed_tokens.weight.requires_grad_(False)
            model.decoder.seg_projection.weight.requires_grad_(False)
        
        if self.use_centering:
            # send centering buffers to a gpu device, if applicable
            self.center = self.center.to('cuda')
            self.center_accumulation = self.center_accumulation.to('cuda')
            self.center_batch_size = self.center_batch_size.to('cuda')
        
        if self.use_alignment:
            self.label_avg = self.label_avg.to('cuda')
            
        if self.use_rdrop:
            construct_rdrop_sample(sample)
        
        if self.alpha != 1.0 and model.training:
            net_output = model(**sample["net_input"], full_context_alignment=self.full_context_alignment, labeled_input=sample["labeled_input"])
            labeled_output = net_output[1].get("labeled_output")
            labeled_loss = self.compute_labeled_loss(model, labeled_output, sample, update_num, reduce=reduce)
            
        else:
            net_output = model(**sample["net_input"])
            labeled_loss = net_output[0].new_zeros(size=(1, ))

        # torch.save(net_output, "net_output.pt")
        # torch.save(sample, "sample.pt")

        if self.unsupervised_segmentation:
            compute_seg_loss = self.compute_unlabled_kld_loss
        else:
            compute_seg_loss = self.compute_loss
                
        seg_loss, metrics, ntokens = compute_seg_loss(model, net_output, sample, update_num, reduce=reduce, ema_model=ema_model)
        
        loss = (1.0 - self.effective_alpha) * labeled_loss + self.effective_alpha * seg_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        
        logging_output = {
            "loss": loss.data,
            "labeled_loss": labeled_loss.data,
            "seg_loss": seg_loss.data,
            "alpha_coefficient": self.effective_alpha, # 20221006 수정사항: alpha coefficient 로깅
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.data
            logging_output[key] = value

        if self.use_centering:
            abs_center = self.center.abs()
            logging_output["abs_center_max"] = abs_center.max()
            logging_output["abs_center_mean"] = abs_center.mean()
        
        # if self.report_accuracy:
        #     n_correct, total = self.compute_accuracy(model, net_output, sample)
        #     logging_output["n_correct"] = utils.item(n_correct.data)
        #     logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def upsample_logits(self, logits):
        logits_ = logits[:, :-1]
        logits_ = rearrange(logits_, 'b (h w) d -> b d h w', h=32, w=32)
        logits_ = resize(logits_, size=(512, 512), mode='bilinear', align_corners=False)
        logits_ = rearrange(logits_, 'b d h w -> b (h w) d')
        logits = torch.cat([logits_, logits[:, -1:]], dim=1)
        
        return logits
        
    def compute_labeled_loss(self, model, net_output, sample, update_num, reduce=True):
        classifier_logits_lowres, extra = net_output
        # calculate upscaled versions
        if self.upscale_lprobs:
            classifier_logits = self.upsample_logits(classifier_logits_lowres) # bilinear upsample

            target = sample.get("target_labeled")
            target_shape = target.shape
            assert target_shape == classifier_logits.shape[:-1]

            sample_masks = torch.logical_or(target == self.padding_idx, target == (self.seg_id_offset+self.output_classes))
            if self.ignore_eos:
                eos_masks = target.eq(self.task.tgt_dict.eos())
                sample_masks = torch.logical_or(sample_masks, eos_masks)
            target = target[~sample_masks] - self.seg_id_offset
            classifier_logits = classifier_logits[~sample_masks]
            logits_labeled = classifier_logits
            target_labeled = target
        else:
            target_lowres = sample.get("downsampled_target_labeled")
            sample_masks_lowres = None # ignoring idx mask for "sample" dimension
            classifier_logits_lowres = classifier_logits_lowres.float()
            
            target_lowres_shape = target_lowres.shape
            assert target_lowres_shape == classifier_logits_lowres.shape[:-1]

            sample_masks_lowres = torch.logical_or(target_lowres == self.padding_idx, target_lowres == (self.seg_id_offset+self.output_classes))
            if self.ignore_eos:
                eos_masks_lowres = target_lowres.eq(self.task.tgt_dict.eos())
                sample_masks_lowres = torch.logical_or(sample_masks_lowres, eos_masks_lowres)

            # apply masking to targets
            target_lowres = target_lowres[~sample_masks_lowres] - self.seg_id_offset
            classifier_logits_lowres = classifier_logits_lowres[~sample_masks_lowres]
            logits_labeled = classifier_logits_lowres
            target_labeled = target_lowres

        loss = F.cross_entropy(logits_labeled, target_labeled.detach(), label_smoothing=self.eps)

        return loss

    def compute_unlabled_kld_loss(self, model, net_output, sample, update_num, reduce=True, ema_model=None):
        classifier_logits_lowres, extra = net_output

        target_lowres = sample.get("downsampled_target")
        sample_masks_lowres = None # ignoring idx mask for "sample" dimension
        constraint_masks_lowres = None # ignoring idx mask for "channel" dimension

        classifier_logits_lowres = classifier_logits_lowres.float()
        
        target_lowres_shape = target_lowres.shape
        assert target_lowres_shape == classifier_logits_lowres.shape[:-1]

        sample_masks_lowres = torch.logical_or(target_lowres == self.padding_idx, target_lowres == (self.seg_id_offset+self.output_classes))
        if self.ignore_eos:
            eos_masks_lowres = target_lowres.eq(self.task.tgt_dict.eos())
            sample_masks_lowres = torch.logical_or(sample_masks_lowres, eos_masks_lowres)

        # calculate upscaled versions
        target = sample.get("target")
        sample_masks = None
        constraint_masks = None

        classifier_logits = self.upsample_logits(classifier_logits_lowres) # bilinear upsample

        target_shape = target.shape
        assert target_shape == classifier_logits.shape[:-1]

        sample_masks = torch.logical_or(target == self.padding_idx, target == (self.seg_id_offset+self.output_classes))
        if self.ignore_eos:
            eos_masks = target.eq(self.task.tgt_dict.eos())
            sample_masks = torch.logical_or(sample_masks, eos_masks)

        # apply masking to targets
        target_lowres = target_lowres[~sample_masks_lowres] - self.seg_id_offset
        target = target[~sample_masks] - self.seg_id_offset

        # classifier_logits_lowres = classifier_logits_lowres[~sample_masks_lowres]
        # classifier_logits = classifier_logits[~sample_masks]

        if self.unlabeled_head_type == 'shared':
            logits_train = classifier_logits_lowres
            target_train = target_lowres
            constraint_masks_train = constraint_masks_lowres
            # sample_masks_train = sample_masks_lowres
            if self.upscale_lprobs:
                logits_train = classifier_logits
                target_train = target
                constraint_masks_train = constraint_masks
                # sample_masks_train = sample_masks
        
        elif self.unlabeled_head_type == 'separate':
            features = extra.get('penultimate')
            logits_train = self.unlabeled_head(features)
            assert target_lowres_shape == logits_train.shape[:-1]

            target_train = target_lowres
            constraint_masks_train = constraint_masks_lowres
            # sample_masks_train = sample_masks_lowres

        with torch.no_grad():
            if ema_model is not None:
                logits_ema, extra_ema = ema_model(**sample["net_input"], full_context_alignment=self.full_context_alignment)
                if self.unlabeled_head_type == 'shared':
                    logits_ema = logits_ema.float()
                elif self.unlabeled_head_type == 'separate':
                    features_ema = extra_ema.get('penultimate')
                    logits_ema = self.unlabeled_head(features_ema)
            else:
                logits_ema = logits_train

            if self.unlabeled_target == 'gt':
                raise NotImplementedError
                len_target = len(target_train)
                mask = target_train.unsqueeze(0) == target_train.unsqueeze(1)
                # mask.diagonal()[:] = False
                
                rand = torch.randn(size=(len_target, len_target), device=target_train.device)
                perm = rand.argsort(-1)
                
                batch_idx_1d = torch.arange(len_target)
                batch_idx_2d = batch_idx_1d.unsqueeze(-1).expand(-1, len_target)

                mask_perm = mask[batch_idx_2d, perm]
                random_choice = mask_perm.max(-1)[1]
                
                target_teacher = perm[batch_idx_1d, random_choice]
                logits_teacher = logits_teacher[target_teacher]
            
            elif self.unlabeled_target == 'self':
                logits_teacher = logits_ema
                logits_teacher = logits_teacher[:, :-1].reshape(-1, logits_teacher.size(-1))


            elif self.unlabeled_target == 'resnet_cosine':
                resnet_features = extra.get('encoder_returns').get('image_embed_before_proj')[0]
                resnet_features_n = F.normalize(resnet_features, dim=-1)

                sim = resnet_features_n @ resnet_features_n.transpose(-2,-1)
                sim.diagonal(dim1=1, dim2=2)[:] = float("-inf")

                closest = sim.argmax(-1)
                bsz, seqlen = closest.size()
                batch_idx = torch.arange(bsz).unsqueeze(-1).expand(-1, seqlen)
                logits_teacher = logits_ema[batch_idx, closest]

                logits_teacher = logits_teacher.reshape(-1, logits_teacher.size(-1))
                logits_teacher = torch.cat([logits_teacher, logits_ema[:, :-1].reshape(-1, logits_ema.size(-1))], dim=0)
            
            elif self.unlabeled_target == 'cosine':
                raise NotImplementedError

            if self.use_centering:
                if model.training and self.effective_alpha != 0.0:
                    self.update_center(logits_teacher)
                logits_teacher = (logits_teacher - self.center)
                if self.teacher_temperature == 0.0:
                    logits_teacher = logits_teacher / 0.07
            
            if constraint_masks_train is not None:
                logits_teacher[constraint_masks_train] = float('-inf')

            if self.unlabeled_target == 'resnet_cosine':
                ns = len(logits_teacher)
                logits_teacher_index = logits_teacher.argmax(-1)
                
                if self.mask_cosine_criterion == 'same':
                    mask_cosine = logits_teacher_index[:ns//2] == logits_teacher_index[ns//2:]
                elif self.mask_cosine_criterion == 'different':
                    mask_cosine = logits_teacher_index[:ns//2] != logits_teacher_index[ns//2:]
                elif self.mask_cosine_criterion == 'all':
                    mask_cosine = torch.ones_like(logits_teacher_index[:ns//2], dtype=torch.bool)
                else:
                    raise NotImplementedError

                pred = F.softmax(logits_teacher, dim=-1)
                max_value, max_index = pred.max(1)
                
                if self.mask_threshold_criterion == 'or':
                    threshold_mask = (max_value >= self.unlabeled_threshold)
                    mask_thres = torch.logical_or(threshold_mask[:ns//2], threshold_mask[ns//2:])     
                    threshold_mask = torch.logical_and(mask_cosine, mask_thres)
                    threshold_mask = torch.cat([threshold_mask, threshold_mask], dim=0)
                elif self.mask_threshold_criterion == 'and':
                    threshold_mask = (max_value >= self.unlabeled_threshold)
                    mask_thres = torch.logical_and(threshold_mask[:ns//2], threshold_mask[ns//2:])     
                    threshold_mask = torch.logical_and(mask_cosine, mask_thres)
                    threshold_mask = torch.cat([threshold_mask, threshold_mask], dim=0)
                elif self.mask_threshold_criterion == 'all':
                    threshold_mask = (max_value >= self.unlabeled_threshold)
                    mask_cosine = torch.cat([mask_cosine, mask_cosine], dim=0)
                    threshold_mask = torch.logical_and(mask_cosine, threshold_mask)
                else:
                    raise NotImplementedError
                        
            else:
                pred = F.softmax(logits_teacher, dim=-1)
                max_value, max_index = pred.max(1)
                threshold_mask = (max_value >= self.unlabeled_threshold)

            if self.use_alignment and self.effective_alpha != 0.0:
                self.unlabel_avg.append(pred.mean(0))
                self.unlabel_avg = self.unlabel_avg[-(128 // get_world_size()):]
                unlabel_avg = torch.stack(self.unlabel_avg, dim=0)
                unlabel_avg = unlabel_avg.mean(0, keepdim=True)

                if is_dist_avail_and_initialized() and get_world_size() > 1:
                    dist.all_reduce(unlabel_avg)
                    unlabel_avg = unlabel_avg / get_world_size()
                                
                target_ankor = (1e-6 + self.label_avg) / (1e-6 + unlabel_avg)
                pred = pred * target_ankor.detach()

            if self.teacher_temperature == 0.0:
                teacher = max_index
            else:
                logits_teacher = logits_teacher / self.teacher_temperature
                teacher = F.softmax(logits_teacher, dim=-1)

        logits_student = logits_train / self.student_temperature
        if self.unlabeled_target == 'resnet_cosine':
            logits_student = torch.cat([logits_student[:, :-1].reshape(-1, logits_student.size(-1)), logits_student[batch_idx, closest].reshape(-1, logits_student.size(-1))], dim=0)
        else:
            logits_student = logits_student[:, :-1].reshape(-1, logits_student.size(-1))

        if self.unlabeled_loss_type == 'ce':
            unlabeled_loss = (F.cross_entropy(logits_student, teacher.detach(), reduction='none') * threshold_mask).mean()
        elif self.unlabeled_loss_type == 'focal':
            prob = logits_student.softmax(-1).max(-1)[0]
            unlabeled_loss = (F.cross_entropy(logits_student, teacher.detach(), reduction='none') * threshold_mask * (1 - prob)**2).mean()
        else:
            raise NotImplementedError
        ntokens = 1

        metrics = dict()
        with torch.no_grad():
            classifier_logits_lowres = classifier_logits_lowres[~sample_masks_lowres]
            classifier_logits = classifier_logits[~sample_masks]
            
            metrics["threshold_mask_ratio"] = threshold_mask.sum() / threshold_mask.numel()
            if self.unlabeled_head_type == 'shared':
                if constraint_masks_train is not None and model.training:
                    classifier_logits_lowres[constraint_masks_train] = float('-inf')
                
                num_samples = classifier_logits_lowres.size(0)
                threshold_mask = threshold_mask[-num_samples:]
                if threshold_mask.sum().item():
                    metrics["threshold_acc"] = (classifier_logits_lowres.argmax(-1) == target_train)[threshold_mask].float().mean()
                else:
                    metrics["threshold_acc"] = torch.tensor([0.0], device=logits_train.device)

            metrics["nll_loss"] = F.cross_entropy(classifier_logits_lowres.detach(), target_lowres.detach()) # just for display

            area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres = self.compute_metric(classifier_logits_lowres.detach(), target_lowres.detach())
            metrics["area_intersect_lowres"] = area_intersect_lowres
            metrics["area_pred_label_lowres"] = area_pred_label_lowres
            metrics["area_label_lowres"] = area_label_lowres
            metrics["area_union_lowres"] = area_union_lowres

            area_intersect, area_pred_label, area_label, area_union = self.compute_metric(classifier_logits.detach(), target.detach())
            metrics["area_intersect"] = area_intersect
            metrics["area_pred_label"] = area_pred_label
            metrics["area_label"] = area_label
            metrics["area_union"] = area_union
        
        return unlabeled_loss, metrics, ntokens

    def compute_loss(self, model, net_output, sample, update_num, reduce=True, ema_model=None):
        classifier_scores_lowres, extra = net_output

        target_lowres = sample.get("downsampled_target")
        sample_masks_lowres = None # ignoring idx mask for "sample" dimension

        classifier_scores_lowres = classifier_scores_lowres.float()
        target_lowres_shape = target_lowres.shape
        assert target_lowres_shape == classifier_scores_lowres.shape[:-1]

        sample_masks_lowres = torch.logical_or(target_lowres == self.padding_idx, target_lowres == (self.seg_id_offset+self.output_classes))
        eos_masks_lowres = target_lowres.eq(self.task.tgt_dict.eos())
        sample_masks_lowres = torch.logical_or(sample_masks_lowres, eos_masks_lowres)

        # calculate upscaled versions
        target = sample.get("target")
        sample_masks = None

        classifier_scores = self.upsample_logits(classifier_scores_lowres) # bilinear upsample
        target_shape = target.shape
        assert target_shape == classifier_scores.shape[:-1]

        sample_masks = torch.logical_or(target == self.padding_idx, target == (self.seg_id_offset+self.output_classes))
        eos_masks = target.eq(self.task.tgt_dict.eos())
        sample_masks = torch.logical_or(sample_masks, eos_masks)

        # apply masking to targets
        target_lowres = target_lowres[~sample_masks_lowres] - self.seg_id_offset
        target = target[~sample_masks] - self.seg_id_offset

        ntokens = 1

        classifier_scores_lowres = classifier_scores_lowres[~sample_masks_lowres]
        classifier_scores = classifier_scores[~sample_masks]

        if self.upscale_lprobs:
            loss = F.cross_entropy(classifier_scores, target.detach(), label_smoothing=self.eps) # just for display
        else:
            loss = F.cross_entropy(classifier_scores_lowres, target_lowres.detach(), label_smoothing=self.eps) # just for display

        area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres = self.compute_metric(classifier_scores_lowres.detach(), target_lowres.detach())
        area_intersect, area_pred_label, area_label, area_union = self.compute_metric(classifier_scores.detach(), target.detach())

        metrics = dict()
        metrics["nll_loss"] = loss

        metrics["area_intersect_lowres"] = area_intersect_lowres
        metrics["area_pred_label_lowres"] = area_pred_label_lowres
        metrics["area_label_lowres"] = area_label_lowres
        metrics["area_union_lowres"] = area_union_lowres

        metrics["area_intersect"] = area_intersect
        metrics["area_pred_label"] = area_pred_label
        metrics["area_label"] = area_label
        metrics["area_union"] = area_union

        return loss, metrics, ntokens

    def compute_metric(self, lprobs, target):
        num_classes = lprobs.size(-1)
        pred_label = lprobs.argmax(-1)

        intersect = pred_label[pred_label == target]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_label = torch.histc(
            target.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        
        return area_intersect, area_pred_label, area_label, area_union

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @torch.no_grad()
    def update_center(self, tokens):
        """
        Update centers
        """
        B = torch.tensor([tokens.size(0)], device=tokens.device, dtype=torch.float)

        tokens_center = torch.sum(tokens.float(), dim=0, keepdim=True)
        if is_dist_avail_and_initialized() and get_world_size() > 1:
            dist.all_reduce(tokens_center)
            dist.all_reduce(B)

        self.center_accumulation = self.center_accumulation.type_as(tokens_center) + tokens_center
        self.center_batch_size = self.center_batch_size.type_as(B) + B

        if (self.iter+1) % self.criterion_update_freq == 0:
            new_center = self.center_accumulation / self.center_batch_size

            self.center = self.center.type_as(self.center_accumulation) * self.center_momentum + new_center * (1 - self.center_momentum)
            
            self.center_accumulation[:] = 0.0
            self.center_batch_size[:] = 0.0

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        labeled_loss_sum = sum(log.get("labeled_loss", 0) for log in logging_outputs)
        seg_loss_sum = sum(log.get("seg_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        # 20221006 수정사항: alpha coefficient 로깅
        alpha_coefficient = logging_outputs[0].get("alpha_coefficient", 0.0)
        
        threshold_mask_ratio_sum = sum(log.get("threshold_mask_ratio", 0.0) for log in logging_outputs)
        threshold_acc_sum = sum(log.get("threshold_acc", 0.0) for log in logging_outputs)

        abs_center_max = logging_outputs[0].get("abs_center_max", 0.0)
        abs_center_mean = logging_outputs[0].get("abs_center_mean", 0.0)
        
        area_intersect_sum = sum(log.get("area_intersect", 0) for log in logging_outputs)
        area_pred_label_sum = sum(log.get("area_pred_label", 0) for log in logging_outputs)
        area_label_sum = sum(log.get("area_label", 0) for log in logging_outputs)
        area_union_sum = sum(log.get("area_union", 0) for log in logging_outputs)

        area_intersect_lowres_sum = sum(log.get("area_intersect_lowres", 0) for log in logging_outputs)
        area_pred_label_lowres_sum = sum(log.get("area_pred_label_lowres", 0) for log in logging_outputs)
        area_label_lowres_sum = sum(log.get("area_label_lowres", 0) for log in logging_outputs)
        area_union_lowres_sum = sum(log.get("area_union_lowres", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "labeled_loss", labeled_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "seg_loss", seg_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        
        # 20221006 수정사항: alpha coefficient 로깅
        metrics.log_scalar(
            "alpha_coefficient", alpha_coefficient / sample_size, 1, round=3
        )
        metrics.log_scalar(
            "threshold_mask_ratio", threshold_mask_ratio_sum / sample_size, 1, round=3
        )
        metrics.log_scalar(
            "threshold_acc", threshold_acc_sum / sample_size, 1, round=3
        )

        metrics.log_scalar(
            "abs_center_max", abs_center_max / sample_size, 1, round=3
        )
        metrics.log_scalar(
            "abs_center_mean", abs_center_mean / sample_size, 1, round=3
        )
        metrics.log_scalar_sum(
            "_area_intersect", area_intersect_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_pred_label", area_pred_label_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_label", area_label_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_union", area_union_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_intersect_lowres", area_intersect_lowres_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_pred_label_lowres", area_pred_label_lowres_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_label_lowres", area_label_lowres_sum, 1
        )
        metrics.log_scalar_sum(
            "_area_union_lowres", area_union_lowres_sum, 1
        )

        def compute_all_acc(meters):
            all_acc = meters['_area_intersect'].sum.sum() / meters['_area_pred_label'].sum.sum()
            all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
            
            return round(all_acc, 4)

        def compute_mean_iou(meters):
            miou = torch.nanmean(meters['_area_intersect'].sum / (meters['_area_union'].sum))
            miou = miou if isinstance(miou, float) else miou.item()
            
            return round(miou, 4)

        def compute_mean_acc(meters):
            macc = torch.nanmean(meters['_area_intersect'].sum / (meters['_area_label'].sum)) # nanmean
            macc = macc if isinstance(macc, float) else macc.item()
            
            return round(macc, 4)

        def compute_all_acc_lowres(meters):
            all_acc = meters['_area_intersect_lowres'].sum.sum() / meters['_area_pred_label_lowres'].sum.sum()
            all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
            
            return round(all_acc, 4)

        def compute_mean_iou_lowres(meters):
            miou = torch.nanmean(meters['_area_intersect_lowres'].sum / (meters['_area_union_lowres'].sum))
            miou = miou if isinstance(miou, float) else miou.item()
            
            return round(miou, 4)

        def compute_mean_acc_lowres(meters):
            macc = torch.nanmean(meters['_area_intersect_lowres'].sum / (meters['_area_label_lowres'].sum))
            macc = macc if isinstance(macc, float) else macc.item()
            
            return round(macc, 4)

        metrics.log_derived("aAcc", compute_all_acc)
        metrics.log_derived("mIoU", compute_mean_iou)
        metrics.log_derived("mAcc", compute_mean_acc)

        metrics.log_derived("aAcc_lowres", compute_all_acc_lowres)
        metrics.log_derived("mIoU_lowres", compute_mean_iou_lowres)
        metrics.log_derived("mAcc_lowres", compute_mean_acc_lowres)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=256, bottleneck_dim=64):
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
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init_weights)

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