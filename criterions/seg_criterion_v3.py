# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from re import S
from typing import Optional
import torch.distributed as dist

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from mmseg.ops import resize

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import logging
logger = logging.getLogger(__name__)

@dataclass
class SegCriterionV3Config(FairseqDataclass):
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
    "seg_criterion_v3", dataclass=SegCriterionV3Config
)
class SegCriterionV3(FairseqCriterion):
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
        self.alpha = alpha
        self.unlabeled_threshold = unlabeled_threshold
        self.criterion_update_freq = criterion_update_freq
        
        self.hard_rampup_iter = hard_rampup_iter
        self.freeze_embedding_iter = freeze_embedding_iter
        
        self.iter = -1
        self.effective_iter = -1 # effective_iter = iter // criterion_update_freq
        
        self.use_centering = resolve_str_true_false(use_centering)
        self.upscale_lprobs = resolve_str_true_false(upscale_lprobs)
        self.unsupervised_segmentation = resolve_str_true_false(unsupervised_segmentation)
        
        self.seg_id_offset = task.target_dictionary.index("<seg_0>")
        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)
        
        output_classes = 150
        if self.use_centering:
            self.register_buffer("center", torch.zeros(size=(1, output_classes))) # self.center
            self.register_buffer("center_accumulation", torch.zeros(size=(1, output_classes)), persistent=False) # tmp buffer for accumulated updates. # self.center_accumulation
            self.center_momentum = 0.9
            self.register_buffer("center_batch_size", torch.zeros(size=(1, )), persistent=False) # tmp buffer for accumulated updates. # self.center_batch_size

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # 20221006 수정사항
        # count effective iterations given iter and criterion_update_freq
        if model.training:
            self.iter += 1
            self.effective_iter = self.iter // self.criterion_update_freq
        
        # 20221006 수정사항
        # hard ramp-up 적용
        if self.effective_iter < self.hard_rampup_iter:
            if (self.iter == 0) and (self.hard_rampup_iter != 0):
                logger.info(f"Set alpha_coefficient == 0.0 until hard_rampup iterations {self.hard_rampup_iter}")
            alpha_coefficient = 0.0
        else:
            if (self.effective_iter == self.hard_rampup_iter) and (self.iter % self.criterion_update_freq == 0) and (self.hard_rampup_iter != 0):
                logger.info(f"Hard ramp-up alpha_coefficient == {self.alpha} as effective_iter reached {self.hard_rampup_iter} (raw iter == {self.iter})")
            alpha_coefficient = self.alpha
        
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
            
        if self.use_rdrop:
            construct_rdrop_sample(sample)
        
        if self.alpha != 1.0:
            net_output = model(**sample["net_input"], aux_input=sample["aux_input"])
            labeled_output = net_output[1].get("aux_output")
            labeled_loss = self.compute_labeled_loss(model, labeled_output, sample, update_num, reduce=reduce)
            
        else:
            net_output = model(**sample["net_input"])
            labeled_loss = net_output[0].new_zeros(size=(1, ))
        
        if self.unsupervised_segmentation:
            compute_seg_loss = self.compute_unlabled_kld_loss
        else:
            compute_seg_loss = self.compute_loss
                
        seg_loss, nll_loss, threshold_mask, ntokens, area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres, area_intersect, area_pred_label, area_label, area_union = compute_seg_loss(model, net_output, sample, update_num, reduce=reduce)
        
        loss = (1.0 - alpha_coefficient) * labeled_loss + alpha_coefficient * seg_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        
        abs_center = self.center.abs()
        logging_output = {
            "loss": loss.data,
            "labeled_loss": labeled_loss.data,
            "nll_loss": nll_loss.data,
            "alpha_coefficient": alpha_coefficient, # 20221006 수정사항: alpha coefficient 로깅
            "threshold_mask_ratio": threshold_mask.sum() / threshold_mask.numel(),
            "abs_center_max": abs_center.max(),
            "abs_center_mean": abs_center.mean(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "area_intersect_lowres": area_intersect_lowres.data,
            "area_pred_label_lowres": area_pred_label_lowres.data,
            "area_label_lowres": area_label_lowres.data,
            "area_union_lowres": area_union_lowres.data,
            "area_intersect": area_intersect.data,
            "area_pred_label": area_pred_label.data,
            "area_label": area_label.data,
            "area_union": area_union.data
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, return_logit=False):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        # lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        
        logits = net_output[0].float()
        logits_lowres = logits
        
        logits_ = logits[:, :-1]
        logits_ = rearrange(logits_, 'b (h w) d -> b d h w', h=32, w=32)
        logits_ = resize(logits_, size=(512, 512), mode='bilinear', align_corners=False)
        logits_ = rearrange(logits_, 'b d h w -> b (h w) d')
        logits = torch.cat([logits_, logits[:, -1:]], dim=1)

        target_lowres = sample["downsampled_target"]
        target = sample["target"]
        
        # if self.ignore_prefix_size > 0:
        #     lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
        #     target = target[:, self.ignore_prefix_size :].contiguous()
            
        #     if constraint_masks is not None:
        #         constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()
                
        if self.ignore_eos:
            bsz, seq_len, embed_dim = logits_lowres.size()
            eos_indices = target_lowres.eq(self.task.tgt_dict.eos())
            logits_lowres = logits_lowres[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target_lowres = target_lowres[~eos_indices].reshape(bsz, seq_len-1)

            bsz, seq_len, embed_dim = logits.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            logits = logits[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len-1)

        if return_logit:
            lprobs = logits
            lprobs_lowres = logits_lowres
        else:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            lprobs_lowres = F.log_softmax(logits_lowres, dim=-1, dtype=torch.float32)

        #     if constraint_masks is not None:
        #         constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len-1, embed_dim)

        # if constraint_masks is not None:
        #     constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
            
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), lprobs_lowres.view(-1, lprobs_lowres.size(-1)), target_lowres.view(-1), constraint_masks
        
    def compute_labeled_loss(self, model, net_output, sample, update_num, reduce=True):
        # lprobs = model.get_normalized_probs(net_output, log_probs=False)
        logits = net_output[0].float()
        target = sample.get('text2seg_target')

        logits = logits[:, :-1]  # remove eos
        target = target[:, :-1]
        
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        
        logits = logits[target != self.padding_idx]
        target = target[target != self.padding_idx]
        
        target = target - self.seg_id_offset

        # if self.use_centering:
        #     self.update_center(logits)

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            0.0,
            update_num
        )
        return loss

    def compute_unlabled_kld_loss(self, model, net_output, sample, update_num, reduce=True):
        logits, target, logits_lowres, target_lowres, constraint_masks = self.get_lprobs_and_target(model, net_output, sample, return_logit=True)
        
        bsz = sample['net_input']['patch_images'].size(0)
        assert self.ignore_eos
        
        mask = torch.logical_and(target != self.padding_idx, target != (self.seg_id_offset+150))
        logits = logits[mask]
        target = target[mask]
        target = target - self.seg_id_offset
        
        mask_lowres = torch.logical_and(target_lowres != self.padding_idx, target_lowres != (self.seg_id_offset+150))
        logits_lowres = logits_lowres[mask_lowres]
        target_lowres = target_lowres[mask_lowres]
        target_lowres = target_lowres - self.seg_id_offset
        
        if self.upscale_lprobs:
            logits_train = logits
            target_train = target
        else:
            logits_train = logits_lowres
            target_train = target_lowres
        
        student_logits = logits_train / self.student_temperature
        with torch.no_grad():
            if self.use_centering:
                if model.training:
                    self.update_center(logits_train)
                logits_teacher = (logits_train - self.center)
            else:
                logits_teacher = logits_train
            
            pred = F.softmax(logits_teacher, dim=-1)
            max_value, max_index = pred.max(1)
            threshold_mask = (max_value >= self.unlabeled_threshold)
            if self.teacher_temperature == 0.0:
                teacher = max_index
                
            else:
                logits_teacher = logits_teacher / self.teacher_temperature
                teacher = F.softmax(logits_teacher, dim=-1)

        unlabeled_loss = (F.cross_entropy(student_logits, teacher.detach(), label_smoothing=self.eps, reduction='none') * threshold_mask).mean()
        
        ntokens = 1
        unlabeled_loss = unlabeled_loss.mean()

        area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres = self.compute_metric(logits_lowres, target_lowres)
        area_intersect, area_pred_label, area_label, area_union = self.compute_metric(logits, target)
        
        return unlabeled_loss, unlabeled_loss, threshold_mask, ntokens, area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres, area_intersect, area_pred_label, area_label, area_union

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target, lprobs_lowres, target_lowres, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        # if constraint_masks is not None:
        #     constraint_masks = constraint_masks[target != self.padding_idx]
        
        bsz = sample['net_input']['patch_images'].size(0)
        assert self.ignore_eos
        
        mask = torch.logical_and(target != self.padding_idx, target != (self.seg_id_offset+150))
        lprobs = lprobs[mask]
        target = target[mask]
        target = target - self.seg_id_offset
        
        mask_lowres = torch.logical_and(target_lowres != self.padding_idx, target_lowres != (self.seg_id_offset+150))
        lprobs_lowres = lprobs_lowres[mask_lowres]
        target_lowres = target_lowres[mask_lowres]
        target_lowres = target_lowres - self.seg_id_offset
        
        if self.upscale_lprobs:
            lprobs_train = lprobs
            target_train = target
        else:
            lprobs_train = lprobs_lowres
            target_train = target_lowres
                
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs_train,
            target_train,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end
        )
        
        area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres = self.compute_metric(lprobs_lowres, target_lowres)
        area_intersect, area_pred_label, area_label, area_union = self.compute_metric(lprobs, target)
        
        return loss, nll_loss, ntokens, area_intersect_lowres, area_pred_label_lowres, area_label_lowres, area_union_lowres, area_intersect, area_pred_label, area_label, area_union

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
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        # 20221006 수정사항: alpha coefficient 로깅
        alpha_coefficient = logging_outputs[0].get("alpha_coefficient", 0.0)
        
        threshold_mask_ratio_sum = sum(log.get("threshold_mask_ratio", 0.0) for log in logging_outputs)
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