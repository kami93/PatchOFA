# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from re import X
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from .selfpatch import PatchAggregationHead, DINOHead, DINOCentering

logger = logging.getLogger(__name__)

@dataclass
class CustomCriterionV3Config(FairseqDataclass):
    # based on AdjustLabelSmoothedCrossEntropyCriterionConfig
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
        default=False,
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
    embed_dim: int = field(
        default=256, metadata={"help": "embedding dimesion for dino"}
    )
    patch_aggregation_type: str = field(
        default='attention_text_token', metadata={"help": "type for patch aggregation: attention_text_token | attention_cls_token | cossim_top | average | ignore"}
    )
    aggregation_cosim_p: float = field(
        default=1.0, metadata={"help": "aggregation_cosim_p; aggregate those with top p percent high cosine similiarity"}
    )
    aggregation_num_heads: int = field(
        default=4, metadata={"help": "aggregation_num_heads; reference: 4 heads for 256 dim feature"}
    )
    aggregation_cls_inclusive: str = field(
        default='false', metadata={"help": "use cls token for (k, v) in attention layers true | false"}
    )
    out_dim: int = field(
        default=59457, metadata={"help": "output dimesion for token_head and dict_head"}
    )
    student_temp: float = field(
        default=0.1, metadata={"help": "student_temp"}
    )
    teacher_temp: float = field(
        default=0.04, metadata={"help": "teacher_temp"}
    )
    update_step: int = field(
        default=1, metadata={"help": "update_step for dino logit; set as that equals to accumulation count"}
    )
    patch_loss_weight: float = field(
        default=1.0, metadata={"help": "patch_loss_weight."}
    )
    patch_loss_type: str = field(
        default='clsloc_clsglo_kl', metadata={"help": "patch_loss_type."}
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

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    loss = loss.sum()
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
    "custom_criterion_v3", dataclass=CustomCriterionV3Config
)
class CustomCriterionV3(FairseqCriterion):
    # based on AdjustLabelSmoothedCrossEntropyCriterion
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=False,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None,
        embed_dim=256,
        patch_aggregation_type='attention_text_token',
        aggregation_num_heads=4,
        aggregation_cosim_p=1.0,
        aggregation_cls_inclusive='false',
        out_dim=59457,
        teacher_temp=0.04,
        student_temp=0.1,
        update_step=1,
        patch_loss_weight=1.0,
        patch_loss_type='clsloc_clsglo_kl'
    ):
        super().__init__(task)

        self.patch_loss_type = patch_loss_type
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

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        self.aggregation_cls_inclusive = resolve_str_true_false(aggregation_cls_inclusive)

        self.patch_aggregation_type = patch_aggregation_type
        
        if patch_aggregation_type == 'attention_text_token':
            self.aggregation = PatchAggregationHead(embed_dim, aggregation_num_heads, use_cls=False, cosim_p=aggregation_cosim_p, cls_inclusive=self.aggregation_cls_inclusive)
        elif patch_aggregation_type == 'attention_cls_token':
            self.aggregation = PatchAggregationHead(embed_dim, aggregation_num_heads, use_cls=True, cosim_p=aggregation_cosim_p, cls_inclusive=self.aggregation_cls_inclusive)
        elif patch_aggregation_type in {'ignore', 'cossim_top'}:
            self.aggregation = None
        else:
            raise NotImplementedError(f"patch_aggregation_type {patch_aggregation_type} is not implemented.")

        self.use_noun = True
        self.use_object = True

        self.dino_head = DINOHead(in_dim=embed_dim,
                                  out_dim=out_dim)

        self.dino_centering = DINOCentering(dim=out_dim,
                                            update_step=update_step)
        
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.patch_loss_weight = patch_loss_weight

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"]) # **sample["aux_input"]
        loss, nll_loss, patch_kl_loss, clsglo_loss, clsloc_loss, ntokens, batch_size = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )

        patch_loss = patch_kl_loss + clsglo_loss + clsloc_loss
        patch_loss = patch_loss * sample_size
        loss = loss + patch_loss * self.patch_loss_weight

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "patch_loss": patch_loss.data,
            "patch_kl_loss": patch_kl_loss.data,
            "clsglo_loss": clsglo_loss.data,
            "clsloc_loss": clsloc_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len-1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
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

        extra = net_output[1]
        encoder_returns = extra.pop("encoder_returns")
        encoder_out = encoder_returns["encoder_out"][0]
        encoder_embedding = encoder_returns["encoder_embedding"][0]
        # TODO: detect & erase unused embedding indices?

        aux_input = sample.get("aux_input")

        eos_idx = aux_input.get("eos_idx")
        noun_idx = aux_input.get("noun_idx")
        noun_batch_idx = aux_input.get("noun_batch_idx")
        noun_patch_mask = aux_input.get("noun_patch_mask")

        object_idx = aux_input.get("object_idx")
        object_batch_idx = aux_input.get("object_batch_idx")
        object_patch_mask = aux_input.get("object_patch_mask")

        net_input = sample.get("net_input")
        src_tokens = net_input.get("src_tokens")

        _, B, D = encoder_out.shape
        batch_size = B
        LN = noun_idx.size(0) if noun_idx is not None else 0
        LO = object_idx.size(0) if object_idx is not None else 0

        HW = 900
        image_encoding = encoder_out[:HW]
        image_encoding = image_encoding.transpose(0, 1) # '(H W) B D -> B (H W) D'
        text_encoding = encoder_out[HW:] # 'T B D'
        text_encoding = text_encoding.transpose(0, 1)

        L = 0
        text_idx = []
        text_batch_idx = []
        text_patch_mask = []
        if self.use_noun and LN:
            text_idx.append(noun_idx)
            text_batch_idx.append(noun_batch_idx)
            text_patch_mask.append(noun_patch_mask)
            L += LN

        if self.use_object and LO:
            text_idx.append(object_idx)
            text_batch_idx.append(object_batch_idx)
            text_patch_mask.append(object_patch_mask)
            L += LO

        assert L > 0

        clsglo_loss = torch.zeros(size=(1, ), device=encoder_out.device)
        clsloc_loss = torch.zeros(size=(1, ), device=encoder_out.device)
        patch_kl_loss = torch.zeros(size=(1, ), device=encoder_out.device)

        text_idx = torch.cat(text_idx, dim=0)
        text_batch_idx = torch.cat(text_batch_idx, dim=0)
        text_patch_mask = torch.cat(text_patch_mask, dim=0)

        text_tokens = text_encoding[text_batch_idx, text_idx]
        word_labels = src_tokens[text_batch_idx, text_idx]

        patch_encoding_glo = None
        patch_encoding_loc = None
        
        loss_names = self.patch_loss_type.split('_')
        for loss_name in loss_names:
            if loss_name == 'clsglo':
                if patch_encoding_glo is None:
                    patch_encoding_glo = self.aggregation(image_encoding,
                                                            aggr_seed=text_tokens,
                                                            batch_idx=text_batch_idx,
                                                            patch_mask=torch.zeros_like(text_patch_mask))
                feature_glo = self.dino_head(patch_encoding_glo)
                clsglo_loss = F.cross_entropy(feature_glo / self.student_temp, word_labels)

            elif loss_name == 'clsloc':
                if patch_encoding_loc is None:
                    patch_encoding_loc = self.aggregation(image_encoding,
                                                            aggr_seed=text_tokens,
                                                            batch_idx=text_batch_idx,
                                                            patch_mask=text_patch_mask)
                feature_loc = self.dino_head(patch_encoding_loc)
                clsloc_loss = F.cross_entropy(feature_loc / self.student_temp, word_labels)

            elif loss_name in {'kl', 'klsym'}:
                if patch_encoding_glo is None:
                    patch_encoding_glo = self.aggregation(image_encoding,
                                                            aggr_seed=text_tokens,
                                                            batch_idx=text_batch_idx,
                                                            patch_mask=torch.zeros_like(text_patch_mask))
                if patch_encoding_loc is None:
                    patch_encoding_loc = self.aggregation(image_encoding,
                                                            aggr_seed=text_tokens,
                                                            batch_idx=text_batch_idx,
                                                            patch_mask=text_patch_mask) 

                feature_glo = self.dino_head(patch_encoding_glo)
                feature_loc = self.dino_head(patch_encoding_loc)

                teacher = torch.cat([feature_loc, feature_glo], dim=0) if loss_name == 'klsym' else feature_loc
                if 'clsglo' not in loss_names and 'clsloc' not in loss_names:
                    teacher = self.dino_centering(teacher) # Apply centering to teacher if classification is not used.
                
                student = torch.cat([feature_glo, feature_loc], dim=0) if loss_name == 'klsym' else feature_glo
                
                teacher = teacher / self.teacher_temp
                student = student / self.student_temp

                patch_kl_loss = -torch.einsum('bd,bd->b', F.softmax(teacher.detach(), dim=-1), F.log_softmax(student, dim=-1)).mean()
            
            else:
                raise ValueError(f"loss name {loss_name} not supported.")

        return loss, nll_loss, patch_kl_loss, clsglo_loss, clsloc_loss, ntokens, batch_size

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        # loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        # loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        patch_loss_sum = sum(log.get("patch_loss", 0) for log in logging_outputs)
        patch_kl_loss_sum = sum(log.get("patch_kl_loss", 0) for log in logging_outputs)
        clsglo_loss_sum = sum(log.get("clsglo_loss", 0) for log in logging_outputs)
        clsloc_loss_sum = sum(log.get("clsloc_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        # sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "patch_loss", patch_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "patch_kl_loss", patch_kl_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "clsglo_loss", clsglo_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "clsloc_loss", clsloc_loss_sum / sample_size, ntokens, round=3
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
