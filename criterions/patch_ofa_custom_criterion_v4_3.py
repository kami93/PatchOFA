# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import LayerNorm
from omegaconf import II

from einops import rearrange

from .selfpatch import HeadMlp, DINOCentering

logger = logging.getLogger(__name__)

@dataclass
class CustomCriterionV4_3Config(FairseqDataclass):
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
    dictionary_size: int = field(
        default=59457, metadata={"help": "dictionary_size for token_head"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "embedding dimesion for token_head"}
    )
    token_head_type: str = field(
        default='random', metadata={"help": "layer type for token_head_type: random | dict | freeze"}
    )
    token_head_depth: int = field(
        default=4, metadata={"help": "token_head_depth"}
    )
    token_head_dropout: float = field(
        default=0.0, metadata={"help": "token_head_dropout"}
    )
    loss_weight: float = field(
        default=1.0, metadata={"help": "labeled loss weight"}
    )
    unlabeled_threshold: float = field(
        default=0.0, metadata={"help": "labeled loss weight"}
    )
    loss_type: str = field(
        default='cross_entropy', metadata={"help": "cross_entropy | kld"}
    )
    teacher_temperature: float = field(
        default=0.1, metadata={"help": "teacher temperature for loss"}
    )
    student_temperature: float = field(
        default=0.1, metadata={"help": "teacher temperature for loss"}
    )
    use_centering: str = field(
        default='true', metadata={"help": "whether to apply logit centering"}
    )
    centering_update_freq: int = field(
        default=1, metadata={"help": "update frequency for logit center."}
    )
    alpha: float = field(
        default=0.5, metadata={"help": "alpha weight"}
    )
    center_type: str = field(
        default='image', metadata={"help": "image | text | sentence"}
    )
    use_self_patch: str = field(
        default='false', metadata={"help": "true | false"}
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
    "custom_criterion_v4-3", dataclass=CustomCriterionV4_3Config
)
class CustomCriterionV4_3(FairseqCriterion):
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
        dictionary_size=59457,
        embed_dim=256,
        token_head_type='random', # random | dict | freeze
        token_head_depth=4,
        token_head_dropout=0.0,
        unlabeled_threshold=0.0,
        loss_type='cross_entropy', # cross_entropy | kld
        loss_weight=1.0,
        teacher_temperature=0.1,
        student_temperature=0.1,
        use_centering='false',
        centering_update_freq=1,
        alpha=0.5,
        center_type='image',
        use_self_patch='false',
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

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        self.token_head_type = token_head_type
        self.token_head_depth = token_head_depth
        self.token_head_dropout = token_head_dropout

        self.loss_type = loss_type
        self.unlabeled_threshold = unlabeled_threshold
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.center_type = center_type

        self.alpha = alpha

        self.token_ln = LayerNorm(embed_dim)
        self.token_head = HeadMlp(in_features=embed_dim, hidden_features=embed_dim, out_features=dictionary_size, nlayers=self.token_head_depth, drop=self.token_head_dropout)
        self.token_head_initialized = False if token_head_type in {'dict', 'freeze'} else True

        if resolve_str_true_false(use_centering):
            self.centering = DINOCentering(dictionary_size, update_step=centering_update_freq)
        else:
            self.centering = nn.Identity()
            
        self.use_self_patch = resolve_str_true_false(use_self_patch)

        self.loss_weight = loss_weight

        self_patch_weight = torch.ones(size=(1, 1, 3, 3))
        self_patch_weight[..., 1, 1] = 0.0
        self.register_buffer('self_patch_weight', self_patch_weight) # self.self_patch_weight

        self_patch_sum_weight = torch.ones(size=(1, 1, 30, 30))
        self_patch_sum_weight = F.conv2d(self_patch_sum_weight, self_patch_weight, padding='same')
        self.register_buffer('self_patch_sum_weight', self_patch_sum_weight) # self.self_patch_sum_weight

        logger.info("Semi supervised learning information: ")
        logger.info(f"Loss type: {loss_type}")
        logger.info(f"Threshold: {unlabeled_threshold}")
        logger.info(f"Teacher Temperature: {teacher_temperature}")
        logger.info(f"Studnet Temperature: {student_temperature}")
        logger.info(f"Alpha: {alpha}")

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.token_head_type in {'dict', 'freeze'} and not self.token_head_initialized:
            self.token_head.layers[-1].weight.data.copy_(model.encoder.embed_tokens.weight.clone())
            self.token_head_initialized = True
            if self.token_head_type == 'freeze':
                for param in self.token_head.layers[-1].parameters():
                    param.requires_grad_(False)
            logger.info("lazy initializing token_head using model embeddings.")

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

        qa_output = model(**sample["aux_input"], encoder_only=True)
        net_output = model(**sample["net_input"])
        labeled_loss, unlabeled_loss, batch_size = self.compute_semi(model, qa_output, sample, update_num, reduce=reduce)
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )

        labeled_loss = labeled_loss * sample_size
        unlabeled_loss = unlabeled_loss * sample_size
        semisup_loss = (1.0 - self.alpha) * labeled_loss + self.alpha * unlabeled_loss

        loss = loss + semisup_loss * self.loss_weight

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "labeled_loss": labeled_loss.data,
            "unlabeled_loss": unlabeled_loss.data,
            "semisup_loss": semisup_loss.data,
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

    def compute_semi(self, model, qa_output, sample, update_num, reduce=True):

        encoder_returns = qa_output        
        encoder_out = encoder_returns["encoder_out"][0]
        encoder_embedding = encoder_returns["encoder_embedding"][0]
        # TODO: detect & erase unused embedding indices?

        net_input = sample.get("aux_input")
        src_token_id = net_input.get("src_tokens")

        mask = src_token_id > 2
        word_tokens_id = src_token_id[mask]

        _, B, D = encoder_out.shape
        batch_size = B

        text_encoding = encoder_out[900:] # 'T B D'
        text_encoding = text_encoding.transpose(0, 1)

        if self.center_type == 'sentence':
            text_logit = self.token_head(self.token_ln(text_encoding))
            text_logit[~mask] = 0.0
            sentence_center = text_logit.sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze(-1)
            text_logit = text_logit[mask]
        elif self.center_type == 'text':
            word_encoding = text_encoding[mask]
            text_logit = self.centering(self.token_head(self.token_ln(word_encoding)))
        else:
            word_encoding = text_encoding[mask]
            text_logit = self.token_head(self.token_ln(word_encoding))
        labeled_loss = F.cross_entropy(text_logit, word_tokens_id.detach(), reduction='mean')

        img_encoding = encoder_out[:900]
        v_word_encoding = img_encoding.transpose(0, 1).reshape(B*900, -1)
        if self.center_type == 'image':
            v_word_logit = self.centering(self.token_head(self.token_ln(v_word_encoding)))
        else:
            v_word_logit = self.token_head(self.token_ln(v_word_encoding))

        student_logit = v_word_logit / self.student_temperature
        with torch.no_grad():
            if self.use_self_patch:
                D = v_word_logit.size(-1)
                v_word_logit = rearrange(v_word_logit, '(b h w) d -> (b d) () h w', b=B, h=30, w=30)
                v_word_logit = F.conv2d(v_word_logit, self.self_patch_weight, padding='same')
                v_word_logit = v_word_logit / self.self_patch_sum_weight                
                v_word_logit = rearrange(v_word_logit, '(b d) () h w -> b (h w) d', b=B, d=D)

            if self.center_type == 'sentence':
                teacher_logit = (v_word_logit.reshape(B, 900, -1) - sentence_center.detach()) / self.teacher_temperature
                teacher_logit = teacher_logit.reshape(B*900, -1)
            else:
                teacher_logit = (v_word_logit - self.centering.center) / self.teacher_temperature
            v_word_pred = F.softmax(teacher_logit, dim=-1)
            max_value, max_index = v_word_pred.max(1)
            mask = (max_value >= self.unlabeled_threshold)

        if self.loss_type == 'cross_entropy':
            target = max_index.detach()
        elif self.loss_type == 'kld':
            target = v_word_pred.detach()
        else:
            raise NotImplementedError("")

        unlabeled_loss = (F.cross_entropy(student_logit, target, reduction='none') * mask).mean()
        
        return labeled_loss, unlabeled_loss, batch_size


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

        return loss, nll_loss, ntokens

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
        labeled_loss_sum = sum(log.get("labeled_loss", 0) for log in logging_outputs)
        unlabeled_loss_sum = sum(log.get("unlabeled_loss", 0) for log in logging_outputs)
        semisup_loss_sum = sum(log.get("semisup_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        # sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        # metrics.log_scalar(
        #     "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        # )
        # metrics.log_scalar(
        #     "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        # )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "labeled_loss", labeled_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "unlabeled_loss", unlabeled_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "semisup_loss", semisup_loss_sum / sample_size, ntokens, round=3
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
        # metrics.log_scalar(
        #     "sample_size_v1", sample_size_v1, 1, round=3
        # )
        # metrics.log_scalar(
        #     "sample_size_v2", sample_size_v2, 1, round=3
        # )

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
