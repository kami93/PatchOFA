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
from omegaconf import II

from .selfpatch import PatchAggregationHead, DINOHead, DINOLogit, HeadMlp

logger = logging.getLogger(__name__)

@dataclass
class CustomCriterionV2Config(FairseqDataclass):
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
    token_head_type: str = field(
        default='linear_copy_dictionary', metadata={"help": "layer type for token_head_type: linear_copy_dictionary | linear_random | mlp"}
    )
    dictionary_size: int = field(
        default=59457, metadata={"help": "dictionary_size for token_head"}
    )
    embed_dim: int = field(
        default=256, metadata={"help": "embedding dimesion for token_head"}
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
    no_text_loss: bool = field(
        default=False, metadata={"help": "do not use text_loss"}
    )
    # dict_dim: int = field(
    #     default=256, metadata={"help": "input dimesion for dict_head note: (59457, 256) for OFA-tiny dictionary"}
    # )
    # out_dim: int = field(
    #     default=4096, metadata={"help": "output dimesion for token_head and dict_head"}
    # )
    # token_warmup_temp: float = field(
    #     default=0.04, metadata={"help": "warmup temp"}
    # )
    # token_temp: float = field(
    #     default=0.04, metadata={"help": "temp"}
    # )
    # token_warmup_temp_iters: int = field(
    #     default=0, metadata={"help": "warmup_temp_iters"}
    # )

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
    "custom_criterion_v2", dataclass=CustomCriterionV2Config
)
class CustomCriterionV2(FairseqCriterion):
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
        token_head_type='linear_copy_dictionary',
        dictionary_size=59457,
        embed_dim=256,
        patch_aggregation_type='attention_text_token',
        aggregation_num_heads=4,
        aggregation_cosim_p=1.0,
        no_text_loss=False,
        # dict_head_type='linear',
        # dict_dim=4096,
        # out_dim=4096,
        # token_warmup_temp=0.04,
        # token_temp=0.04,
        # token_warmup_temp_iters=0
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
        self.use_text_loss = not no_text_loss

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        self.token_head_type = token_head_type
        if token_head_type == 'linear_copy_dictionary':
            self.token_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)
            # self.img_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)
            # self.text_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)
            self.token_head_initialized = False # lazy initialization using model embedding in the first iteration.

        elif token_head_type == 'linear_random':
            self.token_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)
            self.token_head_initialized = True
            # self.img_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)
            # self.text_head = nn.Linear(in_features=embed_dim, out_features=dictionary_size)

        elif token_head_type == 'mlp':
            self.token_head = HeadMlp(in_features=embed_dim, hidden_features=embed_dim, out_features=dictionary_size)
            # self.img_head = Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=dictionary_size)
            # self.text_head = Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=dictionary_size)
            self.token_head_initialized = True
        
        else:
            raise NotImplementedError(f"token_head_type {token_head_type} is not implemented.")

        self.patch_aggregation_type = patch_aggregation_type
        if patch_aggregation_type == 'attention_text_token':
            self.aggregation = PatchAggregationHead(embed_dim, aggregation_num_heads, use_cls=False, cosim_p=aggregation_cosim_p)
        elif patch_aggregation_type == 'attention_cls_token':
            self.aggregation = PatchAggregationHead(embed_dim, aggregation_num_heads, use_cls=True, cosim_p=aggregation_cosim_p)
        elif patch_aggregation_type in {'ignore', 'cossim_top'}:
            self.aggregation = None
        else:
            raise NotImplementedError(f"patch_aggregation_type {patch_aggregation_type} is not implemented.")

        self.use_noun = True
        self.use_object = True

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.token_head_type == 'linear_copy_dictionary' and not self.token_head_initialized:
            self.token_head.weight.data.copy_(model.encoder.embed_tokens.weight.clone())
            self.token_head_initialized = True
            logger.info("lazy initlaized token_head using model embeddings.")

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
        loss, nll_loss, img_loss, text_loss, ntokens, batch_size, patch_token_size = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        img_loss = img_loss * sample_size
        text_loss = text_loss * sample_size

        loss = loss + img_loss + text_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "img_loss": img_loss.data,
            "text_loss": text_loss.data,
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

        image_encoding = encoder_out[:900]
        image_encoding = image_encoding.transpose(0, 1) # '(H W) B D -> B (H W) D'
        text_encoding = encoder_out[900:] # 'T B D'
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

        if L:
            text_idx = torch.cat(text_idx, dim=0)
            text_batch_idx = torch.cat(text_batch_idx, dim=0)
            text_patch_mask = torch.cat(text_patch_mask, dim=0)

            token_ids = src_tokens[text_batch_idx, text_idx]
            text_tokens = text_encoding[text_batch_idx, text_idx]

            if self.use_text_loss:
                text_logit = self.token_head(text_tokens)
                text_loss = F.cross_entropy(text_logit, token_ids, reduction='mean')
            else:
                text_loss = torch.zeros(size=(1, ), device=encoder_out.device)

            patch_encoding = None
            if self.patch_aggregation_type == 'attention_text_token':
                patch_encoding = self.aggregation(image_encoding,
                                                  aggr_seed=text_tokens,
                                                  batch_idx=text_batch_idx,
                                                  patch_mask=text_patch_mask)
            elif self.patch_aggregation_type == 'attention_cls_token':
                patch_encoding = self.aggregation(image_encoding,
                                                  aggr_seed=text_tokens,
                                                  batch_idx=text_batch_idx,
                                                  patch_mask=text_patch_mask)
            
            elif self.patch_aggregation_type == 'cossim_top':
                with torch.no_grad():
                    img_norm = nn.functional.normalize(image_encoding, dim=-1)
                    img_norm = img_norm[text_batch_idx]

                    text_norm = nn.functional.normalize(text_tokens, dim=-1)

                    sim_matrix = torch.einsum("ld,lpd->lp", text_norm, img_norm)
                    sim_matrix += text_patch_mask

                    _, top_idx = sim_matrix.topk(k=1, dim=-1)
                patch_encoding = image_encoding[text_batch_idx.unsqueeze(-1).expand(-1, 1), top_idx].squeeze(1)

            if patch_encoding is not None:
                img_logit = self.token_head(patch_encoding)
                img_loss = F.cross_entropy(img_logit, token_ids, reduction='mean')
                patch_token_size = img_logit.size(0)
            else:
                img_loss = torch.zeros(size=(1, ), device=encoder_out.device)
                patch_token_size = 0.0

        else:
            text_loss = torch.zeros(size=(1, ), device=encoder_out.device)
            img_loss = torch.zeros(size=(1, ), device=encoder_out.device)
            patch_token_size = 0.0

        return loss, nll_loss, img_loss, text_loss, ntokens, batch_size, patch_token_size

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
        img_loss_sum = sum(log.get("img_loss", 0) for log in logging_outputs)
        text_loss_sum = sum(log.get("text_loss", 0) for log in logging_outputs)
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
            "img_loss", img_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_scalar(
            "text_loss", text_loss_sum / sample_size, ntokens, round=3
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
