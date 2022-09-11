# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from fairseq import search
from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.ref_vqa_dataset import RefVQADataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class RefVQAConfig(OFAConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )    


@register_task("refvqa", dataclass=RefVQAConfig)
class RefVQATask(OFATask):
    def __init__(self, cfg: RefVQAConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = RefVQADataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            max_object_length=self.cfg.max_object_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.sequence_generator = self.build_generator(
            [model],
            Namespace()
            )
            
        # gen_args = json.loads(self.cfg.eval_args)
        # self.sequence_generator = self.build_generator(
        #         [model], Namespace(**gen_args)
        # )
        # if self.cfg.eval_acc:
        #     gen_args = json.loads(self.cfg.eval_args)
        #     self.sequence_generator = self.build_generator(
        #         [model], Namespace(**gen_args)
        #     )
        # if self.cfg.scst:
        #     scst_args = json.loads(self.cfg.scst_args)
        #     self.scst_generator = self.build_generator(
        #         [model], Namespace(**scst_args)
        #     )

        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def generation_step(self, sample, model):
        model.eval()

        hyps, _ = self._inference(self.sequence_generator, sample, model)
        hyps = hyps / (self.cfg.num_bins - 1) * self.cfg.max_image_size

        hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

        return hyps

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)

        hyps = []
        for i in range(len(gen_out)):
            hyps.append(gen_out[i][0]["tokens"][:-1] - len(self.src_dict) + self.cfg.num_bins)

        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps[0])

        return torch.stack(hyps, dim=0)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def build_generator(
        self, models, args
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
        """
        from models.sequence_generator_custom import SequenceGeneratorCustom

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = None
        
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                            self.target_dictionary, sampling_topk, sampling_topp
                        )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        seq_gen_cls = SequenceGeneratorCustom

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            constraint_range=self.cfg.constraint_range
        )