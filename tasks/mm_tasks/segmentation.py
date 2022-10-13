# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict
import torch
import os

from einops import rearrange
from mmseg.ops import resize

import numpy as np
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.data import Dictionary

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.segmentation_dataset import SegmentationDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig(OFAConfig):
    eval_acc: bool = field(
        default=True, metadata={"help": "evaluation with accuracy"}
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
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )    
    valid_batch_size: int = field(
        default=1,
        metadata={"help": "valid batch size per step"},
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
    fakeimage_type: str = field(
        default='random',
        metadata={
            "help": 'random | gt_seg'
        },
    )
    prompt_type: str = field(
        default='prompt',
        metadata={
            "help": 'prompt | all_seg | gt_seg'
        },
    )

@register_task("segmentation", dataclass=SegmentationConfig)
class SegmentationTask(OFATask):
    def __init__(self, cfg: SegmentationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.uses_ema = self.cfg.uses_ema

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))

        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        num_segs = 151
        for i in range(num_segs):
            src_dict.add_symbol("<seg_{}>".format(i))
            tgt_dict.add_symbol("<seg_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        
        assert self.cfg.selected_cols == '0,1,2'
        dataset = FileDataset(table_path, self.cfg.selected_cols)

        self.datasets[split] = SegmentationDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_object_length=self.cfg.max_object_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_object=self.cfg.add_object,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            cfg=self.cfg,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

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

    def valid_step(self, sample, model, criterion):
        model.eval()
        loss, sample_size, logging_output = criterion(model, sample)

        # hyps, refs = self._inference(self.sequence_generator, sample, model)

        # pred_label = hyps
        # label = refs - criterion.seg_id_offset
        # num_classes = 151

        # mask = (label != 150)
        # pred_label = pred_label[mask]
        # label = label[mask]

        # intersect = pred_label[pred_label == label]
        # area_intersect = torch.histc(
        #     intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
        # area_pred_label = torch.histc(
        #     pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        # area_label = torch.histc(
        #     label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        # area_union = area_pred_label + area_label - area_intersect

        # logging_output["_area_intersect_infer"] = area_intersect
        # logging_output["_area_pred_label_infer"] = area_pred_label
        # logging_output["_area_label_infer"] = area_label
        # logging_output["_area_union_infer"] = area_union

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_all_acc(meters):
            all_acc = meters['_area_intersect_infer'].sum.sum() / meters['_area_pred_label_infer'].sum.sum()
            all_acc = all_acc if isinstance(all_acc, float) else all_acc.item()
            return round(all_acc, 4)

        def compute_mean_iou(meters):
            miou = torch.nanmean(meters['_area_intersect_infer'].sum / (meters['_area_union_infer'].sum))
            miou = miou if isinstance(miou, float) else miou.item()
            return round(miou, 4)

        def compute_mean_acc(meters):
            macc = torch.nanmean(meters['_area_intersect_infer'].sum / (meters['_area_label_infer'].sum))
            macc = macc if isinstance(macc, float) else macc.item()
            return round(macc, 4)

        if "_area_union_infer" in logging_outputs[0]: # check if valid
            metrics.log_scalar_sum("_area_intersect_infer", sum_logs("_area_intersect_infer"))
            metrics.log_scalar_sum("_area_pred_label_infer", sum_logs("_area_pred_label_infer"))
            metrics.log_scalar_sum("_area_label_infer", sum_logs("_area_label_infer"))
            metrics.log_scalar_sum("_area_union_infer", sum_logs("_area_union_infer"))

            metrics.log_derived("infer_aAcc", compute_all_acc)
            metrics.log_derived("infer_mIoU", compute_mean_iou)
            metrics.log_derived("infer_mAcc", compute_mean_acc)

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        
        hyps = gen_out
        refs = sample["downsampled_target"][:, :-1]
        
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps)
            logger.info("example reference: " + refs)

        return hyps, refs
