#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import os
import os.path as osp
import sys
from multiprocessing import Pool
import pickle as pkl

import numpy as np
import torch
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

import cv2

from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results
from utils.zero_shot_utils import zero_shot_step

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def main(cfg: DictConfig, **kwargs):
    pool = Pool(20)

    utils.import_user_module(cfg.common)

    reset_logging()
    logger.info(cfg)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    results = []
    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()
    for sample in progress:
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
        with torch.no_grad():
            if kwargs["zero_shot"]:
                result = zero_shot_step(task, generator, models, sample)
            else:
                result = eval_ref(task, generator, models, sample, **kwargs)
        
            words_batch = sample['words_batch']
            raw_information = sample['raw_information']

            num_words = 0
            for idx in range(len(result)):
                boxes = result[idx]['box']
                words = words_batch[idx]
                image = raw_information[idx]['image']
                uniq_id = raw_information[idx]['uniq_id']
                line_id = raw_information[idx]['line_id']

                num_words += len(words)
                # img = np.asarray(image)
                
                output_dict = {}
                for word_idx in range(len(words)):
                    # _img = img.copy()
                    coord_list = boxes[word_idx]
                    word = words[word_idx]

                    output_dict[word] = coord_list

                output_dir = osp.join(
                        '/input/nfs-3090-s00/shpark/cache/OFA/vqa_ref_ofa_large',
                        'train',
                        f'{uniq_id}_{line_id}.pkl')
                pool.apply_async(save_pickle, (output_dict, output_dir))

                # cv2.rectangle(
                #     _img,
                #     (int(coord_list[0]), int(coord_list[1])),
                #     (int(coord_list[2]), int(coord_list[3])),
                #     (0, 255, 0),
                #     3
                # )
                # cv2.imwrite(f'{uniq_id}_{word_idx:02d}_{word}.jpg', _img[..., ::-1])

    #     results += result
        progress.log({"sentences": num_words})

    # merge_results(task, cfg, logger, score_cnt, score_sum, results)

def eval_ref(task, generator, models, sample, **kwargs):
    
    gen_out = task.inference_step(generator, models, sample)

    hyps = []
    for i in range(len(gen_out)):
        hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
    hyps = torch.stack(hyps, dim=0)

    word_batch_idx = sample['net_input']['word_batch_idx']

    w_resize_ratios = sample['w_resize_ratios'][word_batch_idx]
    h_resize_ratios = sample['h_resize_ratios'][word_batch_idx]
    
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= w_resize_ratios.unsqueeze(1)
    hyps[:, 1::2] /= h_resize_ratios.unsqueeze(1)

    hyps_cpu = hyps.cpu().numpy().astype(np.float16)
    results = []
    offset = 0
    for i, (sample_id, num_samples) in enumerate(zip(sample["id"].tolist(), sample["num_samples_batch"].tolist())):
        hyps_i = hyps_cpu[offset:offset+num_samples]
        results.append(
            {"uniq_id": sample_id,
             "box": [hyps_i[j] for j in range(num_samples)]}
        )
        # [hyps_i[j][0].item(), hyps_i[j][1].item(), hyps_i[j][2].item(), hyps_i[j][3].item()] for j in range(num_samples)]
        offset += num_samples

    return results

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)

    distributed_utils.call_main(
        cfg, main, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval, zero_shot=args.zero_shot
    )


if __name__ == "__main__":
    cli_main()
