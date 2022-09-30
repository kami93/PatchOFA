# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string
import cv2

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from mmseg.datasets.pipelines import Resize, RandomCrop, RandomFlip, PhotoMetricDistortion, MultiScaleFlipAug, Normalize, ImageToTensor

from torchvision.utils import save_image

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")

        if samples[0].get("downsampled_target", None) is not None:
            downsampled_target = merge("downsampled_target")

    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "downsampled_target": downsampled_target
    }

    return batch


class SegmentationDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_object_length=30,
        max_tgt_length=30,
        patch_image_size=224,
        add_object=False,
        imagenet_default_mean_and_std=True
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.image_normalize = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std)])
        if self.split == 'train':
            self.image_transform = transforms.Compose([
                Resize(img_scale=(2048, 512), ratio_range=(0.5, 2.0), min_size=512),
                RandomCrop(crop_size=(512, 512), cat_max_ratio=0.75),
                RandomFlip(prob=0.5),
                PhotoMetricDistortion(),
            ])
            
            self.downsample_gt_seg = transforms.Resize((32, 32), transforms.InterpolationMode.NEAREST)


        else:
            # self.multiscale_transform = MultiScaleFlipAug(img_scale=(2048, 512),
            #                                               flip=False,
            #                                               transforms=[dict(type='Resize', keep_ratio=True),
            #                                                           dict(type='RandomFlip')])
            
            self.image_transform = Resize(img_scale=(512, 512), keep_ratio=False)
            self.downsample_gt_seg = transforms.Resize(32, transforms.InterpolationMode.NEAREST)
            
        self.prompt = ' what is the segmentation of the image?'

        id2seg = " ".join([f'<seg_{idx}>' for idx in range(151)])
        self.seg2code = self.encode_text(id2seg, use_bpe=False)

    def __getitem__(self, index):
        image, segmentation, uniq_id = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        image_arr = np.asarray(image)
        if len(image_arr.shape) < 3:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
        
        image_arr = image_arr[:, :, ::-1].copy() # to BGR
        
        segmentation = Image.open(BytesIO(base64.urlsafe_b64decode(segmentation)))
        segmentation_arr = np.asarray(segmentation)
        
        patch_mask = torch.tensor([True])

        results = {}
        # results['filename'] = filename
        # results['ori_filename'] = results['img_info']['filename']
        results['img'] = image_arr
        results['img_shape'] = image_arr.shape
        # results['ori_shape'] = image_arr.shape
        # Set initial values for default meta_keys
        # results['pad_shape'] = image_arr.shape
        results['scale_factor'] = 1.0
        # num_channels = 1 if len(image_arr.shape) < 3 else image_arr.shape[2]
        # results['img_norm_cfg'] = dict(
        #     mean=np.zeros(num_channels, dtype=np.float32),
        #     std=np.ones(num_channels, dtype=np.float32),
        #     to_rgb=False)

        # avoid using underflow conversion
        segmentation_arr[segmentation_arr == 0] = 255
        segmentation_arr = segmentation_arr - 1
        segmentation_arr[segmentation_arr == 254] = 150
        results['gt_semantic_seg'] = segmentation_arr
        results['seg_fields'] = ['gt_semantic_seg']

        if self.split == 'train':
            aug_dict = self.image_transform(results)
            
            img = aug_dict.pop('img')
            img = img[:, :, ::-1].copy() # to RGB
            img = self.image_normalize(img)
            # test = img * torch.tensor(IMAGENET_DEFAULT_STD).reshape(3, 1, 1) + torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(3, 1, 1)
            
            gt_semantic_seg = aug_dict.pop('gt_semantic_seg')
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))
            
        else:
            # multiscale_images = self.multiscale_transform(results)
            
            # img_list = multiscale_images.pop('img')
            # img = img_list[0]
            img_dict = self.image_transform(results)
            img = img_dict.pop('img')
            
            img = img[:, :, ::-1].copy() # to RGB
            img = self.image_normalize(img)

            # gt_semantic_seg_list = multiscale_images.pop('gt_semantic_seg')
            # gt_semantic_seg = gt_semantic_seg_list[0]
            
            gt_semantic_seg = img_dict.pop('gt_semantic_seg')
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))

        h, w = gt_semantic_seg.shape[:2]
        if h < 512 or w < 512:
            import pdb; pdb.set_trace()
            abc = 1

        src_item = self.encode_text(self.prompt)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0))
        
        seg_ids = self.seg2code[gt_semantic_seg.flatten()]
        seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled.flatten()]
        
        prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])
        
        downsampled_target = torch.cat([seg_ids_downsampled, self.eos_item])
        target = torch.cat([seg_ids, self.eos_item])
            
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": img,
            "patch_mask": patch_mask,
            "target": target,
            "downsampled_target": downsampled_target,
            "prev_output_tokens": prev_output_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
