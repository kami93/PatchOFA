# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO
from copy import deepcopy

import logging
import warnings
import string
import cv2
import random

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

CLASSES_ADE = np.array([
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
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
    'clock', 'flag', 'unknown'])

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
    'window-blind', 'window-other', 'wood', 'unknown'])

CLASSES_COCOC = np.array([
    'electronic', 'appliance', 'food-things', 'furniture-things', 'indoor', 
    'kitchen', 'accessory', 'animal', 'outdoor', 'person', 
    'sports', 'vehicle', 'ceiling', 'floor', 'food-stuff', 
    'furniture-stuff', 'raw material', 'textile', 'wall', 'window', 
    'building', 'ground', 'plant', 'sky', 'solid', 
    'structural', 'water', 'unknown'])

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

    labeled_input = None
    target_labeled = downsampled_target_labeled = None
    if samples[0].get("source_labeled", None) is not None:
        src_tokens_labeled = merge("source_labeled")
        src_lengths_labeled = torch.LongTensor([s["source_labeled"].ne(pad_idx).long().sum() for s in samples])

        patch_images_labeled = torch.stack([sample['patch_image_labeled'] for sample in samples], dim=0)
        patch_masks_labeled = torch.cat([sample['patch_mask_labeled'] for sample in samples])

        prev_output_tokens_labeled = merge("prev_output_tokens_labeled")
        
        target_labeled = merge("target_labeled")
        downsampled_target_labeled = merge("downsampled_target_labeled")
        
        fake_image_tokens = merge("fake_image_tokens")
        fake_image_token_offsets = torch.cat([sample['fake_image_token_offsets'] for sample in samples])
        
        labeled_input = {
            "src_tokens": src_tokens_labeled,
            "src_lengths": src_lengths_labeled,
            "patch_images": patch_images_labeled,
            "patch_masks": patch_masks_labeled,
            "prev_output_tokens": prev_output_tokens_labeled,
            "fake_image_tokens": fake_image_tokens,
            "fake_image_token_offsets": fake_image_token_offsets,
        }
        
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
        "labeled_input": labeled_input,
        "target": target,
        "downsampled_target": downsampled_target,
        "target_labeled": target_labeled,
        "downsampled_target_labeled": downsampled_target_labeled
        
    }

    return batch


class SegmentationSemiDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        dataset_labeled,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_object_length=30,
        max_tgt_length=30,
        patch_image_size=224,
        add_object=False,
        imagenet_default_mean_and_std=True,
        cfg=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.cfg = cfg

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
            
        self.prompt = self.encode_text(' what is the segmentation of the image?')
        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")
        
        self.prompt_type = cfg.prompt_type
        self.labeled_prompt_type = cfg.labeled_prompt_type
        self.num_seg = cfg.num_seg_tokens # 150 (ade) 171 (coco-fine) 27 (coco-coarse)
        if self.num_seg == 171:
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_COCOF]
        elif self.num_seg == 27:
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_COCOC]
        elif self.num_seg == 150:
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_ADE]
        else:
            raise NotImplementedError
        self.text_length = torch.tensor([len(x) for x in self.id2text])

        self.id2seg = np.array([f'<seg_{idx}>' for idx in range(self.num_seg + 1)])
        self.seg2code = self.encode_text(" ".join(self.id2seg), use_bpe=False)

        self.labeled_num_samples = cfg.labeled_num_samples
        self.dataset_labeled = dataset_labeled

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        line = [self.bpe.encode(' {}'.format(word.strip())) if not word.startswith('<seg_') else word for word in text.strip().split()]
        line = ' '.join(line)
        
        s = self.tgt_dict.encode_line(
            line=line,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def _preprocess_image(self, image, segmentation):
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        image_arr = np.asarray(image)
        if len(image_arr.shape) < 3:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
        
        image_arr = image_arr[:, :, ::-1].copy() # to BGR
        
        segmentation = Image.open(BytesIO(base64.urlsafe_b64decode(segmentation)))
        segmentation_arr = np.asarray(segmentation).copy()

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
        segmentation_arr[segmentation_arr == 254] = self.num_seg
        results['gt_semantic_seg'] = segmentation_arr
        results['seg_fields'] = ['gt_semantic_seg']
        
        return results

    def __getitem__(self, index):
        image, segmentation, uniq_id = self.dataset[index]
        results = self._preprocess_image(image, segmentation)
        patch_mask = torch.tensor([True])
        
        if self.split == 'train':
            aug_dict = self.image_transform(results)
            
            img = aug_dict.pop('img')
            img = img[:, :, ::-1].copy() # to RGB
            img = self.image_normalize(img)
            # test = img * torch.tensor(IMAGENET_DEFAULT_STD).reshape(3, 1, 1) + torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(3, 1, 1)
            
            gt_semantic_seg = aug_dict.pop('gt_semantic_seg')
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))

            image_labeled, segmentation_labeled, uniq_id_labeled = self.dataset_labeled[index]            
            results_labeled = self._preprocess_image(image_labeled, segmentation_labeled)
            aug_dict_labeled = self.image_transform(results_labeled)
            
            img_labeled = aug_dict_labeled.pop('img')
            img_labeled = img_labeled[:, :, ::-1].copy() # to RGB
            img_labeled = self.image_normalize(img_labeled)
            # test = img * torch.tensor(IMAGENET_DEFAULT_STD).reshape(3, 1, 1) + torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(3, 1, 1)
            
            gt_semantic_seg_labeled = aug_dict_labeled.pop('gt_semantic_seg')
            gt_semantic_seg_labeled = torch.from_numpy(gt_semantic_seg_labeled.astype(np.int64))
            
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

        text_target = None

        gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
        seg_ids = self.seg2code[gt_semantic_seg.flatten()]
        seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
        
        prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])
        downsampled_target = torch.cat([seg_ids_downsampled, self.eos_item])
        target = torch.cat([seg_ids, self.eos_item])

        if self.split == 'train':
            gt_semantic_seg_downsampled_labeled = self.downsample_gt_seg(gt_semantic_seg_labeled.unsqueeze(0)).flatten()
            seg_ids_labeled = self.seg2code[gt_semantic_seg_labeled.flatten()]
            seg_ids_downsampled_labeled = self.seg2code[gt_semantic_seg_downsampled_labeled]
            
            prev_output_item_labeled = torch.cat([self.bos_item, seg_ids_downsampled_labeled])
            downsampled_target_labeled = torch.cat([seg_ids_downsampled_labeled, self.eos_item])
            target_labeled = torch.cat([seg_ids_labeled, self.eos_item])

        # build 
        if self.prompt_type == 'gt_seg':
            if self.split == 'train':
                unique_seg_ids = gt_semantic_seg_downsampled.unique()
                randperm = torch.randperm(len(unique_seg_ids))
                unique_seg_ids = unique_seg_ids[randperm]
                src_text = torch.cat([self.id2text[idx] for idx in unique_seg_ids])
                src_item = torch.cat([self.bos_item, src_text, self.eos_item])
                
            else:
                # self.prompt_type is 'all' during validation.
                src_text = torch.cat([self.id2text[idx] for idx in range(self.num_seg)])
                src_item = torch.cat([self.bos_item, src_text, self.eos_item])
        
        elif self.prompt_type == 'prompt':
            src_item = torch.cat([self.bos_item, self.prompt, self.eos_item])
         
        elif self.prompt_type == 'all':
            src_text = torch.cat([self.id2text[idx] for idx in range(self.num_seg)])
            src_item = torch.cat([self.bos_item, src_text, self.eos_item])

        elif self.prompt_type == 'seg':
            src_text = self.seg2code[:self.num_seg]
            src_item = torch.cat([self.bos_item, src_text, self.eos_item])

        else:
            raise NotImplementedError

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": img,
            "patch_mask": patch_mask,
            "target": target,
            "text_target": text_target,
            "downsampled_target": downsampled_target,
            "prev_output_tokens": prev_output_item
        }

        if self.split == 'train':
            if self.labeled_prompt_type == 'all':
                src_text_labeled = torch.cat([self.id2text[idx] for idx in range(self.num_seg)])
                src_item_labeled = torch.cat([self.bos_item, src_text_labeled, self.eos_item])

            elif self.labeled_prompt_type == 'gt_seg':
                unique_seg_ids_labeled = gt_semantic_seg_downsampled_labeled.unique()
                randperm = torch.randperm(len(unique_seg_ids_labeled))
                unique_seg_ids_labeled = unique_seg_ids_labeled[randperm]
                    
                src_text_labeled = torch.cat([self.id2text[idx] for idx in unique_seg_ids_labeled])
                src_item_labeled = torch.cat([self.bos_item, src_text_labeled, self.eos_item])
            
            else:
                raise NotImplementedError
            
            example["source_labeled"] = src_item_labeled
            example["patch_image_labeled"] = img_labeled
            example["patch_mask_labeled"] = patch_mask
            example["target_labeled"] = target_labeled
            example["downsampled_target_labeled"] = downsampled_target_labeled
            example["prev_output_tokens_labeled"] = prev_output_item_labeled
            
            fake_image_tokens = torch.cat([self.id2text[idx] for idx in gt_semantic_seg_downsampled])
            fake_image_token_offsets = torch.tensor([self.text_length[idx] for idx in gt_semantic_seg_downsampled], dtype=torch.long)

            example["fake_image_tokens"] = fake_image_tokens
            example["fake_image_token_offsets"] = fake_image_token_offsets.cumsum(dim=0)
            
        
        return example

    def get_src_item_given_pair(self, pair):
        np.random.shuffle(pair)
        src_item = self.encode_text(
            ' does text1 " {} " and text2 " {} " have the same semantics?'.format(pair[0], pair[1]),
        )
        return src_item

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
