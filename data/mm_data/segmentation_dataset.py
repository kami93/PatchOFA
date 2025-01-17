# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

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

def flatten(l):
    return [item for sublist in l for item in sublist]

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
    'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling', 'ceiling tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk', 'dirt', 'door', 'fence', 'marble floor',
    'other floor', 'stone floor', 'tile floor', 'wood floor',
    'flower', 'fog', 'food', 'fruit', 'furniture', 'grass',
    'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat',
    'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper',
    'snow', 'solid', 'stairs', 'stone', 'straw', 'structural',
    'table', 'tent', 'textile', 'towel', 'tree', 'vegetable',
    'brick wall', 'concrete wall', 'other wall', 'panel wall',
    'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops',
    'blind window', 'other window', 'wood', 'unknown'])

CLASSES_COCOC = np.array([
    'electronic', 'appliance', 'food things', 'furniture things', 'indoor', 
    'kitchen', 'accessory', 'animal', 'outdoor', 'person', 
    'sports', 'vehicle', 'ceiling', 'floor', 'food stuff', 
    'furniture stuff', 'raw material', 'textile', 'wall', 'window', 
    'building', 'ground', 'plant', 'sky', 'solid', 
    'structural', 'water', 'unknown'])

CLASSES_COCO_SEEN = np.array(
    ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bear', 'zebra', 'giraffe', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 
    'building', 'bush', 'cabinet', 'cage', 'cardboard', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 
    'clouds', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor', 'other floor', 'stone floor', 
    'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'ground', 'hill', 
    'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 
    'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railroad', 'river', 'roof', 'rug', 
    'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural', 
    'table', 'tent', 'textile', 'towel', 'tree', 'brick wall', 'other wall', 'panal wall', 'stone wall', 
    'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window', 'other window', 'wood', 'unknown'])

CLASSES_COCO_UNSEEN = np.array([
    'frisbee', 'skateboard', 'cardboard', 'carrot', 'scissors', 
    'suitcase', 'giraffe', 'cow', 'road', 'concrete wall', 
    'tree', 'grass', 'river', 'clouds', 'playingfield', 'unknown'])

CLASSES_COCOC_AUGMENTED = [
    ['electronic', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    ['appliance', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender'],
    ['food', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    ['furniture', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door'],
    ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'],
    ['kitchen', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    ['accessory', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase'],
    ['animal', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    ['traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench'],
    ['person', 'man', 'woman', 'child', 'boy', 'girl'],
    ['sports', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    ['vehicle', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    ['ceiling', 'ceiling tile'],
    ['floor', 'carpet', 'marble flooring', 'stone flooring', 'tile flooring', 'wood flooring'],
    ['food', 'fruit', 'salad', 'vegetable'],
    ['furniture', 'cabinet', 'counter', 'cupboard', 'desk', 'door', 'light', 'mirror', 'shelf', 'stairs', 'table'],
    ['cardboard', 'metal', 'paper', 'plastic'],
    ['textile', 'banner', 'blanket', 'cloth', 'clothes', 'curtain', 'mat', 'napkin', 'pillow', 'rug', 'towel'],
    ['wall', 'brick wall', 'concrete wall', 'panel wall', 'stone wall', 'tile wall', 'wood wall'],
    ['window', 'blind window'],
    ['building', 'bridge', 'house', 'roof', 'skyscraper', 'tent'],
    ['ground', 'dirt', 'gravel', 'mud', 'pavement', 'platform', 'playingfield', 'railroad', 'road', 'sand', 'snow'],
    ['plant', 'branch', 'bush', 'flower', 'grass', 'leaves', 'moss', 'straw', 'tree'],
    ['sky', 'clouds'],
    ['hill', 'mountain', 'rock', 'stone', 'wood'],
    ['structural', 'cage', 'fence', 'net', 'railing'],
    ['water', 'fog', 'river', 'sea', 'ocean', 'waterdrops', 'lake']]

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

    if samples[0].get("ori_shape", None) is not None:
        ori_shape = [x["ori_shape"] for x in samples]
        
    if samples[0].get("ori_semantic_seg", None) is not None:
        ori_semantic_seg = [x["ori_semantic_seg"] for x in samples]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens = None
        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")

        downsampled_target = None
        if samples[0].get("downsampled_target", None) is not None:
            downsampled_target = merge("downsampled_target")

    else:
        ntokens = src_lengths.sum().item()

    aux_input = None
    text2seg_target = None
    if samples[0].get("text2seg_source", None) is not None:
        text2seg_src_tokens = merge("text2seg_source")
        text2seg_src_lengths = torch.LongTensor([s["text2seg_source"].ne(pad_idx).long().sum() for s in samples])

        if samples[0].get("text2seg_target", None) is not None:
            text2seg_target = merge("text2seg_target")

        if samples[0].get("text2seg_prev_output_tokens", None) is not None:
            text2seg_prev_output_tokens = merge("text2seg_prev_output_tokens")

        text2seg_patch_images = None
        text2seg_patch_masks = None
        if samples[0].get("text2seg_patch_image", None) is not None:
            text2seg_patch_images = merge("text2seg_patch_image")
            text2seg_patch_masks = torch.cat([sample['text2seg_patch_mask'] for sample in samples])
        
        aux_input = {
            "src_tokens": text2seg_src_tokens,
            "src_lengths": text2seg_src_lengths,
            "patch_images": text2seg_patch_images,
            "patch_masks": text2seg_patch_masks,
            "prev_output_tokens": text2seg_prev_output_tokens,
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
        "aux_input": aux_input,
        "target": target,
        "downsampled_target": downsampled_target,
        "text2seg_target": text2seg_target,
        "ori_shape": ori_shape,
        "ori_semantic_seg": ori_semantic_seg,
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
        imagenet_default_mean_and_std=True,
        cfg=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        logger.info(f"patch_image_size: {patch_image_size}")
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
                Resize(img_scale=(self.patch_image_size*4, self.patch_image_size), ratio_range=(0.5, 2.0), min_size=self.patch_image_size),
                RandomCrop(crop_size=(self.patch_image_size, self.patch_image_size), cat_max_ratio=0.75),
                RandomFlip(prob=0.5),
                PhotoMetricDistortion(),
            ])
            
            self.downsample_gt_seg = transforms.Resize((self.patch_image_size//16, self.patch_image_size//16), transforms.InterpolationMode.NEAREST)


        else:
            self.image_transform = MultiScaleFlipAug(img_scale=(self.patch_image_size*4, self.patch_image_size),
                                                          flip=False,
                                                          transforms=[dict(type='Resize', keep_ratio=True),
                                                                      dict(type='RandomFlip')])
            self.downsample_gt_seg = transforms.Resize((self.patch_image_size//16, self.patch_image_size//16), transforms.InterpolationMode.NEAREST)

        prompt_prefix=self.cfg.prompt_prefix
        if len(prompt_prefix):
            self.prompt = self.encode_text(f' {prompt_prefix.lstrip()}')
        else:
            self.prompt = None

        self.artificial_image_type = cfg.artificial_image_type
        self.num_seg = cfg.num_seg_tokens # 150 (ade) 171 (coco-fine) 27 (coco-coarse)
        if self.num_seg == 171:
            self.id2rawtext = [x for x in CLASSES_COCOF]
        elif self.num_seg == 27:
            # self.id2rawtext = CLASSES_COCOC_AUGMENTED
            # self.id2numtext = np.array([len(x) for x in CLASSES_COCOC_AUGMENTED])
            # self.id2offset = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(self.id2numtext)[:-1]])
            # self.id2truetext = [x for x in CLASSES_COCOC]
            self.id2rawtext = [x for x in CLASSES_COCOC]
        elif self.num_seg == 150:
            # self.id2rawtext = CLASSES_ADE_AUGMENTED
            # self.id2numtext = np.array([len(x) for x in CLASSES_ADE_AUGMENTED])
            # self.id2offset = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(self.id2numtext)[:-1]])
            # self.id2truetext = [x for x in CLASSES_ADE]
            self.id2rawtext = [x for x in CLASSES_ADE]
        elif self.num_seg == 15:
            self.id2rawtext = [x for x in CLASSES_COCO_UNSEEN]
        elif self.num_seg == 156:
            self.id2rawtext = [x for x in CLASSES_COCO_SEEN]
        elif self.num_seg == 165:
            self.id2rawtext = [x for x in CLASSES_COCO_UNSEEN]
            self.id2rawtext = self.id2rawtext + [x for x in CLASSES_ADE if x not in CLASSES_COCO_UNSEEN]
        else:
            raise NotImplementedError
       
        if isinstance(self.id2rawtext[0], list):
            self.id2text = [self.encode_text(f" {x}") for x in flatten(self.id2rawtext)]
            self.id2text_true = [self.encode_text(f" {x}") for x in self.id2truetext]
        else:
            self.id2text = [self.encode_text(f" {x}") for x in self.id2rawtext]
            self.id2text_true = None

        self.text_length = torch.tensor([len(x) for x in self.id2text])

        self.id2seg = np.array([f'<seg_{idx}>' for idx in range(self.num_seg + 1)])
        self.seg2code = self.encode_text(" ".join(self.id2seg), use_bpe=False)
        self.upsample_gt_seg = transforms.Resize((self.patch_image_size, self.patch_image_size), transforms.InterpolationMode.NEAREST)

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

    def __getitem__(self, index):
        image, segmentation, uniq_id = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        image_arr = np.asarray(image)
        if len(image_arr.shape) < 3:
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
        
        image_arr = image_arr[:, :, ::-1].copy() # to BGR
        
        segmentation = Image.open(BytesIO(base64.urlsafe_b64decode(segmentation)))
        segmentation_arr = np.asarray(segmentation).copy()
        
        patch_mask = torch.tensor([True])

        results = {}
        results['img'] = image_arr
        results['img_shape'] = image_arr.shape
        results['scale_factor'] = 1.0

        # avoid using underflow conversion
        segmentation_arr[segmentation_arr == 0] = 255
        segmentation_arr = segmentation_arr - 1
        segmentation_arr[segmentation_arr == 254] = self.num_seg
        results['gt_semantic_seg'] = segmentation_arr
        results['seg_fields'] = ['gt_semantic_seg']

        ori_shape = image_arr.shape
        ori_semantic_seg = segmentation_arr.copy()
        if self.split == 'train':
            aug_dict = self.image_transform(results)
            
            img = aug_dict.pop('img')
            img = img[:, :, ::-1].copy() # to RGB
            img = self.image_normalize(img)
            gt_semantic_seg = aug_dict.pop('gt_semantic_seg')
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))

            gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
            seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
            downsampled_target = torch.cat([seg_ids_downsampled, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])

        else:
            img_dict = self.image_transform(results)
            img = img_dict.pop('img')[0]
            img = img[:, :, ::-1].copy() # to RGB

            img = self.image_normalize(img)

            gt_semantic_seg = img_dict.pop('gt_semantic_seg')[0]
            gt_semantic_seg = torch.from_numpy(gt_semantic_seg.astype(np.int64))
            
            downsampled_target=None
            gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
            seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
            prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])

        seg_ids = self.seg2code[gt_semantic_seg.flatten()]
        target = torch.cat([seg_ids, self.eos_item])

        # build 
        if self.id2text_true is not None:
            prompt_ids = [idx for idx in range(len(self.id2text_true))]
        else:
            prompt_ids = [idx for idx in range(len(self.id2text))]
        prompt_ids.sort()
        
        src_text = [self.bos_item]
        if self.prompt is not None:
            src_text += [self.prompt]
        
        if self.id2text_true is not None:
            src_text += [self.id2text_true[idx] for idx in prompt_ids]
        else:
            src_text += [self.id2text[idx] for idx in prompt_ids]
        src_text += [self.eos_item]

        src_item = torch.cat(src_text)

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": img,
            "patch_mask": patch_mask,
            "target": target,
            "downsampled_target": downsampled_target,
            "prev_output_tokens": prev_output_item,
            "ori_shape": ori_shape,
            "ori_semantic_seg": ori_semantic_seg,
        }

        if self.artificial_image_type == 'none':
            # no fake image
            return example
            
        elif self.artificial_image_type == 'norand_k':
            artificial_image_ids = np.random.choice(self.num_seg, size=1024, replace=True).tolist()
            artificial_image_target = self.seg2code[artificial_image_ids]

        elif self.artificial_image_type.startswith('rand_k'):
            if self.artificial_image_type == 'rand_k':
                l, r = 1, 33
            elif len(self.artificial_image_type.split('-')) == 3:
                l, r = self.artificial_image_type.split('-')[1:3]
                l, r = int(l), int(r)
            else:
                raise NotImplementedError

            sh, sw = torch.randint(l,r,(2,))
            sh, sw = sh.item(), sw.item()
            rand = np.random.choice(self.num_seg, size=sh*sw, replace=True)
            if isinstance(self.id2rawtext[0], list):
                extended_rand = []
                for class_id in rand:
                    numtext = self.id2numtext[class_id]
                    randint = np.random.randint(numtext)

                    extended_class_id = self.id2offset[class_id] + randint
                    extended_rand.append(extended_class_id)
                extended_rand = np.array(extended_rand)
                extended_rand = torch.from_numpy(extended_rand).reshape(1, 1, sh, sw)
                artificial_image_ids = self.downsample_gt_seg(extended_rand).reshape(-1).tolist()
                rand = torch.from_numpy(rand).reshape(1, 1, sh, sw)
            
            else:
                rand = torch.from_numpy(rand).reshape(1, 1, sh, sw)
                artificial_image_ids = self.downsample_gt_seg(rand).reshape(-1).tolist()
            
            upsample_rand = self.upsample_gt_seg(rand).reshape(-1).tolist()
            downsample_rand = self.downsample_gt_seg(rand).reshape(-1).tolist()
            artificial_image_target = self.seg2code[upsample_rand]
            artificial_image_prev = self.seg2code[downsample_rand]
        else:
            raise NotImplementedError

        embedbag_ids = torch.cat([self.id2text[idx] for idx in artificial_image_ids])
        embedbag_offsets = torch.tensor([self.text_length[idx] for idx in artificial_image_ids], dtype=torch.long).cumsum(dim=0)

        target = torch.cat([artificial_image_target, self.eos_item])
        prev_output_tokens = torch.cat([self.bos_item, artificial_image_prev])

        if self.id2text_true is not None:
            prompt_ids = [idx for idx in range(len(self.id2text_true))]
        else:
            prompt_ids = [idx for idx in range(len(self.id2text))]

        prompt_ids.sort()
        
        src_text = [self.bos_item]
        if self.prompt is not None:
            src_text += [self.prompt]
        
        if self.id2text_true is not None:
            src_text += [self.id2text_true[idx] for idx in prompt_ids]
        else:
            src_text += [self.id2text[idx] for idx in prompt_ids]
        src_text += [self.eos_item]
        src_item = torch.cat(src_text)
        
        example["text2seg_patch_image"] = embedbag_ids
        example["text2seg_patch_mask"] = embedbag_offsets
        example["text2seg_source"] = src_item
        example["text2seg_target"] = target
        example["text2seg_prev_output_tokens"] = prev_output_tokens

        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)


CLASSES_ADE_AUGMENTED = [['wall',
  'pane',
  'wall socket',
  'wall plug',
  'electric outlet',
  'electrical outlet',
  'outlet',
  'electric receptacle',
  'plate',
  'sockets'],
 ['building',
  'edifice',
  'arcades',
  'balcony',
  'balustrade',
  'bars',
  'bell',
  'chimney',
  'column',
  'dome',
  'door',
  'door frame',
  'doors',
  'dormer',
  'double door',
  'entrance',
  'fire escape',
  'garage door',
  'gate',
  'grille',
  'metal shutter',
  'metal shutters',
  'pane',
  'pipe',
  'porch',
  'railing',
  'revolving door',
  'roof',
  'shop window',
  'shutter',
  'sign',
  'skylight',
  'statue',
  'steps',
  'terrace',
  'tower',
  'wall',
  'window',
  'windows'],
 ['sky', 'cloud', 'clouds'],
 ['floor', 'flooring'],
 ['tree', 'branch', 'fruit', 'trunk'],
 ['ceiling', 'beam'],
 ['road', 'route', 'crosswalk'],
 ['bed',
  'base',
  'bedpost',
  'bedspring',
  'drawer',
  'footboard',
  'headboard',
  'ladder',
  'leg',
  'rail',
  'safety rail',
  'side',
  'side rail'],
 ['windowpane',
  'window',
  'casing',
  'door',
  'handle',
  'interior casing',
  'lower sash',
  'muntin',
  'pane',
  'rail',
  'sash',
  'sash lock',
  'shutter',
  'sill',
  'stile',
  'upper sash',
  'window'],
 ['grass'],
 ['cabinet',
  'back',
  'base',
  'door',
  'drawer',
  'front',
  'leg',
  'panel',
  'shelf',
  'side',
  'skirt',
  'top'],
 ['sidewalk', 'pavement'],
 ['person',
  'individual',
  'someone',
  'somebody',
  'mortal',
  'soul',
  'back',
  'head',
  'left arm',
  'left foot',
  'left hand',
  'left leg',
  'left shoulder',
  'neck',
  'right arm',
  'right foot',
  'right hand',
  'right leg',
  'right shoulder',
  'torso'],
 ['earth', 'ground'],
 ['door',
  'door frame',
  'handle',
  'hinge',
  'knob',
  'lock',
  'muntin',
  'pane',
  'panel',
  'window',
  'doorframe',
  'doorcase',
  'double door',
  'door',
  'door frame',
  'handle',
  'pane'],
 ['table',
  'apron',
  'base',
  'door',
  'drawer',
  'front',
  'leg',
  'pedestal',
  'shelf',
  'side',
  'skirt',
  'stretcher',
  'top'],
 ['mountain', 'mount'],
 ['plant', 'flora', 'plant life', 'flower', 'leaf'],
 ['curtain', 'drape', 'drapery', 'mantle', 'pall'],
 ['chair',
  'apron',
  'arm',
  'back',
  'back pillow',
  'base',
  'foot rest',
  'h-stretcher',
  'leg',
  'seat',
  'seat base',
  'seat cushion',
  'skirt',
  'stretcher'],
 ['car',
  'auto',
  'automobile',
  'machine',
  'motorcar',
  'bumper',
  'door',
  'fender',
  'gas cap',
  'handle',
  'headlight',
  'hood',
  'license plate',
  'logo',
  'mirror',
  'roof rack',
  'taillight',
  'wheel',
  'window',
  'windshield',
  'wiper'],
 ['water'],
 ['painting', 'picture', 'frame'],
 ['sofa',
  'couch',
  'lounge',
  'apron',
  'arm',
  'back',
  'back pillow',
  'cushion',
  'leg',
  'seat',
  'seat base',
  'seat cushion',
  'skirt'],
 ['shelf', 'base', 'door', 'leg', 'shelf', 'side', 'top'],
 ['house',
  'balcony',
  'balustrade',
  'chimney',
  'column',
  'door',
  'dormer',
  'double door',
  'garage door',
  'pipe',
  'railing',
  'roof',
  'shutter',
  'steps',
  'window',
  'windows'],
 ['sea', 'wave'],
 ['mirror', 'frame'],
 ['rug', 'carpet', 'carpeting'],
 ['field', 'hay bale'],
 ['armchair',
  'apron',
  'arm',
  'back',
  'back pillow',
  'earmuffs',
  'leg',
  'seat',
  'seat base',
  'seat cushion',
  'skirt',
  'stretcher'],
 ['seat', 'back', 'back pillow', 'seat cushion'],
 ['fence', 'fencing', 'post', 'rail'],
 ['desk', 'door', 'drawer', 'leg', 'shelf', 'side', 'top'],
 ['rock', 'stone'],
 ['wardrobe',
  'closet',
  'press',
  'door',
  'drawer',
  'leg',
  'shelf',
  'side',
  'top'],
 ['lamp',
  'aperture',
  'arm',
  'base',
  'bulb',
  'canopy',
  'chain',
  'column',
  'cord',
  'shade',
  'tube'],
 ['bathtub', 'bathing tub', 'bath', 'tub', 'faucet', 'overflot plate', 'tap'],
 ['railing', 'rail'],
 ['cushion'],
 ['base', 'pedestal', 'stand'],
 ['box', 'tissue'],
 ['column', 'pillar', 'base', 'capital', 'shaft'],
 ['signboard', 'sign'],
 ['chest of drawers',
  'chest',
  'bureau',
  'dresser',
  'base',
  'door',
  'drawer',
  'front',
  'leg',
  'mirror',
  'side',
  'skirt',
  'top'],
 ['counter'],
 ['sand'],
 ['sink', 'bowl', 'faucet', 'pedestal', 'tap'],
 ['skyscraper', 'pane', 'window'],
 ['fireplace', 'hearth', 'open fireplace'],
 ['refrigerator', 'icebox', 'door', 'side'],
 ['grandstand'],
 ['path'],
 ['stairs', 'steps', 'step'],
 ['runway'],
 ['case'],
 ['pool table',
  'billiard table',
  'snooker table',
  'base',
  'bed',
  'cabinet',
  'corner pocket',
  'leg',
  'rail',
  'side pocket'],
 ['pillow'],
 ['screen door'],
 ['stairway', 'staircase', 'rung', 'step', 'stringer'],
 ['river'],
 ['bridge', 'span'],
 ['bookcase', 'door', 'front', 'shelf', 'top'],
 ['blind', 'screen', 'head rail', 'slats'],
 ['coffee table', 'cocktail table', 'apron', 'drawer', 'leg', 'shelf', 'top'],
 ['toilet',
  'can',
  'commode',
  'crapper',
  'pot',
  'potty',
  'stool',
  'throne',
  'bowl',
  'cistern',
  'lid'],
 ['flower'],
 ['book'],
 ['hill'],
 ['bench', 'leg'],
 ['countertop'],
 ['stove',
  'kitchen stove',
  'range',
  'kitchen range',
  'cooking stove',
  'burner',
  'button panel',
  'dial',
  'drawer',
  'oven',
  'stove'],
 ['palm', 'palm tree'],
 ['kitchen island'],
 ['computer',
  'computing machine',
  'computing device',
  'data processor',
  'electronic computer',
  'information processing system',
  'computer case',
  'keyboard',
  'monitor',
  'mouse',
  'speaker'],
 ['swivel chair', 'arm', 'armrest', 'back', 'base', 'piston', 'seat'],
 ['boat', 'window'],
 ['bar'],
 ['arcade machine'],
 ['hovel'],
 ['bus',
  'autobus',
  'coach',
  'charabanc',
  'double-decker',
  'jitney',
  'motorbus',
  'motorcoach',
  'omnibus',
  'passenger vehicle',
  'door',
  'headlight',
  'license plate',
  'mirror',
  'taillight',
  'wheel',
  'window',
  'windshield'],
 ['towel'],
 ['light',
  'light source',
  'aperture',
  'backplate',
  'bulb',
  'canopy',
  'diffusor',
  'shade'],
 ['truck',
  'motortruck',
  'headlight',
  'license plate',
  'mirror',
  'wheel',
  'window',
  'windshield'],
 ['tower'],
 ['chandelier',
  'pendant',
  'pendent',
  'arm',
  'bulb',
  'canopy',
  'chain',
  'shade'],
 ['awning', 'sunshade', 'sunblind'],
 ['streetlight', 'street lamp', 'lamp housing'],
 ['booth'],
 ['television receiver',
  'television',
  'television set',
  'tv',
  'tv set',
  'idiot box',
  'boob tube',
  'telly',
  'goggle box',
  'screen'],
 ['airplane',
  'aeroplane',
  'plane',
  'fuselage',
  'landing gear',
  'stabilizer',
  'turbine engine',
  'wing'],
 ['dirt track'],
 ['apparel', 'wearing apparel', 'dress', 'clothes'],
 ['pole'],
 ['land', 'ground', 'soil'],
 ['bannister', 'banister', 'balustrade', 'balusters', 'handrail'],
 ['escalator'],
 ['ottoman',
  'pouf',
  'pouffe',
  'puff',
  'hassock',
  'leg',
  'seat',
  'seat base',
  'seat cushion'],
 ['bottle', 'base', 'cap', 'label', 'neck'],
 ['buffet'],
 ['poster', 'posting', 'placard', 'notice', 'bill', 'card'],
 ['stage'],
 ['van',
  'door',
  'headlight',
  'license plate',
  'mirror',
  'taillight',
  'wheel',
  'window',
  'windshield'],
 ['ship'],
 ['fountain'],
 ['conveyer belt'],
 ['canopy'],
 ['washer'],
 ['plaything', 'toy'],
 ['swimming pool'],
 ['stool', 'apron', 'footrest', 'leg', 'seat', 'stretcher'],
 ['barrel', 'cask'],
 ['basket', 'handbasket'],
 ['waterfall'],
 ['tent'],
 ['bag',
  'traveling bag',
  'travelling bag',
  'grip',
  'suitcase',
  'handbag',
  'pocketbook',
  'purse'],
 ['minibike', 'motorbike', 'license plate', 'wheel'],
 ['cradle'],
 ['oven', 'button panel', 'dial', 'door'],
 ['pool ball', 'ball'],
 ['food', 'solid food'],
 ['step', 'stair'],
 ['tank'],
 ['trade name', 'brand name', 'brand', 'marque'],
 ['microwave',
  'microwave oven',
  'button',
  'button panel',
  'buttons',
  'dial',
  'display',
  'door',
  'screen'],
 ['pot', 'flowerpot'],
 ['animal', 'animate being', 'beast', 'brute', 'creature', 'fauna'],
 ['bicycle', 'bike', 'wheel', 'cycle'],
 ['lake'],
 ['dishwasher', 'dish washer', 'dishwashing machine', 'button panel', 'door'],
 ['screen'],
 ['blanket', 'cover'],
 ['sculpture'],
 ['hood', 'exhaust hood', 'body', 'filter', 'vent'],
 ['sconce', 'arm', 'backplate', 'bulb', 'shade'],
 ['vase'],
 ['traffic light', 'traffic signal', 'stoplight', 'housing', 'pole'],
 ['tray'],
 ['ashcan',
  'trash can',
  'garbage can',
  'wastebin',
  'ash bin',
  'ash-bin',
  'ashbin',
  'dustbin',
  'trash barrel',
  'trash bin',
  'can',
  'tin',
  'tin can'],
 ['fan', 'blade', 'canopy', 'motor', 'shade', 'tube'],
 ['pier'],
 ['screen', 'crt screen'],
 ['plate'],
 ['monitor', 'monitoring device', 'screen'],
 ['bulletin board', 'notice board', ['board', 'plank']],
 ['shower'],
 ['radiator'],
 ['glass', 'drinking glass', 'base', 'bowl', 'opening', 'stem'],
 ['clock', 'face'],
 ['flag'],
 ['unknown']]