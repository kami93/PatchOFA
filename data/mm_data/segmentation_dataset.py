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

    net_input_aug = None
    if samples[0].get("source_2", None) is not None:
        src_tokens_2 = merge("source_2")
        src_lengths_2 = torch.LongTensor([s["source_2"].ne(pad_idx).long().sum() for s in samples])
        net_input_aug = {
            "src_tokens": src_tokens_2,
            "src_lengths": src_lengths_2
        }


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

    text_target = None
    if samples[0].get("text_target", None) is not None:
        text_target = merge("text_target")

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
        "net_input_aug": net_input_aug,
        "aux_input": aux_input,
        "target": target,
        "text_target": text_target,
        "downsampled_target": downsampled_target,
        "text2seg_target": text2seg_target
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
        
        prompt_prefix=self.cfg.prompt_prefix
        if len(prompt_prefix):
            self.prompt = self.encode_text(f' {prompt_prefix.lstrip()}')
        else:
            self.prompt = None
        self.prompt_order = self.cfg.prompt_order

        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")
        
        self.fakeimage_type = cfg.fakeimage_type
        self.prompt_type = cfg.prompt_type
        self.fakeimage_prompt_type = cfg.fakeimage_prompt_type
        self.num_seg = cfg.num_seg_tokens # 150 (ade) 171 (coco-fine) 27 (coco-coarse)
        if self.num_seg == 171:
            self.id2rawtext = [x for x in CLASSES_COCOF]
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_COCOF]
        elif self.num_seg == 27:
            self.id2rawtext = [x for x in CLASSES_COCOC]
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_COCOC]
        elif self.num_seg == 150:
            self.id2rawtext = [x for x in CLASSES_ADE]
            self.id2text = [self.encode_text(f" {x}") for x in CLASSES_ADE]
        else:
            raise NotImplementedError
        self.text_length = torch.tensor([len(x) for x in self.id2text])

        self.id2seg = np.array([f'<seg_{idx}>' for idx in range(self.num_seg + 1)])
        self.seg2code = self.encode_text(" ".join(self.id2seg), use_bpe=False)

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

        text_target = None
        # text_target = self.seg2code[:150].repeat_interleave(repeats=self.text_length, dim=0)
        
        # target_on_final_token = False
        # if target_on_final_token:
        #     fill_idx = self.text_length[:150].cumsum(dim=0)-1
        #     text_target = text_target.new_full(size=text_target.size(), fill_value=self.pad, dtype=torch.long)
        #     text_target[fill_idx] = self.seg2code[:150]
            
        gt_semantic_seg_downsampled = self.downsample_gt_seg(gt_semantic_seg.unsqueeze(0)).flatten()
        seg_ids = self.seg2code[gt_semantic_seg.flatten()]
        seg_ids_downsampled = self.seg2code[gt_semantic_seg_downsampled]
        
        prev_output_item = torch.cat([self.bos_item, seg_ids_downsampled])
        downsampled_target = torch.cat([seg_ids_downsampled, self.eos_item])
        target = torch.cat([seg_ids, self.eos_item])

        # build 
        src_item_2 = None
        if self.prompt_type in {'gt_seg', 'all'}:
            if self.prompt_type == 'gt_seg' and self.split == 'train':
                prompt_ids = gt_semantic_seg_downsampled.unique().tolist()
            else:
                prompt_ids = [idx for idx in range(self.num_seg)]

            if self.prompt_order == 'random':
                np.random.shuffle(prompt_ids)
            elif self.prompt_order == 'sorted':
                prompt_ids.sort(key=lambda idx: self.id2rawtext[idx])
            elif self.prompt_order == 'none':
                pass
            else:
                raise NotImplementedError
            
            src_text = [self.bos_item]
            if self.prompt is not None:
                src_text += [self.prompt]
            
            src_text += [self.id2text[idx] for idx in prompt_ids]
            src_text += [self.eos_item]

            src_item = torch.cat(src_text)

        elif self.prompt_type == 'seg':
            src_text = [self.bos_item]
            if self.prompt is not None:
                src_text += [self.prompt]

            src_text += [self.seg2code[:self.num_seg], self.eos_item]
            src_item = torch.cat(src_text)

        elif self.prompt_type == 'prompt':
            assert self.prompt is not None
            src_item = torch.cat([self.bos_item, self.prompt, self.eos_item])

        elif self.prompt_type == 'random_20':
            if self.split == 'train':
                prompt_ids = np.random.choice(self.num_seg, size=20).tolist()
                src_text = torch.cat([self.id2text[idx] for idx in rand_idx])
                src_item = torch.cat([self.bos_item, src_text, self.eos_item]) # src_item is the student (strong aug.)
                
                # rand_idx = np.random.choice(150, size=20)
                src_text = torch.cat([self.id2text[idx] for idx in range(self.num_seg)])
                src_item_2 = torch.cat([self.bos_item, src_text, self.eos_item]) # src_item_2 is the teacher (weak aug.)
            
            else:
                # self.prompt_type is 'all' during validation.
                src_text = torch.cat([self.id2text[idx] for idx in range(self.num_seg)])
                src_item = torch.cat([self.bos_item, src_text, self.eos_item])

        else:
            raise NotImplementedError

        # Self-patch teacher index
        # mask = seg_ids_downsampled.unsqueeze(0) == seg_ids_downsampled.unsqueeze(1)
        
        # rand = torch.randn(size=(len(seg_ids_downsampled), len(seg_ids_downsampled)))
        
        # batch_idx_1d = torch.arange(len(seg_ids_downsampled))
        # batch_idx_2d = batch_idx_1d.unsqueeze(-1).expand(-1, len(seg_ids_downsampled))
        
        # perm = rand.argsort(-1)

        # mask_perm = mask[batch_idx_2d, perm]
        # random_choice = mask_perm.max(-1)[1]
        
        # target_teacher = perm[batch_idx_1d, random_choice]
        example = {
            "id": uniq_id,
            "source": src_item,
            "source_2": src_item_2,
            "patch_image": img,
            "patch_mask": patch_mask,
            "target": target,
            "text_target": text_target,
            "downsampled_target": downsampled_target,
            "prev_output_tokens": prev_output_item
        }

        if self.fakeimage_type == 'none':
            # no fake image
            return example
        
        elif self.fakeimage_type == 'gt_seg':
            fakeimage_ids = gt_semantic_seg_downsampled.tolist()
            
        elif self.fakeimage_type == 'random':
            fakeimage_ids = np.random.choice(self.num_seg, size=1024, replace=True).tolist()

        elif self.fakeimage_type.startswith('upsampling'):
            if self.fakeimage_type == 'upsampling':
                l, r = 1, 12
            elif len(self.fakeimage_type.split('_')) == 3:
                l, r = self.fakeimage_type.split('_')[1:3]
                l, r = int(l), int(r)
            else:
                raise NotImplementedError

            sh, sw = torch.randint(l,r,(2,))
            sh, sw = sh.item(), sw.item()
            rand = np.random.choice(self.num_seg, size=sh*sw, replace=True)
            rand = torch.from_numpy(rand).reshape(1, 1, sh, sw)
            fakeimage_ids = self.downsample_gt_seg(rand).reshape(-1).tolist()
                
        else:
            raise NotImplementedError

        embedbag_ids = torch.cat([self.id2text[idx] for idx in fakeimage_ids])
        embedbag_offsets = torch.tensor([self.text_length[idx] for idx in fakeimage_ids], dtype=torch.long).cumsum(dim=0)

        codes = self.seg2code[fakeimage_ids]
        target = torch.cat([codes, self.eos_item])
        prev_output_tokens = torch.cat([self.bos_item, codes])

        if self.fakeimage_prompt_type in {'gt_seg', 'all'}:
            if self.fakeimage_prompt_type == 'gt_seg' and self.split == 'train':
                prompt_ids = fakeimage_ids.unique().tolist()
            else:
                prompt_ids = [idx for idx in range(self.num_seg)]

            if self.prompt_order == 'random':
                np.random.shuffle(prompt_ids)
            elif self.prompt_order == 'sorted':
                prompt_ids.sort(key=lambda idx: self.id2rawtext[idx])
            elif self.prompt_order == 'none':
                pass
            else:
                raise NotImplementedError
            
            src_text = [self.bos_item]
            if self.prompt is not None:
                src_text += [self.prompt]
            
            src_text += [self.id2text[idx] for idx in prompt_ids]
            src_text += [self.eos_item]
            src_item = torch.cat(src_text)

        elif self.fakeimage_prompt_type == 'seg':
            src_text = [self.bos_item]
            if self.prompt is not None:
                src_text += [self.prompt]

            src_text += [self.seg2code[:self.num_seg], self.eos_item]
            src_item = torch.cat(src_text)

        elif self.fakeimage_prompt_type == 'prompt':
            assert self.prompt is not None
            src_item = torch.cat([self.bos_item, self.prompt, self.eos_item])

        else:
            raise NotImplementedError
        
        example["text2seg_patch_image"] = embedbag_ids
        example["text2seg_patch_mask"] = embedbag_offsets
        example["text2seg_source"] = src_item
        example["text2seg_target"] = target
        example["text2seg_prev_output_tokens"] = prev_output_tokens

        return example

        # rand = random.randint(0, 149)
        # text = CLASSES[rand]
        # seg = self.id2seg[rand]
        # segcode = self.seg2code[rand]
        
        # prob = random.random()

        # if prob >= 0.5:
        #     text2seg_prompt = f' " {text} "' + self.prompt
        #     text2seg_source = self.encode_text(text2seg_prompt)
        #     text2seg_src_item = torch.cat([self.bos_item, text2seg_source, self.eos_item])
        #     example["text2seg_source"] = text2seg_src_item

        #     text2seg_prev_output_item = torch.cat([self.bos_item, text2seg_source])
        #     example["text2seg_prev_output_tokens"] = text2seg_prev_output_item

        #     text2seg_target = torch.cat([text2seg_source, segcode.unsqueeze(0)])
        #     text2seg_target[:-1] = self.tgt_dict.pad()
        #     example["text2seg_target"] = text2seg_target
        # else:
        #     text2seg_prompt = f' " {seg} "' + self.prompt
        #     text2seg_source = self.encode_text(text2seg_prompt)
        #     text2seg_src_item = torch.cat([self.bos_item, text2seg_source, self.eos_item])
        #     example["text2seg_source"] = text2seg_src_item
            
        #     target_text = self.encode_text(f" {text}")
        #     text2seg_target = torch.cat([text2seg_source, target_text])
        #     text2seg_target[:-len(target_text)] = self.tgt_dict.pad()
        #     example["text2seg_target"] = text2seg_target

        #     text2seg_prev_output_item = torch.cat([self.bos_item, text2seg_source, target_text])
        #     example["text2seg_prev_output_tokens"] = text2seg_prev_output_item[:-1]


        # rand0, rand1 = np.random.choice(150, size=2, replace=False)
        # prob = random.random()
        # if prob >= 0.5:
        #     # Make Positive Pair
        #     text = CLASSES[rand0]
        #     seg = self.id2seg[rand0]
        #     pos_pair = [text, seg]
        #     pos_src_item = torch.cat([self.bos_item, self.get_src_item_given_pair(pos_pair), self.eos_item])
            
        #     example["text2seg_source"] = pos_src_item
            
        #     pos_prev_output_item = pos_src_item[:-1].clone()
        #     pos_target_item = torch.cat([pos_prev_output_item[1:], self.pos_tgt_item])
        #     pos_target_item[:-1] = self.tgt_dict.pad()
            
        #     example["text2seg_target"] = pos_target_item
        #     example["text2seg_prev_output_tokens"] = pos_prev_output_item
            
        # else:
        #     # Make Negative Pair
        #     text = CLASSES[rand0]
        #     seg = self.id2seg[rand1]
        #     neg_pair = [text, seg]
        #     neg_src_item = torch.cat([self.bos_item, self.get_src_item_given_pair(neg_pair), self.eos_item])
            
        #     example["text2seg_source"] = neg_src_item
            
        #     neg_prev_output_item = neg_src_item[:-1].clone()
        #     neg_target_item = torch.cat([neg_prev_output_item[1:], self.neg_tgt_item])
        #     neg_target_item[:-1] = self.tgt_dict.pad()
            
        #     example["text2seg_target"] = neg_target_item
        #     example["text2seg_prev_output_tokens"] = neg_prev_output_item

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
