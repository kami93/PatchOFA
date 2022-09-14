# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import os.path as osp
import pickle as pkl

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

import nltk
nltk.download('averaged_perceptron_tagger')

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

    conf = None
    if samples[0].get("conf", None) is not None:
        conf = torch.cat([s['conf'] for s in samples], dim=0)

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prefix_tokens = None
    if samples[0].get("decoder_prompt", None) is not None:
        prefix_tokens = merge("decoder_prompt")
        prefix_tokens = prefix_tokens[:, 1:]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    noun_idx_all = []
    noun_patch_mask_all = []
    noun_batch_idx_all = []
    for batch_idx, s in enumerate(samples):
        if s['noun_idx'] is not None:
            noun_idx_all.append(s['noun_idx'])
            noun_patch_mask_all.append(s['noun_patch_mask'])
            noun_batch_idx_all.append(s['noun_batch_idx']+batch_idx)
    
    if len(noun_idx_all):
        noun_idx_all = torch.cat(noun_idx_all)
        noun_patch_mask_all = torch.cat(noun_patch_mask_all)
        noun_batch_idx_all = torch.cat(noun_batch_idx_all)
    else:
        noun_idx_all = None
        noun_patch_mask_all = None
        noun_batch_idx_all = None

    object_idx_all = []
    object_patch_mask_all = []
    object_batch_idx_all = []
    for batch_idx, s in enumerate(samples):
        if s['object_idx'] is not None:
            object_idx_all.append(s['object_idx'])
            object_patch_mask_all.append(s['object_patch_mask'])
            object_batch_idx_all.append(s['object_batch_idx']+batch_idx)
    
    if len(object_idx_all):
        object_idx_all = torch.cat(object_idx_all)
        object_patch_mask_all = torch.cat(object_patch_mask_all)
        object_batch_idx_all = torch.cat(object_batch_idx_all)
    else:
        object_idx_all = None
        object_patch_mask_all = None
        object_batch_idx_all = None

    raw_information = [s['raw_information'] for s in samples]
    boxes = [s['boxes'] for s in samples]

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens,
        },
        "aux_input": {
            "eos_idx": src_lengths - 1,
            "noun_idx": noun_idx_all,
            "noun_batch_idx": noun_batch_idx_all,
            "noun_patch_mask": noun_patch_mask_all,
            "object_idx": object_idx_all,
            "object_batch_idx": object_batch_idx_all,
            "object_patch_mask": object_patch_mask_all
        },
        "conf": conf,
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "prefix_tokens": prefix_tokens,
        "raw_information": raw_information,
        "boxes": boxes
    }

    return batch


class CustomVqaGenDataset(OFADataset):
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
        patch_size=16,
        add_object=False,
        constraint_trie=None,
        imagenet_default_mean_and_std=False,
        prompt_type="none",
        box_dir=None,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.patch_size = patch_size
        self.num_patches = self.patch_image_size // self.patch_size

        self.add_object = add_object
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type
        self.box_dir = box_dir
        if self.box_dir is not None and not osp.exists(self.box_dir):
            raise ValueError(f"box_dir not exists: {box_dir}")

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # def _get_boxes(self, uniq_id: int, line_id: int) -> dict:
        

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 6:
            uniq_id, image, line_id, question, ref, predict_objects = item
        elif len(item) == 5:
            line_id = None
            uniq_id, image, question, ref, predict_objects = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        w, h = image.size

        patch_image = self.patch_resize_transform(image)

        boxes = None
        if line_id is not None:
            boxes_file = osp.join(self.box_dir, self.split, f'{line_id}.pkl')
            with open(boxes_file, "rb") as f:
                boxes = pkl.load(f)

            new_boxes = dict()
            patch_boxes = dict()
            for name, box in boxes.items():
                new_box = box.copy()
                new_box[::2] = new_box[::2] * self.patch_image_size / w
                new_box[1::2] = new_box[1::2] * self.patch_image_size / h
                new_boxes[name] = new_box
                patch_boxes[name] = (new_box/self.patch_size).astype(np.int)

            boxes = new_boxes

        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        # src_item = self.encode_text(' {}'.format(question))

        src_item = []
        noun_idx = []
        noun_patch_mask = []
        idx = 0
        pos_tag = nltk.pos_tag(question[:-1].split())
        for word, tag in pos_tag:
            tokens = self.encode_text(' {}'.format(word))
            src_item.append(tokens)
            idx += len(tokens)
            if tag in {'NN', 'NNS', 'NNPS', 'NNP'}:
                noun_idx.append(idx-1)

                if boxes is not None:
                    box = patch_boxes.get(word)
                    box_mask = torch.full(size=(self.num_patches, self.num_patches), fill_value=float("-inf"))
                    
                    box_mask[box[1]:max(box[1]+1, box[3]), box[0]:max(box[0]+1, box[2])] = 0.0
                    if (box_mask == 0.0).sum() == 0:
                        logger.info("Bug Detected")
                        assert False

                    noun_patch_mask.append(box_mask.flatten())

        src_item.append(self.encode_text('?'))

        src_item = torch.cat(src_item)

        if len(noun_idx):
            noun_idx = torch.tensor(noun_idx, dtype=torch.long)

        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        tgt_item = self.encode_text(" {}".format(answer))

        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            # predict_object_item = self.encode_text(" object: {}".format(predict_object_seq))
            
            object_idx = []
            object_patch_mask = []
            predict_object_item = [self.encode_text(" object:")]
            idx = 2
            for word in predict_object_seq.split():
                tokens = self.encode_text(' {}'.format(word))
                predict_object_item.append(tokens)
                idx += len(tokens)
                object_idx.append(idx-1)

                if boxes is not None:
                    box = patch_boxes.get(word)
                    box_mask = torch.full(size=(self.num_patches, self.num_patches), fill_value=float("-inf"))
                    
                    box_mask[box[1]:max(box[1]+1, box[3]), box[0]:max(box[0]+1, box[2])] = 0.0
                    if (box_mask == 0.0).sum() == 0:
                        logger.info("Bug Detected")
                        assert False


                    object_patch_mask.append(box_mask.flatten())

            predict_object_item = torch.cat(predict_object_item)

            if len(object_idx):
                object_idx = torch.tensor(object_idx, dtype=torch.long)
                object_idx += len(src_item)

            src_item = torch.cat([src_item, predict_object_item])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])

        # account for self.bos_item
        if len(noun_idx):
            noun_idx += 1
        if len(object_idx):
            object_idx += 1

        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1] # src_item 에서 self.eos_item 를 제외 하고 decoder prompt 로 사용.
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item)-1] = self.tgt_dict.pad() # 질문에 해당하는 토큰들을 padding 토큰으로 바꿔줌.

        raw_information = {
            "question": question,
            "object": "object: {}".format(predict_object_seq),
            "answer": answer,
            "uniq_id": uniq_id
        }

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "conf": conf,
            "noun_idx": noun_idx if len(noun_idx) else None,
            "noun_patch_mask": torch.stack(noun_patch_mask) if boxes is not None and len(noun_idx) else None,
            "noun_batch_idx": torch.zeros_like(noun_idx) if len(noun_idx) else None,
            "object_idx": object_idx if len(object_idx) else None,
            "object_patch_mask": torch.stack(object_patch_mask) if boxes is not None and len(object_idx) else None,
            "object_batch_idx": torch.zeros_like(object_idx) if len(object_idx) else None,
            "raw_information": raw_information,
            "boxes": boxes
        }
        if self.constraint_trie is not None:
            # NLP 에서 같이 나오면 안되는 단어들에 대한 트리를 쓰는 알고리즘.
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
