# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
import utils.transforms as T
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

    def merge(src_items):
        return data_utils.collate_tokens(
            src_items,
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])

    words_batch = []
    src_items = []
    word_batch_idx = []
    num_samples_batch = []
    for batch_idx, sample in enumerate(samples):
        src_items_i = sample['src_items']
        words = []
        for word, src_item in src_items_i.items():
            src_items.append(src_item)
            words.append(word)

        words_batch.append(words)
        num_samples_batch.append(len(src_items_i))
        word_batch_idx.append(torch.zeros(size=(len(src_items_i), ), dtype=torch.long) + batch_idx)

        # for noun, src_item in sample["noun_src_items"].items():
        #     words.append(noun)
        #     src_items.append(src_item)
        # word_batch_idx.append(
        #     torch.zeros(size=(len(sample["noun_src_items"]), ), dtype=torch.long) + batch_idx
        #     )
        # num_samples += len(sample["noun_src_items"])

        # for object, src_item in sample["object_src_items"].items():
        #     words.append(object)
        #     src_items.append(src_item)
        # word_batch_idx.append(
        #     torch.zeros(size=(len(sample["object_src_items"]), ), dtype=torch.long) + batch_idx
        #     )
        # num_samples += len(sample["object_src_items"])

        # for answer, src_item in sample["answer_src_items"].items():
        #     words.append(answer)
        #     src_items.append(src_item)
        # word_batch_idx.append(
        #     torch.zeros(size=(len(sample["answer_src_items"]), ), dtype=torch.long) + batch_idx
        #     )
        # num_samples += len(sample["answer_src_items"])


    word_batch_idx = torch.cat(word_batch_idx)
    num_samples_batch = torch.tensor(num_samples_batch, dtype=torch.long)

    src_tokens = merge(src_items)
    src_lengths = torch.LongTensor([src.ne(pad_idx).long().sum() for src in src_tokens])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    w_resize_ratios = torch.tensor([s["w_resize_ratio"] for s in samples])
    h_resize_ratios = torch.tensor([s["h_resize_ratio"] for s in samples])
    
    noun_ids = np.array([s["noun_ids"] for s in samples])
    object_ids = np.array([s["object_ids"] for s in samples])
    answer_ids = np.array([s["answer_ids"] for s in samples])
    raw_information = np.array([s['raw_information'] for s in samples])

    batch = {
        "id": id,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": None,
            "word_batch_idx": word_batch_idx,
        },
        "num_samples_batch": num_samples_batch,
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        "noun_ids": noun_ids,
        "object_ids": object_ids,
        "answer_ids": answer_ids,
        "raw_information": raw_information,
        "words_batch": words_batch
    }

    return batch

class RefVQADataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        max_object_length=30,
        patch_image_size=512,
        imagenet_default_mean_and_std=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict=None)
        self.max_object_length = max_object_length

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_image_size = patch_image_size
        # for positioning
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = ' which region does the text " {} " describe?'
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = '这段文字" {} "描述的是哪个区域？'

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        s = self.src_dict.encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
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
        item = self.dataset[index]
        if len(item) == 6:
            uniq_id, image, line_id, question, ref, predict_objects = item
        elif len(item) == 5:
            uniq_id, image, question, ref, predict_objects = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))) # .convert("RGB")
        patch_image = self.patch_resize_transform(image)

        w, h = image.size

        question = self.pre_question(question)
        question = question + '?' if not question.endswith('?') else question

        idx = 0
        pos_tag = nltk.pos_tag(question[:-1].split())
        noun_src_items = dict()
        noun_ids = dict()
        for word, tag in pos_tag:
            tokens = self.encode_text(' {}'.format(word))
            idx += len(tokens)
            if tag in {'NN', 'NNS', 'NNPS', 'NNP'}:
                if word not in noun_ids:
                    src_item = self.encode_text(self.prompt.format(word))
                    noun_src_items[word] = torch.cat([self.bos_item, src_item, self.eos_item])
                    noun_ids[word] = [idx]
                else:
                    noun_ids[word].append(idx)

        predict_object_seq = []
        if predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            
            idx = 0
            object_src_items = dict()
            object_ids = dict()
            for word in predict_object_seq.split():
                tokens = self.encode_text(' {}'.format(word))
                idx += len(tokens)

                if word not in object_ids:
                    src_item = self.encode_text(self.prompt.format(word))
                    object_src_items[word] = torch.cat([self.bos_item, src_item, self.eos_item])
                    object_ids[word] = [idx]
                else:
                    object_ids[word].append(idx)


        answers = [item.split('|!+')[1] for item in ref.split('&&')]

        idx = 0
        answer_src_items = dict()
        answer_ids = dict()
        for word in answers:
            tokens = self.encode_text(' {}'.format(word))
            idx += len(tokens)
        
            if word not in answer_ids:
                src_item = self.encode_text(self.prompt.format(word))
                answer_src_items[word] = torch.cat([self.bos_item, src_item, self.eos_item])
                answer_ids[word] = [idx]
            else:
                answer_ids[word].append(idx)

        w_resize_ratio = self.patch_image_size / w
        h_resize_ratio = self.patch_image_size / h

        patch_mask = torch.tensor([True])

        raw_information = {
            "question": question,
            "object": predict_object_seq,
            "answer": answers,
            "image": image,
            "uniq_id": uniq_id,
            "line_id": line_id
        }

        src_items = dict()
        src_items.update(noun_src_items)
        src_items.update(object_src_items)
        src_items.update(answer_src_items)

        example = {
            "id": uniq_id,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "src_items": src_items,
            "noun_src_items": noun_src_items,
            "object_src_items": object_src_items,
            "answer_src_items": answer_src_items,
            "w_resize_ratio": w_resize_ratio,
            "h_resize_ratio": h_resize_ratio,
            "noun_ids": noun_ids,
            "object_ids": object_ids,
            "answer_ids": answer_ids,
            "raw_information": raw_information
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
