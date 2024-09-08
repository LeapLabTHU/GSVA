# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (ANSWER_LIST, ANSWER_LIST_MODE4_END,
                    ANSWER_LIST_MODE4_START, ANSWER_LIST_MODE4_TEMPLATE,
                    SHORT_QUESTION_LIST, SHORT_QUESTION_LIST_MODE4)

from model.segment_anything import ResizeLongestSide
from .refzom import REFZOM_REFER
from .grefer import G_REFER
from .refer import REFER



class ReferSegDataset(TorchDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        no_sampling=False,
    ):
        self.no_sampling = no_sampling
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST_MODE4
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            elif ds == 'refzom':
                refer_api = REFZOM_REFER(DATA_DIR, ds)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds
            
        if self.no_sampling:
            assert len(self.refer_seg_ds_list) == 1

    def __len__(self):
        if self.no_sampling:
            ds = self.refer_seg_ds_list[0]
            return len(self.refer_seg_data[ds]["images"])
        else:
            return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        if not self.no_sampling:
            idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)
        
        n_pos_to_sample = self.num_classes_per_sample
        
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= n_pos_to_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=n_pos_to_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
            
        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if isinstance(ann["segmentation"], dict):
                                rle = ann["segmentation"]
                                assert isinstance(rle["counts"], list)
                                # convert to compressed RLE
                                rle = mask.frPyObjects(rle, image_info["height"], image_info["width"])
                            else: 
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                            m = np.sum(m, axis=2).astype(np.uint8)  # convert to np.uint8
                            m = m
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []
        choice = np.random.randint(0, len(ANSWER_LIST_MODE4_START))
        question_template = random.choice(self.short_question_list)
        texts = []
        for text in sampled_sents:
            text = text.strip()
            assert text != ""
            assert len(text.split("||")) == 1
            texts.append(text.strip('.'))
        questions.append(question_template.format(class_name=", ".join(texts).lower()))
        ans_start, ans_template, ans_end = ANSWER_LIST_MODE4_START[choice], ANSWER_LIST_MODE4_TEMPLATE[choice],ANSWER_LIST_MODE4_END[choice]
        seg_token_parts = []
        all_rej = True
        for t_idx, t in enumerate(texts):
            output_cls_prompt = ans_template.format(class_name=t)
            if masks[t_idx].sum() < 1.0:
                output_cls_prompt = output_cls_prompt.replace("SEG", "REJ")
            else:
                all_rej = False
            seg_token_parts.append(output_cls_prompt)
        if all_rej:
            return self.__getitem__(0)
        
        answers.append(
            ans_start + " " + ", ".join(seg_token_parts) + ans_end
        )
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks.astype(np.uint8))
        masks = masks.bool().byte()
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        do_seg = True
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            do_seg
        )
