# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything import ResizeLongestSide

from .data_processing import get_mask_from_json

from .reason_seg_dataset import ReasonSegDataset
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

from .refer import REFER
from .grefer import G_REFER
from .refzom import REFZOM_REFER

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    do_segs = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        do_seg,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        do_segs.append(do_seg)
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else: # conv_type == 'llava_llama_2'
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    if inferences[0] == False:
        Np = images_clip.size(1) * images_clip.size(2) // 196
        truncate_len = tokenizer.model_max_length - (Np - 1)

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "do_segs": do_segs
    }


class MixedTrainingDataset(TorchDataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        no_sampling=False
    ):
        self.no_sampling = no_sampling
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        no_sampling
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        no_sampling
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        no_sampling
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        no_sampling
                    )
                )
            
            if self.no_sampling:
                assert len(self.all_datasets) == 1, "Only one dataset is allowed with the no-sampling strategy."
                

    def __len__(self):
        if self.no_sampling:
            return len(self.all_datasets[0])
        else:
            return self.samples_per_epoch

    def __getitem__(self, idx):
        
        if self.no_sampling:
            data = self.all_datasets[0]
            return *data[idx], False
        else:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            data = self.all_datasets[ind]
            return *data[0], False


class ValDataset(TorchDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"
            
            if ds == "grefcoco":
                refer_api = G_REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            elif ds == 'refzom':
                refer_api = REFZOM_REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds)
            else:
                refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, 'refer_seg', "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        'refer_seg',
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"
            
        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        
        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                # grefcoco multiple annid start
                if self.ds in ['grefcoco', 'refzom']:
                    no_target = ann_id == [-1] if self.ds == 'grefcoco' else ann_id == []
                    if no_target: # no target
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    elif len(ann_id) > 1: # multi target / already merged ?
                        m = []
                        for sub_ann_id in ann_id:
                            sub_mask_info = annotations[sub_ann_id]['segmentation']
                            if len(sub_mask_info) == 0:
                                sub_m = np.zeros((image_info["height"], image_info["width"], 1))
                            else:
                                if isinstance(sub_mask_info, dict):
                                    if isinstance(sub_mask_info["counts"], list):
                                        # convert to compressed RLE
                                        rle = mask.frPyObjects(sub_mask_info, image_info["height"], image_info["width"])
                                else:
                                    # filter out invalid polygons (< 3 points)
                                    polygons = [poly for poly in sub_mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                    if len(polygons) == 0:
                                        continue  # ignore this instance
                                    rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                                sub_m = mask.decode(rle)
                                if sub_m.ndim < 3:
                                    assert sub_m.ndim == 2
                                    sub_m = sub_m[..., np.newaxis]
                            sub_m = np.sum(sub_m, axis=2)
                            m.append(sub_m)
                        m = np.sum(m, axis=0)[..., np.newaxis]
                    else:
                        assert len(ann_id) == 1 and ann_id[0] != -1
                        mask_info = annotations[ann_id[0]]['segmentation']
                        if len(mask_info) == 0:
                            m = np.zeros((image_info["height"], image_info["width"], 1))
                        else:
                            if isinstance(mask_info, dict):
                                if isinstance(mask_info["counts"], list):
                                    # convert to compressed RLE
                                    rle = mask.frPyObjects(mask_info, image_info["height"], image_info["width"])
                            else:
                                # filter out invalid polygons (< 3 points)
                                polygons = [poly for poly in mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                if len(polygons) == 0:
                                    continue  # ignore this instance
                                rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                    m = np.sum(m, axis=2)
                    masks.append(m)
                else:
                    ann = annotations[ann_id]
                    if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if type(ann["segmentation"][0]) == list:  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
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
        else:
            masks = [mask_json]

        if self.data_type == 'refer_seg':
            conv.messages = []
            stripped_refers = [s.strip().strip('.') for s in sampled_sents]
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN + "\n What are " +
                ", ".join(stripped_refers) + 
                "in this image? Please output segmentation masks."
            )
            ref_w_segs = []
            assert len(stripped_refers) == len(masks)
            for t_idx, t in enumerate(stripped_refers):
                formatted_text = f"{t}:[REJ]" if masks[t_idx].sum() < 1.0 else f"{t}:[SEG]"
                ref_w_segs.append(formatted_text)
            conv.append_message(
                conv.roles[1],
                "Sure," + ", ".join(ref_w_segs) + "."
            )
            conversations.append(conv.get_prompt())
        else:
            i = 0
            while i < len(sampled_sents):
                conv.messages = []
                text = sampled_sents[i].strip()
                seg_token = "[SEG]"
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n {} Please output segmentation mask.".format(text),
                    )
                    
                    conv.append_message(conv.roles[1], seg_token)
                else:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            text
                        ),
                    )
                    conv.append_message(conv.roles[1], seg_token)
                conversations.append(conv.get_prompt())
                i += 1
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks.astype(np.uint8))
        masks = masks.bool().byte()
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True
        do_seg = True
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            do_seg,
            inference
        )