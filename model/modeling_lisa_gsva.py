# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from .losses import dice_loss, sigmoid_ce_loss


class LisaGSVAMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
        else:
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
            self.init_seg_and_proj(self.config)

    def init_seg_and_proj(self, config):
        # SAM
        builder_sam = build_sam_vit_h if "sam_vit_h" in self.segmentation_model_path else \
            build_sam_vit_l if "sam_vit_l" in self.segmentation_model_path else build_sam_vit_b
        self.visual_model = builder_sam(self.segmentation_model_path)
        # Projection layer for SAM
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

class LisaGSVAModel(LisaGSVAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.seg_token_idx = kwargs.get("seg_token_idx", 0)


class LisaGSVAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.rej_token_idx = kwargs.pop("rej_token_idx")
        self.llm_tokenizer = kwargs.get("tokenizer", None)
        super().__init__(config, **kwargs)

        self.model = LisaGSVAModel(config, seg_token_idx=self.seg_token_idx, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.Np = self.model.vision_tower.num_patches
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def pad_sequnce_and_stack(self, input_ids, attention_masks, labels):
        input_ids = nn.utils.rnn.pad_sequence(input_ids, True, 0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, True, False)
        labels = nn.utils.rnn.pad_sequence(labels, True, IGNORE_INDEX)
        return input_ids, attention_masks, labels
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        do_segs: List[bool],
        inference: bool = False,
        reeval: bool = False,
        **kwargs,
    ):
        device, dtype = images.device, images.dtype
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        if inference: # Segmentation Eval
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            output_ids = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True
                )
                output_hidden_states.append(output_i.hidden_states)
                for k in range(length):
                    pred_output_ids = output_i.logits[k].argmax(dim=1)
                    pred_ids = input_ids[k].clone()
                    img_idx = (pred_ids == IMAGE_TOKEN_INDEX).nonzero().item()
                    pred_ids = torch.cat([pred_ids[0:img_idx], torch.zeros(self.Np, device=device, dtype=torch.int64), pred_ids[img_idx + 1:]], dim=0)
                    # [SEG] token prediction:
                    seg_index_gt = (pred_ids == self.seg_token_idx).nonzero(as_tuple=True)[0]
                    seg_index_pred = seg_index_gt - 1
                    pred_seg_values = torch.where((pred_output_ids[seg_index_pred] != self.seg_token_idx), self.rej_token_idx, self.seg_token_idx)
                    # [REJ] token prediction:
                    rej_index_gt = (pred_ids == self.rej_token_idx).nonzero(as_tuple=True)[0]
                    rej_index_pred = rej_index_gt - 1
                    pred_rej_values = torch.where((pred_output_ids[rej_index_pred] != self.rej_token_idx), self.seg_token_idx, self.rej_token_idx)
                    # Update 
                    pred_ids[seg_index_gt] = pred_seg_values
                    pred_ids[rej_index_gt] = pred_rej_values
                    # The above steps woll make the [SEG/REJ] predictions have the same number of elements to masks
                    output_ids.append(pred_ids)
                if reeval:
                    # Replace all [REJ] to [SEG], then re-eval
                    input_ids[input_ids == self.rej_token_idx] = self.seg_token_idx
                    output_i_reeval = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True
                    )
                    output_hidden_states[-1] = output_i_reeval.hidden_states
                    torch.cuda.empty_cache()
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else: # Training 
            images_clip_list = []
            for i in range(len(offset) - 1): # offset marks each begin and end index for each images.
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            # VLM inference, obtain LLaVA output
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx # mask for gathering [SEG] tokens
        seg_token_mask = torch.cat([seg_token_mask, torch.zeros(seg_token_mask.shape[0], 1, dtype=torch.bool, device=device)], dim=1)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat([torch.zeros(seg_token_mask.shape[0], self.Np - 1, dtype=torch.bool, device=device), seg_token_mask], dim=1)
        
        rej_token_mask = input_ids[:, 1:] == self.rej_token_idx
        rej_token_mask = torch.cat([rej_token_mask, torch.zeros(rej_token_mask.shape[0], 1, dtype=torch.bool, device=device)], dim=1)
        rej_token_mask = torch.cat([torch.zeros(rej_token_mask.shape[0], self.Np - 1, dtype=torch.bool, device=device),rej_token_mask], dim=1)
        mask_list_comp = []

        for lang_i in range(len(input_ids)):
            this_seg_token_m = seg_token_mask[lang_i].long() * 2
            this_rej_token_m = rej_token_mask[lang_i].long() * 1
            this_seg_rej = this_seg_token_m + this_rej_token_m
            gathered_idx = this_seg_rej.nonzero(as_tuple=True)[0]
            this_seg_rej = this_seg_rej[gathered_idx].eq(2).nonzero(as_tuple=True)[0]
            mask_list_comp.append(this_seg_rej)        
        
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset], dim=0
        )     
        
        pred_embeddings_ = []
        num_pred_embs = len(seg_token_offset) - 1
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        pred_masks = []
        pred_ious = []
        mask_img_map = [(t >= offset).long().argmin().item() - 1 for t in range(num_pred_embs)]
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[mask_img_map[i]].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[mask_img_map[i]],
                original_size=label_list[mask_img_map[i]].shape
            )
            pred_masks.append(pred_mask[:, 0])
            pred_ious.append(iou_predictions[:, 0])
        model_output = output
        gt_masks = masks_list
        if not inference: # training, only train the [SEG] masks, ignore [REJ] masks
            pred_masks_, mask_list_comp_ = [], []
            for k in range(len(offset) - 1):
                begin, end = offset[k], offset[k + 1]
                select_preds = pred_masks[begin:end]
                select_comps = mask_list_comp[begin:end]
                if len(select_preds) == 1:
                    pred_masks_.extend(select_preds)
                else:
                    pred_masks_.append(torch.cat(select_preds, dim=0))
                if len(select_comps) == 1:
                    mask_list_comp_.extend(select_comps)
                else:
                    mask_list_comp_.append(select_comps)
            pred_masks = pred_masks_
            mask_list_comp = mask_list_comp_
            assert len(gt_masks) == len(pred_masks)
            assert len(gt_masks) == len(mask_list_comp)
            pred_masks_= []
            for b_idx in range(batch_size):
                L, h, w = pred_masks[b_idx].shape
                if L == 0:
                    pred_masks_.append(pred_masks[b_idx])
                    # gt_masks[b_idx] = pred_masks[b_idx].detach()
                    continue
                this_pred_masks_ = torch.zeros_like(gt_masks[b_idx], dtype=torch.float32)
                if isinstance(mask_list_comp[b_idx], torch.Tensor):
                    this_pred_masks_[mask_list_comp[b_idx]] = pred_masks[b_idx]
                else:
                    assert isinstance(mask_list_comp[b_idx], list) and len(mask_list_comp[b_idx]) == L
                    for j in range(L):
                        this_pred_masks_[j] = pred_masks[b_idx][j:j + 1][mask_list_comp[b_idx][j]]
                pred_masks_.append(this_pred_masks_)
            pred_masks = pred_masks_
        
        for b in range(batch_size):
            for pm, gm in zip(pred_masks[b], gt_masks[b]):
                assert pm.shape == gm.shape, f"b_idx: {b}, pm.shape: {pm.shape}, gm.shape: {gm.shape}"
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "output_ids": output_ids
            }
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            if batch_idx >= len(gt_masks):
                raise ValueError(f"gt_masks are not in good shape with b_idx={batch_idx} >= len(gt_masks)={len(gt_masks)}, also len(preds)={len(pred_masks)}.")
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if (
                gt_mask.shape[0] != pred_mask.shape[0]
            ):
                i0, i1 = input_ids[0], input_ids[1]
                i0, i1 = i0[i0 != IMAGE_TOKEN_INDEX], i1[i1 != IMAGE_TOKEN_INDEX]
                print(f"gt: {gt_mask.shape}, pred: {pred_mask.shape}\n" + \
                    f"Prompt0: {self.llm_tokenizer.decode(i0)}\n" + \
                    f"Prompt1: {self.llm_tokenizer.decode(i1)}\n" + \
                    f"GT_MASK sum :{gt_mask.sum(dim=(1, 2))}\n"
                )
                raise RuntimeError("Found it!")
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss = ce_loss + mask_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss
        }