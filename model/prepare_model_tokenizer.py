# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import torch
from peft import get_peft_model, LoraConfig
from model.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

def add_task_tokens(tokenizer, args):
    # 1. pad_token set to unknown token
    tokenizer.pad_token = tokenizer.unk_token
    # 2. add a [SEG] and a [REJ] token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens("[REJ]")
    args.rej_token_idx = tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
    # 3. add <im_start> and <im_end>, same as llava into tokenizer
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    return tokenizer, args

def init_vision_seg_for_model(model, tokenizer, args):
    # Register special token ids
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Set up gradckpt for saving memory
    model.gradient_checkpointing_enable()
    # Init CLIP-ViT
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=args.torch_dtype, device=args.local_rank)
    # Init segmentation module
    model.get_model().init_seg_and_proj(model.get_model().config)
    # Freeze all parameters
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    # Get Lora model, validation lora_r must be 0
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA finetuning with rank = {lora_r}.")
    
    model.resize_token_embeddings(len(tokenizer))
    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts_keys = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    if lora_r < 0:
        trainable_parts_keys.append("model.layers")
        print("No LoRA, full LLM finetuning.")
    elif lora_r == 0:
        print("LLM left frozen.")
    if not args.eval_only:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_parts_keys
                ]
            ):
                p.requires_grad_(True)
        # Set up input with grads
        model.enable_input_require_grads()


    return model