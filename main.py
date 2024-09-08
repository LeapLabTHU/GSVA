# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import argparse
import os
import shutil
from functools import partial

import torch
import transformers
import deepspeed
import deepspeed.comm as dist
from torch.utils.data import DataLoader, DistributedSampler

import model.llava.conversation as conversation_lib
from model import LisaGSVAForCausalLM, add_task_tokens, init_vision_seg_for_model
from data import MixedTrainingDataset, ValDataset, collate_fn
from solver import train_one_epoch, validate, eval_gres
from utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="GSVA Training and Evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="For local rank in distributed training")
    parser.add_argument(
        "--mllm_model_path", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="Where do we store the huge datasets?")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="precision for training and inference")
    parser.add_argument("--image_size", default=1024, type=int, help="Image size of segmentation model.")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=0, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    
    parser.add_argument("--log_base_dir", default="./outputs", type=str)
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=20, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--segmentation_model_path", default=None, type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False, help='Whether resume the latest checkpoint when training is interrupted.')
    parser.add_argument("--no_sampling", action="store_true", default=False, help="Only one dataset finetuning, train on full length dataset.")
    parser.add_argument('--val_refzom', action='store_true', default=False, help='Default gres/zom evaluation, if True, RefZOM, else gRefCOCO.')
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--merge_lora_path", type=str, default=None, help="Path to destination HF checkpoint.")
    parser.add_argument("--weight", type=str, default=None, help="Path to a bin ckpt.")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    # Get arguments from commandline
    args = parse_args()
    
    # Set up Deepspeed distributed environment
    torch.cuda.set_device(args.local_rank)
    dist.init_distributed()
    args.world_size = world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()

    # Set up logging dir
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir, rank, name=args.exp_name)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mllm_model_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )
    tokenizer, args = add_task_tokens(tokenizer, args)

    # Determine working model precision
    args.torch_dtype = torch.float32
    if args.precision == "bf16":
        args.torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        args.torch_dtype = torch.half

    # Prepare model creation arguments
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "segmentation_model_path": args.segmentation_model_path,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "tokenizer": tokenizer,
        "rej_token_idx": args.rej_token_idx
    }
    model = LisaGSVAForCausalLM.from_pretrained(
        args.mllm_model_path, 
        torch_dtype=args.torch_dtype,
        **model_args
    )
    # Set up two vision models for whole model, and lora
    model = init_vision_seg_for_model(model, tokenizer, args)
    # Evaluation or finetuning, btw, merge-lora always fails
    if args.weight is not None: # `args.weight`` is a large `*.bin` file.
        state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Load trained weights successfully!")
    # Specify the conversation type
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    # Build training set
    if args.eval_only:
         train_dataset = None
    else:
        train_dataset = MixedTrainingDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            no_sampling=args.no_sampling
        )
    if args.no_eval:
        val_dataset = None
        logger.info(f"Training with {len(train_dataset)} examples.")
    else:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size
        )
        grefcoco_val_ds = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            'refzom|final|test' if args.val_refzom else 'grefcoco|unc|val',
            args.image_size
        )
        if args.eval_only:
            logger.info(f"Testing with {len(val_dataset)} examples.")
        else:
            logger.info(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples, also validating on gRefCOCO with {len(grefcoco_val_ds)} examples.")
    # The accelerated training configurations only work for ZeRO-2.
    if args.eval_only:
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            }
        }
    else:
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "weight_decay": 0.0,
                    "betas": (args.beta1, args.beta2),
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": args.epochs * args.steps_per_epoch,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.lr,
                    "warmup_num_steps": 100,
                    "warmup_type": "linear",
                },
            },
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 1e9,
                "allgather_bucket_size": 1e9
            }
        }
    # Build a model engine wrapped with Deepspeed
    if args.eval_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config
        )
    else:
        logger.info('Before initializing deepspeed zero optimizer...')
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=local_rank,
            ),
            config=ds_config
        )
        train_loader.num_local_io_workers = args.workers
        logger.info('After initializing deepspeed zero optimizer!')
    # resume deepspeed checkpoint, `auto-resume` snippets are borrowed from Swin Transfomer codebase:
    # https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/utils.py#L163
    if args.auto_resume:
        checkpoints = os.listdir(args.log_dir)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.startswith('ckpt_model')]
        if len(checkpoints) > 0:
            args.resume = max([os.path.join(args.log_dir, d) for d in checkpoints], key=os.path.getmtime)
            logger.info(f"Auto resume found latest: {args.resume}")
        else:
            logger.info("No auto resume.")
    if args.resume: # resume from training, scattered checkpoints (list of ***.pt)
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        logger.info(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )
    # Build validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=local_rank
            )
        )
        if val_dataset.ds not in ['grefcoco', 'refzom']:
            grefcoco_sampler = DistributedSampler(grefcoco_val_ds, shuffle=False, drop_last=False)
            grefcoco_loader = DataLoader(
                grefcoco_val_ds,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=grefcoco_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=local_rank
                )
            )
        else:
            grefcoco_loader = None
    # If we only want to evaluate models, then we evaluate them and quit the program.
    if args.eval_only:
        if val_dataset.ds in ['grefcoco', 'refzom']:
            eval_gres(val_loader, model_engine, 0, args, logger)
        else:
            validate(val_loader, model_engine, 0, args, logger)
        return
    # Otherwise, we train the model using the initialized Deepspeed-Zero model engine.
    logger.info("Training begin!")
    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, keep a `train_iter`` for iter-based training
        train_iter = train_one_epoch(train_loader, model_engine, epoch, train_iter, args, logger)
        # barrier for saving checkpoints
        dist.barrier()
        save_dir = os.path.join(args.log_dir, f"ckpt_model_{epoch + 1:02d}")
        if rank == 0 and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model_engine.save_checkpoint(save_dir)
        dist.barrier()
        # Skip if we don't need evalutation
        if args.no_eval:
            continue
        else:
            reason_giou, reason_ciou = validate(val_loader, model_engine, epoch, args, logger)
            grefcoco_giou, grefcoco_ciou, n_acc, t_acc = eval_gres(grefcoco_loader, model_engine, epoch, args, logger)
            if rank == 0:
                with open(os.path.join(args.log_dir, "quick_look_result.log"), "a") as t:
                    t.write(
                        f"[{epoch + 1}] reasonseg_val: gIoU:{reason_giou:.4f}, cIoU:{reason_ciou:.4f}, grefcoco_val: gIoU:{grefcoco_giou:.4f}, cIoU:{grefcoco_ciou:.4f}, NAcc:{n_acc:.4f}, TAcc:{t_acc:.4f}.\n"
                    )

if __name__ == "__main__":
    main()
