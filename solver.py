# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import torch
import time
import tqdm
from utils import AverageMeter, ProgressMeter, Summary

def train_one_epoch(train_loader, model_engine, epoch, train_iter, args, logger):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")


    progress = ProgressMeter(
        len(train_loader) if args.no_sampling else args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs),
        logger=logger
    )

    # switch to train mode
    model_engine.train()
    end = time.time()
    if args.no_sampling:
        for global_step, input_dict in enumerate(train_loader):
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model_engine(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_obj_loss = output_dict.get("mask_obj_loss", torch.zeros_like(mask_loss))
            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model_engine.backward(loss)
            model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter
    else:
        for global_step in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = model_engine(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                losses.update(loss.item(), input_dict["images"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

                model_engine.backward(loss)
            
                model_engine.step()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

        return train_iter

@torch.no_grad()
def validate(val_loader, model_engine, epoch, args, logger):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].long()
        output_list = (pred_masks[0] > 0).long()
        assert len(pred_masks) == 1

        device = pred_masks[0].device
        intersection, union, acc_iou = 0.0, 0.0, 0.0  
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    logger.info(f"[{epoch + 1:d}] On {val_loader.dataset.ds} giou: {giou:.4f}, ciou: {ciou:.4f}.")
    return giou, ciou

@torch.no_grad()
def eval_gres(val_loader, model_engine, epoch, args, logger):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    is_grefcoco = val_loader.dataset.ds == 'grefcoco' 
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        input_dict["reeval"] = True
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"][0].ge(0).int()
        gt_masks = output_dict["gt_masks"][0].int()
        
        output_ids = output_dict["output_ids"][0]
        seg_or_rej_index = ((output_ids == args.seg_token_idx) | (output_ids == args.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = (output_ids[seg_or_rej_index] == args.rej_token_idx)
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)
            
        for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            
            if gt.sum() < 1.0: # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)
                    g_iou_meter.update(0.0)
                    if is_grefcoco:
                        union_meter.update(union_i)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)
                else:
                    nt_tn_meter.update(1.0)
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred.contiguous().clone(),
                    gt.contiguous().clone(),
                    K=2, ignore_index=255
                )
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()
    
    # total_masks = nt_tp_meter.sum + nt_tn_meter.sum + nt_fp_meter.sum + nt_fn_meter.sum
    # masks_have_targets = nt_tn_meter.sum + nt_fp_meter.sum
    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum) # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    logger.info(f"[{epoch + 1:d}] {val_loader.dataset.ds} giou: {g_iou:.4f}, ciou: {c_iou:.4f}, N_acc: {N_acc:.4f}, T_acc: {T_acc:.4f}.")
    return g_iou, c_iou, N_acc, T_acc

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output_shape = {output.shape}, target_shape = {target.shape}"
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
