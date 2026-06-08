#!/usr/bin/env python
'''
@File    :   validation.py
@Time    :   2021/06/30 17:08:35
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''
import os

import numpy as np
import torch
from tqdm import tqdm

from src.losses.utils import bbox_oiou, bbox_overlaps

from .utils import (visualize_centerness_overlap_gt, visualize_overlap_gt,
                    visualize_overlap_mask_gt)


def _recalls(ious, thrs):
    img_num = ious.shape[0]
    recalls = np.zeros(thrs.size)
    for i, thr in enumerate(thrs):
        recalls[i] = (ious >= thr).sum() / float(img_num)
    return recalls


def eval_recalls(ious, iou_thrs=0.5, logger=None, title='Validation results'):
    """Calculate recalls.

    Args:
        ious (list[ndarray]): a list of arrays of shape (n, 4)
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    recalls = _recalls(np.array(ious), np.array(iou_thrs))
    if logger:
        logger.info(title + ':')
        logger.info('Recalls\t R0.5\t R0.75\t R0.9\t')
        logger.info('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(
            recalls[0], recalls[5], recalls[8]))
    else:
        print(title + ':')
        print('Recalls\t R0.5\t R0.75\t R0.9\t')
        print('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(
            recalls[0], recalls[5], recalls[8]))
    return recalls


def compute_mask_iou(pred_mask, gt_mask, valid_mask=None, threshold=0.5):
    pred_mask = (pred_mask >= threshold).bool()
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    gt_mask = (gt_mask > 0).bool()
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.bool()
        pred_mask = pred_mask & valid_mask
        gt_mask = gt_mask & valid_mask

    intersection = (pred_mask & gt_mask).flatten(1).sum(dim=1).float()
    union = (pred_mask | gt_mask).flatten(1).sum(dim=1).float()
    empty = union == 0
    return torch.where(
        empty, torch.ones_like(union), intersection / union.clamp(min=1.0),
    )


def build_valid_mask(valid_hw, image_hw):
    if valid_hw is None:
        return None
    h, w = image_hw
    mask = torch.zeros(
        (valid_hw.shape[0], 1, h, w), dtype=torch.bool, device=valid_hw.device,
    )
    for i, hw in enumerate(valid_hw):
        valid_h = int(hw[0].item())
        valid_w = int(hw[1].item())
        mask[i, :, :valid_h, :valid_w] = True
    return mask


@torch.no_grad()
def evaluate(model,
             dataloader,
             logger,
             save_path,
             iou_thrs=np.arange(0.5, 0.96, 0.05),
             epoch=0,
             oiou=False,
             viz=False):
    ious = []
    oious = []
    mask_ious = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        data = model(batch, validation=True)

        if not oiou:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  data['pred_bbox1'].cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  data['pred_bbox2'].cpu(),
                                  is_aligned=True)
        else:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  data['pred_bbox1'].cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  data['pred_bbox2'].cpu(),
                                  is_aligned=True)
            oious1 = bbox_oiou(batch['overlap_box1'], data['pred_bbox1'].cpu())
            oious2 = bbox_oiou(batch['overlap_box2'], data['pred_bbox2'].cpu())
            oious += list(oious1.numpy()) + list(oious2.numpy())
        ious += list(ious1.numpy()) + list(ious2.numpy())

        has_masks = (
            'pred_mask1' in data and 'pred_mask2' in data
            and 'gt_mask1' in batch and 'gt_mask2' in batch
        )
        if has_masks:
            valid_mask1 = build_valid_mask(
                batch.get('valid_hw1'), batch['image1'].shape[1:3],
            )
            valid_mask2 = build_valid_mask(
                batch.get('valid_hw2'), batch['image2'].shape[1:3],
            )
            miou1 = compute_mask_iou(
                data['pred_mask1'].cpu(), batch['gt_mask1'],
                None if valid_mask1 is None else valid_mask1.cpu(),
            )
            miou2 = compute_mask_iou(
                data['pred_mask2'].cpu(), batch['gt_mask2'],
                None if valid_mask2 is None else valid_mask2.cpu(),
            )
            mask_ious += list(miou1.numpy()) + list(miou2.numpy())

        if i % 10 == 0 and viz:
            bbox1 = data['pred_bbox1'][0].cpu().numpy().astype(int)
            bbox2 = data['pred_bbox2'][0].cpu().numpy().astype(int)
            gt_bbox1 = batch['overlap_box1'][0].numpy().astype(int)
            gt_bbox2 = batch['overlap_box2'][0].numpy().astype(int)
            viz_name = os.path.join(
                str(save_path),
                'epoch' + str(epoch) + '_' + batch['file_name'][0])
            if has_masks:
                pred_mask1 = (
                    data['pred_mask1'][0, 0].cpu().numpy() >= 0.5
                ).astype(np.uint8)
                pred_mask2 = (
                    data['pred_mask2'][0, 0].cpu().numpy() >= 0.5
                ).astype(np.uint8)
                gt_mask1 = batch['gt_mask1'][0].numpy().astype(np.uint8)
                gt_mask2 = batch['gt_mask2'][0].numpy().astype(np.uint8)
                visualize_overlap_mask_gt(
                    batch['image1'][0].cpu().numpy() * 255,
                    bbox1,
                    gt_bbox1,
                    pred_mask1,
                    gt_mask1,
                    batch['image2'][0].cpu().numpy() * 255,
                    bbox2,
                    gt_bbox2,
                    pred_mask2,
                    gt_mask2,
                    viz_name,
                )
            elif 'pred_center1' in data.keys():
                visualize_centerness_overlap_gt(
                    batch['image1'][0].cpu().numpy() * 255, bbox1, gt_bbox1,
                    data['pred_center1'][0].cpu().numpy(),
                    batch['image2'][0].cpu().numpy() * 255, bbox2, gt_bbox2,
                    data['pred_center2'][0].cpu().numpy(), viz_name)
            else:
                visualize_overlap_gt(batch['image1'][0].cpu().numpy() * 255,
                                     bbox1, gt_bbox1,
                                     batch['image2'][0].cpu().numpy() * 255,
                                     bbox2, gt_bbox2, viz_name)

    eval_recalls(ious, iou_thrs, logger, title='BBox validation results')
    if oiou:
        eval_recalls(oious, iou_thrs, logger, title='BBox OIoU results')
    if mask_ious:
        if logger:
            logger.info('Mask mean IoU\t {:.5f}'.format(np.mean(mask_ious)))
        else:
            print('Mask mean IoU\t {:.5f}'.format(np.mean(mask_ious)))
        eval_recalls(mask_ious, iou_thrs, logger, title='Mask validation results')


@torch.no_grad()
def evaluate_dummy(model,
                   dataloader,
                   logger,
                   save_path,
                   iou_thrs=np.arange(0.5, 0.96, 0.05),
                   epoch=0,
                   oiou=False,
                   viz=False):
    ious = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image1, image2 = batch['image1'].cuda(), batch['image2'].cuda()
        pred_bbox1, pred_bbox2 = model.forward_dummy(image1, image2)
        if not oiou:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  pred_bbox1.cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  pred_bbox2.cpu(),
                                  is_aligned=True)
        else:
            ious1 = bbox_oiou(batch['overlap_box1'], pred_bbox1.cpu())
            ious2 = bbox_oiou(batch['overlap_box2'], pred_bbox2.cpu())
        ious += list(ious1.numpy()) + list(ious2.numpy())
        if viz:
            bbox1 = pred_bbox1[0].cpu().numpy().astype(int)
            bbox2 = pred_bbox2[0].cpu().numpy().astype(int)
            gt_bbox1 = batch['overlap_box1'][0].numpy().astype(int)
            gt_bbox2 = batch['overlap_box2'][0].numpy().astype(int)
            viz_name = os.path.join(
                str(save_path),
                'epoch' + str(epoch) + '_' + batch['file_name'][0])
            visualize_overlap_gt(batch['image1'][0].numpy() * 255, bbox1,
                                 gt_bbox1, batch['image2'][0].numpy() * 255,
                                 bbox2, gt_bbox2, viz_name)
    eval_recalls(ious, iou_thrs, logger)
