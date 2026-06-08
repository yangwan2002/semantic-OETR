#!/usr/bin/env python
"""
@File    :   __init__.py
@Time    :   2021/08/06 20:29:50
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import os

from .carla_fullimage_pairs import CarlaFullImagePairsDataset
from .megadepth_pairs import MegaDepthPairsDataset


def build_dataloader(cfg, dataset_path):
    if cfg.DATA_SOURCE == 'megadepth_pairs':
        dataset = MegaDepthPairsDataset(
            pairs_list_path=cfg.LIST_PATH,
            scene_info_path=os.path.join(dataset_path, 'scene_info'),
            base_path=dataset_path,
            pairs_per_scene=cfg.PAIRS_LENGTH,
            with_mask=cfg.WITH_MASK,
            train=cfg.TRAIN,
            image_size=cfg.IMAGE_SIZE,
        )
        return dataset
    if cfg.DATA_SOURCE == 'carla_fullimage_pairs':
        dataset = CarlaFullImagePairsDataset(
            pairs_list_path=cfg.LIST_PATH,
            base_path=dataset_path,
            pairs_per_scene=cfg.PAIRS_LENGTH,
            with_mask=cfg.WITH_MASK,
            train=cfg.TRAIN,
            image_size=cfg.IMAGE_SIZE,
            augment=getattr(cfg, 'AUGMENT', False),
            augment_brightness=getattr(cfg, 'AUGMENT_BRIGHTNESS', 0.2),
            augment_contrast=getattr(cfg, 'AUGMENT_CONTRAST', 0.2),
            augment_gamma=getattr(cfg, 'AUGMENT_GAMMA', 0.15),
            augment_rgb_shift=getattr(cfg, 'AUGMENT_RGB_SHIFT', 0.04),
            augment_blur_prob=getattr(cfg, 'AUGMENT_BLUR_PROB', 0.2),
        )
        return dataset
    else:
        raise ValueError(f'DATASET {cfg.DATA_SOURCE} not supported.')

