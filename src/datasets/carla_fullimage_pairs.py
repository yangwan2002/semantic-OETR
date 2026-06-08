#!/usr/bin/env python
"""
Full-image CARLA pair dataset for OETR.

Unlike the original MegaDepth pair loader, this dataset keeps the original
image pair intact, resizes each image with preserved aspect ratio, pads to a
stride-aligned canvas, and directly supervises the single coarse overlap box
exported from CARLA depth reprojection.
"""

import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CarlaFullImagePairsDataset(Dataset):
    def __init__(
        self,
        pairs_list_path,
        base_path,
        train=True,
        pairs_per_scene=None,
        image_size=(1536, 864),
        with_mask=False,
        size_divisor=32,
        augment=False,
        augment_brightness=0.2,
        augment_contrast=0.2,
        augment_gamma=0.15,
        augment_rgb_shift=0.04,
        augment_blur_prob=0.2,
    ):
        self.pairs_list_path = pairs_list_path
        self.base_path = base_path
        self.train = train
        self.pairs_per_scene = pairs_per_scene
        self.image_size = tuple(image_size)
        self.with_mask = with_mask
        self.size_divisor = size_divisor
        self.augment = augment
        self.augment_brightness = float(augment_brightness)
        self.augment_contrast = float(augment_contrast)
        self.augment_gamma = float(augment_gamma)
        self.augment_rgb_shift = float(augment_rgb_shift)
        self.augment_blur_prob = float(augment_blur_prob)

        self.total_dataset = []
        self.dataset = []
        self.init_dataset()

    def init_dataset(self):
        self.total_dataset = []
        with open(self.pairs_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                bbox1 = np.array(item["overlap_box1"], dtype=np.float32)
                bbox2 = np.array(item["overlap_box2"], dtype=np.float32)
                if (
                    bbox1[0] >= bbox1[2]
                    or bbox1[1] >= bbox1[3]
                    or bbox2[0] >= bbox2[2]
                    or bbox2[1] >= bbox2[3]
                ):
                    continue
                self.total_dataset.append(item)

    def build_dataset(self):
        if not self.total_dataset:
            self.dataset = []
            return

        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)

        if self.pairs_per_scene:
            replace = len(self.total_dataset) < self.pairs_per_scene
            selected_ids = np.random.choice(
                len(self.total_dataset), self.pairs_per_scene, replace=replace,
            )
        else:
            selected_ids = np.arange(len(self.total_dataset))

        self.dataset = list(np.array(self.total_dataset, dtype=object)[selected_ids])
        if self.train:
            np.random.shuffle(self.dataset)
        else:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def _resolve_path(self, rel_or_abs_path):
        if os.path.isabs(rel_or_abs_path):
            return rel_or_abs_path
        return os.path.join(self.base_path, rel_or_abs_path)

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return image.astype(np.float32)

    def _photometric_augment(self, image):
        image = image.copy()

        if self.augment_contrast > 0 or self.augment_brightness > 0:
            alpha = 1.0 + np.random.uniform(
                -self.augment_contrast, self.augment_contrast,
            )
            beta = 255.0 * np.random.uniform(
                -self.augment_brightness, self.augment_brightness,
            )
            image = image * alpha + beta

        if self.augment_gamma > 0 and np.random.rand() < 0.8:
            gamma = 1.0 + np.random.uniform(
                -self.augment_gamma, self.augment_gamma,
            )
            image = 255.0 * np.power(
                np.clip(image / 255.0, 0.0, 1.0), np.clip(gamma, 0.7, 1.5),
            )

        if self.augment_rgb_shift > 0:
            rgb_shift = np.random.uniform(
                -self.augment_rgb_shift, self.augment_rgb_shift, size=(1, 1, 3),
            ) * 255.0
            image = image + rgb_shift.astype(np.float32)

        if self.augment_blur_prob > 0 and np.random.rand() < self.augment_blur_prob:
            kernel = int(np.random.choice([3, 5]))
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)

        return np.clip(image, 0.0, 255.0).astype(np.float32)

    def _resize_pad_image(self, image):
        target_w = int(self.image_size[0])
        target_h = int(self.image_size[1] if len(self.image_size) > 1 else self.image_size[0])

        h, w = image.shape[:2]
        scale = min(float(target_w) / float(w), float(target_h) / float(h))
        new_w = max(1, int(w * scale + 0.5))
        new_h = max(1, int(h * scale + 0.5))

        resized = cv2.resize(image, (new_w, new_h)).astype(np.float32)

        pad_w = int(math.ceil(float(target_w) / self.size_divisor) * self.size_divisor)
        pad_h = int(math.ceil(float(target_h) / self.size_divisor) * self.size_divisor)
        canvas = np.zeros((pad_h, pad_w, image.shape[2]), dtype=np.float32)
        canvas[:new_h, :new_w, :] = resized

        valid_h = int(math.ceil(float(new_h) / self.size_divisor))
        valid_w = int(math.ceil(float(new_w) / self.size_divisor))
        resize_mask = np.zeros(
            (pad_h // self.size_divisor, pad_w // self.size_divisor), dtype=bool,
        )
        resize_mask[:valid_h, :valid_w] = True
        return canvas, scale, resize_mask, (new_h, new_w)

    def _scale_bbox(self, bbox, scale, shape_hw):
        scaled = np.array(bbox, dtype=np.float32) * float(scale)
        new_h, new_w = shape_hw
        scaled[0::2] = np.clip(scaled[0::2], 0.0, max(0.0, float(new_w - 1)))
        scaled[1::2] = np.clip(scaled[1::2], 0.0, max(0.0, float(new_h - 1)))
        return scaled

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path1 = self._resolve_path(item["image_path1"])
        image_path2 = self._resolve_path(item["image_path2"])

        image1 = self._load_image(image_path1)
        image2 = self._load_image(image_path2)
        if self.train and self.augment:
            image1 = self._photometric_augment(image1)
            image2 = self._photometric_augment(image2)
        image1, scale1, resize_mask1, shape1 = self._resize_pad_image(image1)
        image2, scale2, resize_mask2, shape2 = self._resize_pad_image(image2)

        overlap_box1 = self._scale_bbox(item["overlap_box1"], scale1, shape1)
        overlap_box2 = self._scale_bbox(item["overlap_box2"], scale2, shape2)

        file_name = (
            f"frame{int(item['frame_id']):06d}_{item['src_sensor']}_"
            f"{item['tgt_sensor']}"
        )

        output = {
            "image1": torch.from_numpy(image1 / 255.0).float(),
            "image2": torch.from_numpy(image2 / 255.0).float(),
            "resize_mask1": torch.from_numpy(resize_mask1.astype(np.float32)),
            "resize_mask2": torch.from_numpy(resize_mask2.astype(np.float32)),
            "valid_hw1": torch.tensor(shape1, dtype=torch.int64),
            "valid_hw2": torch.tensor(shape2, dtype=torch.int64),
            "overlap_box1": torch.from_numpy(overlap_box1.astype(np.float32)),
            "overlap_box2": torch.from_numpy(overlap_box2.astype(np.float32)),
            "file_name": file_name,
            "overlap_valid": True,
        }

        if self.with_mask and item.get("mask_path1") and item.get("mask_path2"):
            mask1 = cv2.imread(self._resolve_path(item["mask_path1"]), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(self._resolve_path(item["mask_path2"]), cv2.IMREAD_GRAYSCALE)
            if mask1 is not None and mask2 is not None:
                mask1 = cv2.resize(mask1, (shape1[1], shape1[0]), interpolation=cv2.INTER_NEAREST)
                mask2 = cv2.resize(mask2, (shape2[1], shape2[0]), interpolation=cv2.INTER_NEAREST)
                padded_mask1 = np.zeros(image1.shape[:2], dtype=np.uint8)
                padded_mask2 = np.zeros(image2.shape[:2], dtype=np.uint8)
                padded_mask1[:shape1[0], :shape1[1]] = mask1
                padded_mask2[:shape2[0], :shape2[1]] = mask2
                output["gt_mask1"] = torch.from_numpy((padded_mask1 > 0).astype(np.uint8))
                output["gt_mask2"] = torch.from_numpy((padded_mask2 > 0).astype(np.uint8))

        return output

