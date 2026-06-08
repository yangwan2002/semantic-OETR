#!/usr/bin/env python3
"""Run OETR overlap bbox+mask inference on an arbitrary image pair."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "baseline"
    / "carla_fullimg_uavrelay_soft_planar_finetune_pe_infer.py"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "weights" / "carla_uavrelay_soft_planar_pe_ep11" / "model_epoch_11.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer OETR overlap bbox and mask for two images.",
    )
    parser.add_argument("--image1", required=True, type=Path, help="Source image (e.g. UAV).")
    parser.add_argument("--image2", required=True, type=Path, help="Target image (e.g. relay).")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="OETR config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for overlays and summary.json. Defaults next to image1.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--image-size", type=int, nargs=2, default=[1280, 720])
    parser.add_argument("--size-divisor", type=int, default=32)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    return parser.parse_args()


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def clip_box(box: Sequence[float], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x1 = int(np.floor(box[0]))
    y1 = int(np.floor(box[1]))
    x2 = int(np.ceil(box[2]))
    y2 = int(np.ceil(box[3]))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return x1, y1, x2, y2


def resize_pad_image(image: np.ndarray, target_wh, size_divisor: int):
    target_w, target_h = int(target_wh[0]), int(target_wh[1])
    h, w = image.shape[:2]
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(w * scale + 0.5))
    new_h = max(1, int(h * scale + 0.5))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = int(math.ceil(float(target_w) / size_divisor) * size_divisor)
    pad_h = int(math.ceil(float(target_h) / size_divisor) * size_divisor)

    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    canvas[:new_h, :new_w, :] = resized.astype(np.float32)

    valid_h = int(math.ceil(float(new_h) / size_divisor))
    valid_w = int(math.ceil(float(new_w) / size_divisor))
    resize_mask = np.zeros((pad_h // size_divisor, pad_w // size_divisor), dtype=np.float32)
    resize_mask[:valid_h, :valid_w] = 1.0
    return canvas / 255.0, resize_mask, scale, (new_h, new_w)


def restore_bbox(bbox_xyxy, scale: float, image_hw):
    h, w = image_hw
    bbox = np.array(bbox_xyxy, dtype=np.float32) / float(scale)
    bbox[0::2] = np.clip(bbox[0::2], 0.0, max(0.0, float(w - 1)))
    bbox[1::2] = np.clip(bbox[1::2], 0.0, max(0.0, float(h - 1)))
    return bbox


def restore_binary_mask(
    pred_mask: np.ndarray,
    valid_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    valid_h, valid_w = valid_hw
    cropped = pred_mask[:valid_h, :valid_w]
    binary = (cropped >= threshold).astype(np.uint8) * 255
    orig_h, orig_w = image_hw
    return cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def overlay_mask_and_box(
    image: np.ndarray,
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = image.copy()
    overlay = np.zeros_like(out)
    overlay[mask > 0] = color
    out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0.0)
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def build_model(config_path: Path, checkpoint_path: Path, device_name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.config.default import get_cfg_defaults
    from src.model import OETR

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(config_path))

    device_str = device_name
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = OETR(cfg.OETR).eval().to(device)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    load_info = model.load_state_dict(state_dict, strict=False)
    return model, device, load_info


@torch.inference_mode()
def predict_pair(
    model,
    device: torch.device,
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    image_size,
    size_divisor: int,
    mask_thresh: float,
) -> Dict:
    inp1, resize_mask1, scale1, valid_hw1 = resize_pad_image(image1, image_size, size_divisor)
    inp2, resize_mask2, scale2, valid_hw2 = resize_pad_image(image2, image_size, size_divisor)

    tensor1 = torch.from_numpy(inp1).unsqueeze(0).float().to(device)
    tensor2 = torch.from_numpy(inp2).unsqueeze(0).float().to(device)
    mask_tensor1 = torch.from_numpy(resize_mask1).unsqueeze(0).float().to(device)
    mask_tensor2 = torch.from_numpy(resize_mask2).unsqueeze(0).float().to(device)
    valid_hw_tensor1 = torch.tensor([valid_hw1], dtype=torch.int64, device=device)
    valid_hw_tensor2 = torch.tensor([valid_hw2], dtype=torch.int64, device=device)

    pred1, pred2, pred_mask1, pred_mask2 = model.forward_dummy(
        tensor1,
        tensor2,
        mask_tensor1,
        mask_tensor2,
        return_masks=True,
        valid_hw1=valid_hw_tensor1,
        valid_hw2=valid_hw_tensor2,
    )

    pred_box1 = restore_bbox(pred1[0].detach().cpu().numpy(), scale1, image1.shape[:2])
    pred_box2 = restore_bbox(pred2[0].detach().cpu().numpy(), scale2, image2.shape[:2])
    pred_mask1 = restore_binary_mask(
        pred_mask1[0, 0].detach().cpu().numpy(),
        valid_hw1,
        image1.shape[:2],
        mask_thresh,
    )
    pred_mask2 = restore_binary_mask(
        pred_mask2[0, 0].detach().cpu().numpy(),
        valid_hw2,
        image2.shape[:2],
        mask_thresh,
    )
    return {
        "pred_box1": clip_box(pred_box1, image1.shape),
        "pred_box2": clip_box(pred_box2, image2.shape),
        "pred_mask1": pred_mask1,
        "pred_mask2": pred_mask2,
    }


def main() -> int:
    args = parse_args()
    image1 = load_bgr(args.image1)
    image2 = load_bgr(args.image2)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.image1.parent / f"{args.image1.stem}_oetr_infer"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device, load_info = build_model(args.config, args.checkpoint, args.device)
    pred = predict_pair(
        model,
        device,
        image1,
        image2,
        image_size=args.image_size,
        size_divisor=args.size_divisor,
        mask_thresh=args.mask_thresh,
    )

    panel1 = overlay_mask_and_box(image1, pred["pred_mask1"], pred["pred_box1"])
    panel2 = overlay_mask_and_box(image2, pred["pred_mask2"], pred["pred_box2"])
    out1 = output_dir / f"{args.image1.stem}_pred.png"
    out2 = output_dir / f"{args.image2.stem}_pred.png"
    cv2.imwrite(str(out1), panel1)
    cv2.imwrite(str(out2), panel2)

    summary = {
        "image1": str(args.image1.resolve()),
        "image2": str(args.image2.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "pred_box1": list(pred["pred_box1"]),
        "pred_box2": list(pred["pred_box2"]),
        "mask_area_ratio1": float((pred["pred_mask1"] > 0).mean()),
        "mask_area_ratio2": float((pred["pred_mask2"] > 0).mean()),
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
        "outputs": {
            "image1_overlay": str(out1.resolve()),
            "image2_overlay": str(out2.resolve()),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
