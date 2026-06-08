#!/usr/bin/env python
"""
@File    :   trainner.py
@Time    :   2021/06/29 19:21:04
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from kornia.utils import create_meshgrid

from .losses.losses import CycleOverlapLoss, IouOverlapLoss, MaskOverlapLoss
from .losses.utils import bbox_oiou, bbox_overlaps
from .models.backbone import (FeaturePyramidNetwork, MultiScaleResnetEncoder,
                              PatchMerging)
from .models.semantic_fusion import DINOv2SemanticExtractor, SemanticGuidedFusion
from .models.transformer import QueryTransformer
from .models.utils import (PositionEncodingSine, ScaleAdaptivePositionEncoding,
                           box_tlbr_to_xyxy, box_xyxy_to_cxywh)

INF = 1e9


def MLP(channels, do_bn=True):
    """Multi-layer perceptron."""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OETR(nn.Module):
    """Overlap Estimation TRansformer for large-scale-difference
    scenarios.

    Three architectural components target extreme scale differences:

    1. Multi-Scale Feature Pyramid: extracts features from ResNet layer2/3/4
       and fuses them via FPN, enabling cross-scale feature alignment.
    2. Semantic Feature Fusion: a frozen pretrained backbone provides
       scale-invariant semantic priors, fused via gated cross-attention.
    3. Scale-Adaptive Position Encoding: estimates the inter-image scale ratio
       and modulates PE frequencies so cross-attention can better associate
       features across different scales.
    """

    def __init__(self, cfg):
        super().__init__()
        enhanced_cfg = cfg.ENHANCED

        # --- Multi-scale backbone + FPN ---
        self.backbone = MultiScaleResnetEncoder(cfg)
        in_channels = list(self.backbone.layer_channels.values())
        self.d_model = enhanced_cfg.FPN_DIM
        self.fpn = FeaturePyramidNetwork(in_channels, self.d_model)

        # --- Semantic feature fusion ---
        # Two sources for the semantic prior:
        #   'layer4': reuse the trainable backbone's layer4 (detached) as a
        #             cheap semantic prior. Suitable when the test-time
        #             domain matches ImageNet (mostly ground-level scenes).
        #   'dinov2': use a frozen DINOv2 ViT as a viewpoint- and
        #             scale-invariant semantic prior. Strongly preferred for
        #             cross-view air-ground (UAV-UGV) matching.
        self.use_semantic = enhanced_cfg.SEMANTIC_ENABLE
        self.semantic_source = getattr(
            enhanced_cfg, 'SEMANTIC_SOURCE', 'layer4',
        )
        if self.use_semantic:
            if self.semantic_source == 'layer4':
                sem_in_ch = self.backbone.layer_channels['layer4']
                self.semantic_proj = nn.Sequential(
                    nn.Conv2d(sem_in_ch, self.d_model, 1, bias=False),
                    nn.GroupNorm(32, self.d_model),
                    nn.ReLU(inplace=True),
                )
            elif self.semantic_source == 'dinov2':
                dino_name = getattr(
                    enhanced_cfg, 'DINOV2_MODEL', 'dinov2_vitb14',
                )
                self.semantic_extractor = DINOv2SemanticExtractor(
                    d_model=self.d_model, model_name=dino_name,
                )
            else:
                raise ValueError(
                    f'Unsupported SEMANTIC_SOURCE: {self.semantic_source}. '
                    f"Choose 'layer4' or 'dinov2'."
                )
            self.semantic_fusion = SemanticGuidedFusion(self.d_model, nhead=8)

        # --- PatchMerging (stride-16 ??stride-32) ---
        self.patchmerging = PatchMerging(
            (20, 20),
            self.d_model,
            norm_layer=nn.LayerNorm,
            patch_size=[4, 8, 16],
        )
        self.input_proj = nn.Conv2d(
            self.d_model * 2, self.d_model, kernel_size=1,
        )

        # --- Scale-adaptive position encoding ---
        self.use_scale_pe = enhanced_cfg.SCALE_PE
        if self.use_scale_pe:
            self.pos_encoding = ScaleAdaptivePositionEncoding(
                self.d_model, max_shape=cfg.NECK.MAX_SHAPE,
            )
        else:
            self.pos_encoding = PositionEncodingSine(
                self.d_model, max_shape=cfg.NECK.MAX_SHAPE,
            )

        # --- Transformer ---
        num_queries = 1
        self.query_embed1 = nn.Embedding(num_queries, self.d_model)
        self.query_embed2 = nn.Embedding(num_queries, self.d_model)
        self.transformer = QueryTransformer(
            self.d_model, nhead=8, num_layers=4,
        )

        # --- Regression heads ---
        self.tlbr_reg = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, False),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4),
        )

        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(
                self.d_model, self.d_model, (3, 3),
                padding=(1, 1), stride=(1, 1), bias=True,
            ),
            nn.GroupNorm(32, self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 1, (1, 1)),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.d_model * 2 + 1, self.d_model, 3, padding=1, bias=False),
            nn.GroupNorm(32, self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.d_model, self.d_model // 2, 3, padding=1, bias=False,
            ),
            nn.GroupNorm(32, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 2, 1, (1, 1)),
        )
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

        # --- Loss ---
        self.iouloss = IouOverlapLoss(reduction='mean', oiou=cfg.LOSS.OIOU)
        self.mask_loss = MaskOverlapLoss(
            bce_weight=getattr(cfg.LOSS, 'MASK_BCE_WEIGHT', 1.0),
            dice_weight=getattr(cfg.LOSS, 'MASK_DICE_WEIGHT', 1.0),
            pos_weight_max=getattr(cfg.LOSS, 'MASK_POS_WEIGHT_MAX', 1.0),
        )
        self.cycle_loss = CycleOverlapLoss()
        self.sem_loss_weight = enhanced_cfg.SEM_LOSS_WEIGHT
        self.semantic_loss_mode = getattr(
            enhanced_cfg, 'SEM_LOSS_MODE', 'bbox_consistency',
        )
        self.semantic_bg_weight = getattr(
            enhanced_cfg, 'SEM_BG_WEIGHT', 0.0,
        )
        self.semantic_bg_margin = getattr(
            enhanced_cfg, 'SEM_BG_MARGIN', 0.0,
        )
        self.scale_loss_weight = getattr(
            enhanced_cfg, 'SCALE_LOSS_WEIGHT', 0.0,
        )
        self.mask_loss_weight = getattr(
            cfg.LOSS, 'MASK_LOSS_WEIGHT', getattr(cfg.LOSS, 'MASK_AUX_WEIGHT', 0.0),
        )
        self.mask_use_gt_box_prior = getattr(
            cfg.LOSS, 'MASK_USE_GT_BOX_PRIOR', False,
        )
        self.mask_prior_input_weight = getattr(
            cfg.LOSS, 'MASK_PRIOR_INPUT_WEIGHT', 1.0,
        )
        self.mask_prior_logit_weight = getattr(
            cfg.LOSS, 'MASK_PRIOR_LOGIT_WEIGHT', 1.0,
        )

        # --- Hyperparameters ---
        self.max_shape = cfg.NECK.MAX_SHAPE
        self.cycle = cfg.LOSS.CYCLE_OVERLAP
        self.softmax_temperature = 1

    def feature_extraction(self, image1, image2, mask1=None, mask2=None):
        """Enhanced feature extraction with FPN + semantic fusion.

        Pipeline:
          1. Multi-scale backbone ??layer2/3/4 features
          2. FPN fuses the hierarchy ??stride-16 features
          3. PatchMerging ??stride-32 features
          4. Semantic branch: either backbone layer4 (detached) or a frozen
             DINOv2 ViT, projected and fused via gated cross-attention
          5. Scale-adaptive position encoding
        """
        ms_feats1 = self.backbone(image1)
        ms_feats2 = self.backbone(image2)

        fpn_feat1 = self.fpn([
            ms_feats1['layer2'], ms_feats1['layer3'], ms_feats1['layer4'],
        ])
        fpn_feat2 = self.fpn([
            ms_feats2['layer2'], ms_feats2['layer3'], ms_feats2['layer4'],
        ])

        feat1 = self.patchmerging(fpn_feat1)
        feat2 = self.patchmerging(fpn_feat2)
        feat1 = self.input_proj(feat1)
        feat2 = self.input_proj(feat2)

        sem_feat1, sem_feat2 = None, None
        if self.use_semantic:
            if self.semantic_source == 'layer4':
                sem_feat1 = self.semantic_proj(ms_feats1['layer4'].detach())
                sem_feat2 = self.semantic_proj(ms_feats2['layer4'].detach())
            else:  # 'dinov2'
                sem_feat1 = self.semantic_extractor(image1)
                sem_feat2 = self.semantic_extractor(image2)
            feat1 = self.semantic_fusion(feat1, sem_feat1)
            feat2 = self.semantic_fusion(feat2, sem_feat2)

        hf1, wf1 = feat1.shape[2:]
        hf2, wf2 = feat2.shape[2:]

        if self.use_scale_pe:
            pos1, pos2, log_scale = self.pos_encoding(feat1, feat2)
        else:
            pos1 = self.pos_encoding(feat1)
            pos2 = self.pos_encoding(feat2)
            log_scale = None

        return (feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2,
                sem_feat1, sem_feat2, log_scale, fpn_feat1, fpn_feat2)

    def feature_correlation(self, feat1, feat2, pos1, pos2, mask1, mask2):
        hs1, hs2, memory1, memory2 = self.transformer(
            feat1, feat2,
            self.query_embed1.weight, self.query_embed2.weight,
            pos1, pos2, mask1, mask2,
        )
        return hs1, hs2, memory1, memory2

    def generate_mesh_grid(self, feat_hw, stride, device='cpu'):
        coord_xy_map = (
            create_meshgrid(feat_hw[0], feat_hw[1], False, device) + 0.5
        ) * stride
        return coord_xy_map.reshape(1, feat_hw[0] * feat_hw[1], 2)

    def center_estimation(self, hs1, hs2, memory1, memory2,
                          hf1, wf1, hf2, wf2, mask1, mask2):
        att1 = torch.einsum('blc, bnc->bln', memory1, hs1)
        att2 = torch.einsum('blc, bnc->bln', memory2, hs2)

        heatmap1 = rearrange(
            memory1 * att1, 'n (h w) c -> n c h w', h=hf1, w=wf1,
        )
        heatmap2 = rearrange(
            memory2 * att2, 'n (h w) c -> n c h w', h=hf2, w=wf2,
        )
        heatmap1_flatten = (
            rearrange(self.heatmap_conv(heatmap1), 'n c h w -> n (h w) c')
            * self.softmax_temperature
        )
        heatmap2_flatten = (
            rearrange(self.heatmap_conv(heatmap2), 'n c h w -> n (h w) c')
            * self.softmax_temperature
        )

        if mask1 is not None:
            heatmap1_flatten.masked_fill_(
                ~mask1.flatten(1)[..., None].bool(), -INF,
            )
            heatmap2_flatten.masked_fill_(
                ~mask2.flatten(1)[..., None].bool(), -INF,
            )

        prob_map1 = F.softmax(heatmap1_flatten, dim=1)
        prob_map2 = F.softmax(heatmap2_flatten, dim=1)
        coord_xy_map1 = self.generate_mesh_grid(
            (hf1, wf1), stride=self.h1 // hf1, device=memory1.device,
        )
        coord_xy_map2 = self.generate_mesh_grid(
            (hf2, wf2), stride=self.h2 // hf2, device=memory2.device,
        )

        box_cxy1 = (prob_map1 * coord_xy_map1).sum(1)
        box_cxy2 = (prob_map2 * coord_xy_map2).sum(1)

        return box_cxy1, box_cxy2

    def size_regression(self, hs1, hs2):
        tlbr1 = self.tlbr_reg(hs1).sigmoid().squeeze(1)
        tlbr2 = self.tlbr_reg(hs2).sigmoid().squeeze(1)
        return tlbr1, tlbr2

    def obtain_overlap_bbox(self, box_cxy1, tlbr1, box_cxy2, tlbr2):
        pred_bbox_xyxy1 = torch.stack([
            box_cxy1[:, 0] - tlbr1[:, 1] * self.w1,
            box_cxy1[:, 1] - tlbr1[:, 0] * self.h1,
            box_cxy1[:, 0] + tlbr1[:, 3] * self.w1,
            box_cxy1[:, 1] + tlbr1[:, 2] * self.h1,
        ], dim=1)
        pred_bbox_xyxy2 = torch.stack([
            box_cxy2[:, 0] - tlbr2[:, 1] * self.w2,
            box_cxy2[:, 1] - tlbr2[:, 0] * self.h2,
            box_cxy2[:, 0] + tlbr2[:, 3] * self.w2,
            box_cxy2[:, 1] + tlbr2[:, 2] * self.h2,
        ], dim=1)
        pred_bbox_cxywh1 = torch.cat([
            (pred_bbox_xyxy1[:, :2] + pred_bbox_xyxy1[:, 2:]) / 2,
            pred_bbox_xyxy1[:, 2:] - pred_bbox_xyxy1[:, :2],
        ], dim=-1)
        pred_bbox_cxywh2 = torch.cat([
            (pred_bbox_xyxy2[:, :2] + pred_bbox_xyxy2[:, 2:]) / 2,
            pred_bbox_xyxy2[:, 2:] - pred_bbox_xyxy2[:, :2],
        ], dim=-1)
        return (pred_bbox_xyxy1, pred_bbox_xyxy2,
                pred_bbox_cxywh1, pred_bbox_cxywh2)

    def _soft_bbox_mask(self, bbox, H, W):
        """Differentiable soft mask from bounding box via sigmoid boundaries.

        tau controls the sharpness of the boundary. On a 20x20 feature map,
        tau=2.0 gives a ~1-pixel-wide transition band, ensuring gradients
        propagate meaningfully across the entire bbox region rather than
        only at the exact boundary pixels (which happens with tau=10).
        """
        N = bbox.shape[0]
        device = bbox.device
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy = y.view(1, 1, H, 1).expand(N, 1, H, W)
        xx = x.view(1, 1, 1, W).expand(N, 1, H, W)

        x1 = bbox[:, 0].view(N, 1, 1, 1)
        y1 = bbox[:, 1].view(N, 1, 1, 1)
        x2 = bbox[:, 2].view(N, 1, 1, 1)
        y2 = bbox[:, 3].view(N, 1, 1, 1)

        tau = 2.0
        mask = (torch.sigmoid(tau * (xx - x1)) * torch.sigmoid(tau * (x2 - xx))
                * torch.sigmoid(tau * (yy - y1)) * torch.sigmoid(tau * (y2 - yy)))
        return mask

    def _resize_soft_mask(self, mask, size):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        if mask.shape[-2:] != size:
            mask = F.interpolate(
                mask, size=size, mode='bilinear', align_corners=False,
            )
        return mask.clamp_(0.0, 1.0)

    def _masked_semantic_pool(self, sem_feat, soft_mask, valid_mask=None,
                              min_mass=1e-4):
        if soft_mask is None:
            return None, None
        if valid_mask is not None:
            soft_mask = soft_mask * valid_mask.float()
        mass = soft_mask.sum(dim=[2, 3]).squeeze(1)
        pooled = (
            (sem_feat * soft_mask).sum(dim=[2, 3])
            / mass.clamp(min=min_mass).unsqueeze(1)
        )
        return pooled, mass > min_mass

    def build_valid_image_mask(self, valid_hw, image_hw, device):
        if valid_hw is None:
            return None
        h, w = image_hw
        mask = torch.zeros(
            (valid_hw.shape[0], 1, h, w), dtype=torch.bool, device=device,
        )
        for i, hw in enumerate(valid_hw):
            valid_h = int(hw[0].item())
            valid_w = int(hw[1].item())
            mask[i, :, :valid_h, :valid_w] = True
        return mask

    def compute_mask_iou(self, pred_mask, gt_mask, valid_mask=None, threshold=0.5):
        pred_mask = (pred_mask >= threshold).bool()
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        gt_mask = (gt_mask > 0).bool()
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            pred_mask = pred_mask & valid_mask
            gt_mask = gt_mask & valid_mask

        intersection = (pred_mask & gt_mask).flatten(1).sum(dim=1).float()
        union = (pred_mask | gt_mask).flatten(1).sum(dim=1).float()
        empty = union == 0
        return torch.where(
            empty, torch.ones_like(union), intersection / union.clamp(min=1.0),
        ).mean()

    def predict_masks(self, fpn_feat1, fpn_feat2, memory1, memory2,
                      hf1, wf1, hf2, wf2, pred_bbox_xyxy1, pred_bbox_xyxy2,
                      image_hw1, image_hw2, valid_hw1=None, valid_hw2=None):
        memory_map1 = rearrange(memory1, 'n (h w) c -> n c h w', h=hf1, w=wf1)
        memory_map2 = rearrange(memory2, 'n (h w) c -> n c h w', h=hf2, w=wf2)

        mask_feat1 = F.interpolate(
            memory_map1, size=fpn_feat1.shape[2:], mode='bilinear',
            align_corners=False,
        )
        mask_feat2 = F.interpolate(
            memory_map2, size=fpn_feat2.shape[2:], mode='bilinear',
            align_corners=False,
        )

        mh1, mw1 = fpn_feat1.shape[2:]
        mh2, mw2 = fpn_feat2.shape[2:]
        h1, w1 = image_hw1
        h2, w2 = image_hw2

        scale1 = pred_bbox_xyxy1.new_tensor([mw1 / w1, mh1 / h1, mw1 / w1, mh1 / h1])
        scale2 = pred_bbox_xyxy2.new_tensor([mw2 / w2, mh2 / h2, mw2 / w2, mh2 / h2])
        bbox_prior1 = self._soft_bbox_mask(pred_bbox_xyxy1 * scale1, mh1, mw1)
        bbox_prior2 = self._soft_bbox_mask(pred_bbox_xyxy2 * scale2, mh2, mw2)
        mask_prior1 = bbox_prior1 * self.mask_prior_input_weight
        mask_prior2 = bbox_prior2 * self.mask_prior_input_weight

        mask_logits1 = self.mask_head(
            torch.cat([fpn_feat1, mask_feat1, mask_prior1], dim=1),
        )
        mask_logits2 = self.mask_head(
            torch.cat([fpn_feat2, mask_feat2, mask_prior2], dim=1),
        )

        prior_img1 = F.interpolate(
            bbox_prior1, size=(h1, w1), mode='bilinear', align_corners=False,
        ).clamp(min=1e-4, max=1.0 - 1e-4)
        prior_img2 = F.interpolate(
            bbox_prior2, size=(h2, w2), mode='bilinear', align_corners=False,
        ).clamp(min=1e-4, max=1.0 - 1e-4)
        mask_logits1 = F.interpolate(
            mask_logits1, size=(h1, w1), mode='bilinear', align_corners=False,
        ) + self.mask_prior_logit_weight * torch.log(
            prior_img1 / (1.0 - prior_img1),
        )
        mask_logits2 = F.interpolate(
            mask_logits2, size=(h2, w2), mode='bilinear', align_corners=False,
        ) + self.mask_prior_logit_weight * torch.log(
            prior_img2 / (1.0 - prior_img2),
        )

        valid_mask1 = self.build_valid_image_mask(
            valid_hw1, (h1, w1), mask_logits1.device,
        )
        valid_mask2 = self.build_valid_image_mask(
            valid_hw2, (h2, w2), mask_logits2.device,
        )
        if valid_mask1 is not None:
            mask_logits1 = mask_logits1.masked_fill(~valid_mask1, -12.0)
        if valid_mask2 is not None:
            mask_logits2 = mask_logits2.masked_fill(~valid_mask2, -12.0)

        pred_mask1 = torch.sigmoid(mask_logits1)
        pred_mask2 = torch.sigmoid(mask_logits2)
        return (
            mask_logits1, mask_logits2, pred_mask1, pred_mask2,
            valid_mask1, valid_mask2,
        )

    def _semantic_bbox_consistency_loss(self, sem1, sem2, pred_bbox1, pred_bbox2,
                                        h1, w1, h2, w2):
        if sem1 is None or sem2 is None:
            zero = pred_bbox1.new_zeros(())
            return zero, {'sem_fg_loss': zero, 'sem_bg_loss': zero}

        sem1 = sem1.detach()
        sem2 = sem2.detach()

        _, _, sH1, sW1 = sem1.shape
        _, _, sH2, sW2 = sem2.shape

        sx1, sy1 = sW1 / w1, sH1 / h1
        scaled_bbox1 = pred_bbox1 * pred_bbox1.new_tensor([sx1, sy1, sx1, sy1])
        sx2, sy2 = sW2 / w2, sH2 / h2
        scaled_bbox2 = pred_bbox2 * pred_bbox2.new_tensor([sx2, sy2, sx2, sy2])

        mask1 = self._soft_bbox_mask(scaled_bbox1, sH1, sW1)
        mask2 = self._soft_bbox_mask(scaled_bbox2, sH2, sW2)

        pooled1 = (sem1 * mask1).sum(dim=[2, 3]) / (mask1.sum(dim=[2, 3]) + 1e-6)
        pooled2 = (sem2 * mask2).sum(dim=[2, 3]) / (mask2.sum(dim=[2, 3]) + 1e-6)

        fg_loss = (1.0 - F.cosine_similarity(pooled1, pooled2, dim=-1)).mean()
        zero = fg_loss.new_zeros(())
        return fg_loss, {'sem_fg_loss': fg_loss, 'sem_bg_loss': zero}

    def _semantic_mask_alignment_loss(self, sem1, sem2,
                                      pred_mask1, pred_mask2,
                                      gt_mask1, gt_mask2,
                                      valid_mask1=None, valid_mask2=None):
        if (
            sem1 is None or sem2 is None
            or pred_mask1 is None or pred_mask2 is None
            or gt_mask1 is None or gt_mask2 is None
        ):
            zero_ref = next(
                item for item in (
                    sem1, sem2, pred_mask1, pred_mask2, gt_mask1, gt_mask2,
                ) if item is not None
            )
            zero = zero_ref.new_zeros(())
            return zero, {'sem_fg_loss': zero, 'sem_bg_loss': zero}

        sem1 = sem1.detach()
        sem2 = sem2.detach()

        sem_hw1 = sem1.shape[-2:]
        sem_hw2 = sem2.shape[-2:]

        pred_mask1 = self._resize_soft_mask(pred_mask1, sem_hw1).clamp_(min=1e-4)
        pred_mask2 = self._resize_soft_mask(pred_mask2, sem_hw2).clamp_(min=1e-4)
        gt_mask1 = self._resize_soft_mask(gt_mask1, sem_hw1)
        gt_mask2 = self._resize_soft_mask(gt_mask2, sem_hw2)
        valid_mask1 = self._resize_soft_mask(valid_mask1, sem_hw1)
        valid_mask2 = self._resize_soft_mask(valid_mask2, sem_hw2)

        pred_proto1, pred_ok1 = self._masked_semantic_pool(
            sem1, pred_mask1, valid_mask1,
        )
        pred_proto2, pred_ok2 = self._masked_semantic_pool(
            sem2, pred_mask2, valid_mask2,
        )
        gt_proto1, gt_ok1 = self._masked_semantic_pool(
            sem1, gt_mask1, valid_mask1,
        )
        gt_proto2, gt_ok2 = self._masked_semantic_pool(
            sem2, gt_mask2, valid_mask2,
        )

        fg_sims = []
        if pred_proto1 is not None:
            fg_ok1 = pred_ok1 & gt_ok1
            if fg_ok1.any():
                fg_sims.append(F.cosine_similarity(
                    pred_proto1[fg_ok1], gt_proto1[fg_ok1], dim=-1,
                ))
        if pred_proto2 is not None:
            fg_ok2 = pred_ok2 & gt_ok2
            if fg_ok2.any():
                fg_sims.append(F.cosine_similarity(
                    pred_proto2[fg_ok2], gt_proto2[fg_ok2], dim=-1,
                ))

        if fg_sims:
            fg_sim = torch.cat(fg_sims, dim=0)
            fg_loss = (1.0 - fg_sim).mean()
        else:
            zero = sem1.new_zeros(())
            return zero, {'sem_fg_loss': zero, 'sem_bg_loss': zero}

        bg_loss = sem1.new_zeros(())
        if self.semantic_bg_weight > 0:
            bg_terms = []
            if valid_mask1 is None:
                bg_mask1 = 1.0 - gt_mask1
            else:
                bg_mask1 = (valid_mask1 - gt_mask1).clamp_(min=0.0, max=1.0)
            if valid_mask2 is None:
                bg_mask2 = 1.0 - gt_mask2
            else:
                bg_mask2 = (valid_mask2 - gt_mask2).clamp_(min=0.0, max=1.0)

            bg_proto1, bg_ok1 = self._masked_semantic_pool(sem1, bg_mask1)
            bg_proto2, bg_ok2 = self._masked_semantic_pool(sem2, bg_mask2)

            if pred_proto1 is not None:
                margin_ok1 = pred_ok1 & gt_ok1 & bg_ok1
                if margin_ok1.any():
                    fg_sim1 = F.cosine_similarity(
                        pred_proto1[margin_ok1], gt_proto1[margin_ok1], dim=-1,
                    )
                    bg_sim1 = F.cosine_similarity(
                        pred_proto1[margin_ok1], bg_proto1[margin_ok1], dim=-1,
                    )
                    bg_terms.append(F.relu(
                        self.semantic_bg_margin - fg_sim1 + bg_sim1,
                    ))
            if pred_proto2 is not None:
                margin_ok2 = pred_ok2 & gt_ok2 & bg_ok2
                if margin_ok2.any():
                    fg_sim2 = F.cosine_similarity(
                        pred_proto2[margin_ok2], gt_proto2[margin_ok2], dim=-1,
                    )
                    bg_sim2 = F.cosine_similarity(
                        pred_proto2[margin_ok2], bg_proto2[margin_ok2], dim=-1,
                    )
                    bg_terms.append(F.relu(
                        self.semantic_bg_margin - fg_sim2 + bg_sim2,
                    ))

            if bg_terms:
                bg_loss = torch.cat(bg_terms, dim=0).mean()

        total = fg_loss + self.semantic_bg_weight * bg_loss
        return total, {'sem_fg_loss': fg_loss, 'sem_bg_loss': bg_loss}

    def semantic_consistency_loss(self, sem1, sem2, pred_bbox1, pred_bbox2,
                                  h1, w1, h2, w2,
                                  pred_mask1=None, pred_mask2=None,
                                  gt_mask1=None, gt_mask2=None,
                                  valid_mask1=None, valid_mask2=None):
        """Semantic guidance for overlap prediction.

        Two modes are supported:
          * bbox_consistency: legacy cross-view prototype alignment inside the
            predicted bbox.
          * mask_align: align the predicted mask's semantic prototype to the
            GT overlap mask within each image, with an optional background
            margin term.
        """
        if self.semantic_loss_mode == 'mask_align':
            return self._semantic_mask_alignment_loss(
                sem1, sem2,
                pred_mask1, pred_mask2,
                gt_mask1, gt_mask2,
                valid_mask1=valid_mask1, valid_mask2=valid_mask2,
            )
        return self._semantic_bbox_consistency_loss(
            sem1, sem2, pred_bbox1, pred_bbox2, h1, w1, h2, w2,
        )

    def scale_consistency_loss(self, log_scale,
                               pred_bbox1, pred_bbox2,
                               gt_bbox1, gt_bbox2):
        """Geometric consistency between predicted scale and bbox geometry.

        For an overlap region of physical size L x M observed by two cameras
        with ground sample distances (GSD) s1 and s2, the projected bbox
        areas satisfy:

            area_k = L * M / s_k^2
            => sqrt(area_2 / area_1) = s_1 / s_2 = inter-view scale ratio

        So 0.5 * log(area_2 / area_1) is exactly the log scale ratio implied
        by the GT bboxes. This loss enforces two things:

        1. The ScaleAdaptivePositionEncoding's ``log_scale`` estimate, which
           is otherwise only supervised implicitly via downstream losses,
           is now directly matched to the GT scale ratio. This converts
           ``scale_net`` from an unsupervised auxiliary head into a properly
           supervised regressor and speeds up its convergence considerably.

        2. The predicted bbox area ratio is forced to match the GT bbox
           area ratio. This is a scale-invariant complement to ``wh_loss``
           (which penalises absolute width / height) and is particularly
           useful in the early stages of training when individual w / h
           predictions are noisy but their ratio can still be correct.

        Args:
            log_scale: [N, 1] or [N], estimate from
                ``ScaleAdaptivePositionEncoding`` (may be None when
                ``SCALE_PE=False``).
            pred_bbox1, pred_bbox2: [N, 4] predicted overlap bboxes (xyxy)
                for image 1 and 2.
            gt_bbox1, gt_bbox2: [N, 4] ground-truth overlap bboxes (xyxy).
        """
        eps = 1e-6

        def log_area(bbox):
            w = (bbox[:, 2] - bbox[:, 0]).clamp(min=1.0)
            h = (bbox[:, 3] - bbox[:, 1]).clamp(min=1.0)
            return torch.log(w * h + eps)

        gt_log_scale = 0.5 * (log_area(gt_bbox2) - log_area(gt_bbox1))

        loss = pred_bbox1.new_zeros(())

        if log_scale is not None:
            est = log_scale.view(-1)
            loss = loss + F.l1_loss(est, gt_log_scale)

        pred_log_scale = 0.5 * (log_area(pred_bbox2) - log_area(pred_bbox1))
        loss = loss + 0.5 * F.l1_loss(pred_log_scale, gt_log_scale)

        return loss

    def forward_dummy(self, image1, image2, mask1=None, mask2=None,
                      return_masks=False, valid_hw1=None, valid_hw2=None):
        """Inference pipeline."""
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2

        (feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2,
         _, _, _, fpn_feat1, fpn_feat2) = self.feature_extraction(
            image1, image2, mask1, mask2,
        )

        hs1, hs2, memory1, memory2 = self.feature_correlation(
            feat1, feat2, pos1, pos2, mask1, mask2,
        )

        box_cxy1, box_cxy2 = self.center_estimation(
            hs1, hs2, memory1, memory2, hf1, wf1, hf2, wf2, mask1, mask2,
        )
        tlbr1, tlbr2 = self.size_regression(hs1, hs2)

        pred_bbox_xyxy1 = box_tlbr_to_xyxy(box_cxy1, tlbr1, max_h=h1, max_w=w1)
        pred_bbox_xyxy2 = box_tlbr_to_xyxy(box_cxy2, tlbr2, max_h=h2, max_w=w2)

        if return_masks:
            (_, _, pred_mask1, pred_mask2,
             _, _) = self.predict_masks(
                fpn_feat1, fpn_feat2,
                memory1, memory2,
                hf1, wf1, hf2, wf2,
                pred_bbox_xyxy1, pred_bbox_xyxy2,
                (h1, w1), (h2, w2),
                valid_hw1=valid_hw1, valid_hw2=valid_hw2,
            )
            return pred_bbox_xyxy1, pred_bbox_xyxy2, pred_mask1, pred_mask2

        return pred_bbox_xyxy1, pred_bbox_xyxy2

    def forward(self, data, validation=False):
        """Training pipeline."""
        if 'resize_mask1' in data:
            mask1 = data['resize_mask1'][data['overlap_valid']]
            mask2 = data['resize_mask2'][data['overlap_valid']]
        else:
            mask1, mask2 = None, None
        valid_hw1 = (
            data['valid_hw1'][data['overlap_valid']]
            if 'valid_hw1' in data else None
        )
        valid_hw2 = (
            data['valid_hw2'][data['overlap_valid']]
            if 'valid_hw2' in data else None
        )

        h1, w1 = data['image1'][data['overlap_valid']].shape[1:3]
        h2, w2 = data['image2'][data['overlap_valid']].shape[1:3]
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2

        (feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2,
         sem_feat1, sem_feat2, log_scale,
         fpn_feat1, fpn_feat2) = self.feature_extraction(
            data['image1'][data['overlap_valid']],
            data['image2'][data['overlap_valid']],
            mask1, mask2,
        )

        hs1, hs2, memory1, memory2 = self.feature_correlation(
            feat1, feat2, pos1, pos2, mask1, mask2,
        )

        box_cxy1, box_cxy2 = self.center_estimation(
            hs1, hs2, memory1, memory2, hf1, wf1, hf2, wf2, mask1, mask2,
        )
        tlbr1, tlbr2 = self.size_regression(hs1, hs2)

        (pred_bbox_xyxy1, pred_bbox_xyxy2,
         pred_bbox_cxywh1, pred_bbox_cxywh2) = self.obtain_overlap_bbox(
            box_cxy1, tlbr1, box_cxy2, tlbr2,
        )

        # --- Ground truth ---
        gt_bbox_xyxy1 = data['overlap_box1'][data['overlap_valid']]
        gt_bbox_xyxy2 = data['overlap_box2'][data['overlap_valid']]
        gt_bbox_cxywh1 = box_xyxy_to_cxywh(gt_bbox_xyxy1, max_h=h1, max_w=w1)
        gt_bbox_cxywh2 = box_xyxy_to_cxywh(gt_bbox_xyxy2, max_h=h2, max_w=w2)

        wh_scale1 = torch.tensor([w1, h1], device=data['image1'].device)
        wh_scale2 = torch.tensor([w2, h2], device=data['image2'].device)

        loc_l1_loss = F.l1_loss(
            pred_bbox_cxywh1[:, :2] / wh_scale1,
            gt_bbox_cxywh1[:, :2] / wh_scale1, reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, :2] / wh_scale2,
            gt_bbox_cxywh2[:, :2] / wh_scale2, reduction='mean',
        )

        wh_l1_loss = (F.l1_loss(
            pred_bbox_cxywh1[:, 2:] / wh_scale1,
            gt_bbox_cxywh1[:, 2:] / wh_scale1, reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, 2:] / wh_scale2,
            gt_bbox_cxywh2[:, 2:] / wh_scale2, reduction='mean',
        )) / 2

        iouloss = self.iouloss(
            pred_bbox_xyxy1, gt_bbox_xyxy1, pred_bbox_xyxy2, gt_bbox_xyxy2,
        )

        iou1 = bbox_overlaps(
            pred_bbox_xyxy1,
            data['overlap_box1'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        iou2 = bbox_overlaps(
            pred_bbox_xyxy2,
            data['overlap_box2'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        oiou1 = bbox_oiou(
            data['overlap_box1'][data['overlap_valid']], pred_bbox_xyxy1,
        ).mean()
        oiou2 = bbox_oiou(
            data['overlap_box2'][data['overlap_valid']], pred_bbox_xyxy2,
        ).mean()

        results = {
            'pred_bbox1': pred_bbox_xyxy1,
            'pred_bbox2': pred_bbox_xyxy2,
            'iouloss': iouloss.mean(),
            'wh_loss': wh_l1_loss.mean(),
            'loc_loss': loc_l1_loss.mean(),
            'iou1': iou1,
            'iou2': iou2,
            'oiou1': oiou1,
            'oiou2': oiou2,
        }

        pred_mask1 = pred_mask2 = None
        valid_mask_img1 = valid_mask_img2 = None
        gt_mask1 = gt_mask2 = None
        has_gt_masks = (
            'gt_mask1' in data and 'gt_mask2' in data
            and data['gt_mask1'] is not None and data['gt_mask2'] is not None
        )
        if has_gt_masks or validation or self.mask_loss_weight > 0:
            mask_bbox1 = pred_bbox_xyxy1
            mask_bbox2 = pred_bbox_xyxy2
            if has_gt_masks and self.training and self.mask_use_gt_box_prior:
                mask_bbox1 = gt_bbox_xyxy1.detach()
                mask_bbox2 = gt_bbox_xyxy2.detach()
            (pred_mask_logits1, pred_mask_logits2, pred_mask1, pred_mask2,
             valid_mask_img1, valid_mask_img2) = self.predict_masks(
                fpn_feat1, fpn_feat2,
                memory1, memory2,
                hf1, wf1, hf2, wf2,
                mask_bbox1, mask_bbox2,
                (h1, w1), (h2, w2),
                valid_hw1=valid_hw1, valid_hw2=valid_hw2,
            )
            results['pred_mask1'] = pred_mask1
            results['pred_mask2'] = pred_mask2

            if has_gt_masks:
                gt_mask1 = data['gt_mask1'][data['overlap_valid']].unsqueeze(1)
                gt_mask2 = data['gt_mask2'][data['overlap_valid']].unsqueeze(1)
                results['mask_iou1'] = self.compute_mask_iou(
                    pred_mask1, gt_mask1, valid_mask_img1,
                )
                results['mask_iou2'] = self.compute_mask_iou(
                    pred_mask2, gt_mask2, valid_mask_img2,
                )
                if self.mask_loss_weight > 0:
                    mask_loss1 = self.mask_loss(
                        pred_mask_logits1, gt_mask1, valid_mask_img1,
                    )
                    mask_loss2 = self.mask_loss(
                        pred_mask_logits2, gt_mask2, valid_mask_img2,
                    )
                    results['mask_loss'] = (
                        (mask_loss1 + mask_loss2) / 2.0
                    ) * self.mask_loss_weight

        # --- Semantic consistency loss ---
        if self.use_semantic and self.sem_loss_weight > 0:
            sem_loss, sem_stats = self.semantic_consistency_loss(
                sem_feat1, sem_feat2,
                pred_bbox_xyxy1, pred_bbox_xyxy2,
                h1, w1, h2, w2,
                pred_mask1=pred_mask1, pred_mask2=pred_mask2,
                gt_mask1=gt_mask1, gt_mask2=gt_mask2,
                valid_mask1=valid_mask_img1, valid_mask2=valid_mask_img2,
            )
            results['sem_loss'] = sem_loss * self.sem_loss_weight
            for key, value in sem_stats.items():
                results[key] = value * self.sem_loss_weight

        # --- Scale consistency loss ---
        if self.scale_loss_weight > 0:
            scale_loss = self.scale_consistency_loss(
                log_scale,
                pred_bbox_xyxy1, pred_bbox_xyxy2,
                gt_bbox_xyxy1, gt_bbox_xyxy2,
            )
            results['scale_loss'] = scale_loss * self.scale_loss_weight

        # --- Cycle consistency loss ---
        if self.cycle:
            box_cxy1from2, box_cxy2from1 = self.center_estimation(
                hs2, hs1, memory1, memory2,
                hf1, wf1, hf2, wf2, mask1, mask2,
            )
            (_, _, pred_bbox_cxywh1from2,
             pred_bbox_cxywh2from1) = self.obtain_overlap_bbox(
                box_cxy1from2, tlbr1, box_cxy2from1, tlbr2,
            )
            cycle_loss = F.l1_loss(
                pred_bbox_cxywh1from2[:, :2] / wh_scale1,
                gt_bbox_cxywh1[:, :2] / wh_scale1, reduction='mean',
            ) + F.l1_loss(
                pred_bbox_cxywh2from1[:, :2] / wh_scale2,
                gt_bbox_cxywh2[:, :2] / wh_scale2, reduction='mean',
            )
            results['cycle_loss'] = cycle_loss.mean()

        return results


def build_detectors(cfg):
    """Instantiate the OETR model from an OETR config node."""
    return OETR(cfg)


def load_legacy_oetr_checkpoint(model, checkpoint_path, verbose=True):
    """Warm-start the current OETR model from a legacy (original) OETR checkpoint.

    The original OETR (ResNet-layer3 single-scale + fixed PE) and the current
    OETR (multi-scale + FPN + DINOv2 semantic + scale-adaptive PE) share most
    of their parameters: the ResNet backbone (layer0-3), PatchMerging,
    QueryTransformer, query embeddings, and the regression / heatmap heads.

    This loader remaps legacy state-dict keys onto the current model's keys
    and skips parameters that have no counterpart, so you can fine-tune from
    an existing OETR checkpoint instead of training from scratch.

    Specifically:
        * ``backbone.layer0/1/2/3.*``      -> kept as-is
        * ``backbone.encoder.layer4.*``    -> remapped to ``backbone.layer4.*``
          (legacy ResnetEncoder kept the raw ResNet, including its ImageNet
          layer4, even when only layer3 was used at inference time)
        * ``input_proj2.*``                -> remapped to ``input_proj.*``
          (both are Conv2d(512 -> 256, kernel=1))
        * ``patchmerging.*``               -> kept
        * ``transformer.*``                -> kept
        * ``query_embed1/2.*``             -> kept
        * ``tlbr_reg.*``                   -> kept
        * ``heatmap_conv.*``               -> kept
        * ``input_proj.*`` (Conv2d 1024->256) -> dropped (FPN replaces it)
        * Anything else (e.g. fpn, semantic_*, pos_encoding.scale_*)
          stays at its current (typically zero- or ImageNet-initialised) value.

    Args:
        model: an instance of ``OETR``.
        checkpoint_path: path to a legacy OETR .pth file, or a state_dict
            already loaded into memory.
        verbose: print a summary of loaded / remapped / skipped keys.

    Returns:
        A dict with ``loaded``, ``remapped``, ``dropped``, ``missing``
        and ``unexpected`` key lists, for inspection.
    """
    import torch

    if isinstance(checkpoint_path, dict):
        legacy_sd = checkpoint_path
    else:
        legacy_sd = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in legacy_sd:
        legacy_sd = legacy_sd['state_dict']
    # Strip DataParallel/DDP prefix if present.
    legacy_sd = {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in legacy_sd.items()
    }

    remap = {}
    dropped = []
    for k, v in legacy_sd.items():
        if k.startswith('backbone.encoder.layer4.'):
            remap[k.replace('backbone.encoder.', 'backbone.', 1)] = v
        elif k.startswith('backbone.encoder.'):
            dropped.append(k)  # redundant with backbone.layer*
        elif k.startswith('input_proj2.'):
            remap[k.replace('input_proj2.', 'input_proj.', 1)] = v
        elif k.startswith('input_proj.'):
            dropped.append(k)  # 1024->256, no counterpart (FPN replaces it)
        else:
            remap[k] = v  # backbone.layer0-3, patchmerging, transformer,
            # query_embed*, tlbr_reg, heatmap_conv all keep their key

    # Filter against the current model's state-dict, dropping any remapped
    # keys whose shape doesn't match (defensive against future schema drift).
    current_sd = model.state_dict()
    final_sd = {}
    shape_mismatch = []
    for k, v in remap.items():
        if k in current_sd and current_sd[k].shape == v.shape:
            final_sd[k] = v
        elif k in current_sd:
            shape_mismatch.append(
                (k, tuple(v.shape), tuple(current_sd[k].shape))
            )
        else:
            dropped.append(k)

    missing, unexpected = model.load_state_dict(final_sd, strict=False)

    if verbose:
        print(f'[legacy-load] loaded   : {len(final_sd)} tensors')
        print(f'[legacy-load] dropped  : {len(dropped)} tensors '
              f'(redundant or no counterpart)')
        print(f'[legacy-load] missing  : {len(missing)} tensors '
              f'(new modules, kept at fresh init)')
        print(f'[legacy-load] unexpected: {len(unexpected)} tensors')
        if shape_mismatch:
            print('[legacy-load] shape mismatches (skipped):')
            for name, old_s, new_s in shape_mismatch[:10]:
                print(f'    {name}: legacy {old_s} vs current {new_s}')

    return {
        'loaded': sorted(final_sd.keys()),
        'remapped': sorted(
            k for k in final_sd if k not in legacy_sd  # i.e. renamed
        ),
        'dropped': sorted(dropped),
        'missing': sorted(list(missing)),
        'unexpected': sorted(list(unexpected)),
        'shape_mismatch': shape_mismatch,
    }

