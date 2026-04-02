#!/usr/bin/env python
"""
Semantic Feature Extraction and Guided Fusion Module.

Provides scale-invariant semantic priors for overlap estimation by:
1. Extracting high-level semantic features from a frozen pretrained backbone
2. Fusing them with appearance features via gated cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SemanticExtractor(nn.Module):
    """Extract semantic features using a frozen pretrained backbone.

    A separate frozen backbone provides stable, scale-invariant semantic
    representations that complement the trainable appearance features.
    The frozen weights ensure semantic features serve as a consistent prior
    rather than drifting during training.
    """

    def __init__(self, d_model=256, backbone='resnet50', freeze=True):
        super().__init__()
        self.backbone_type = backbone

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            out_dim = 2048
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            out_dim = 512
        else:
            raise ValueError(f'Unsupported semantic backbone: {backbone}')

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        del resnet

        self.proj = nn.Sequential(
            nn.Conv2d(out_dim, d_model, 1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'proj' not in name:
                param.requires_grad = False

    @torch.no_grad()
    def _extract_backbone(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        """
        Args:
            x: [N, 3, H, W] normalized image in CHW format
        Returns:
            semantic features: [N, d_model, H/32, W/32]
        """
        feat = self._extract_backbone(x)
        return self.proj(feat)


class SemanticGuidedFusion(nn.Module):
    """Gated cross-attention fusion of appearance and semantic features.

    Appearance features (Q) attend to semantic features (K, V) through
    multi-head cross-attention. A learned gating mechanism controls fusion
    strength per channel, preventing semantic features from overwhelming
    appearance details needed for precise localization.
    """

    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.norm_app = nn.LayerNorm(d_model)
        self.norm_sem = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self._init_residual_paths()

    def _init_residual_paths(self):
        """Zero-init residual output layers for stable no-op startup.

        At init: out_proj produces zeros → sem_ctx=0 → fused=app_seq,
        and FFN's last layer produces zeros → FFN residual=0.
        The module starts as identity and gradually activates as it learns.
        """
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, app_feat, sem_feat):
        """
        Args:
            app_feat: [N, C, H, W] appearance features from FPN
            sem_feat: [N, C, Hs, Ws] semantic features (may differ spatially)
        Returns:
            fused: [N, C, H, W]
        """
        N, C, H, W = app_feat.shape

        if sem_feat.shape[2:] != (H, W):
            sem_feat = F.interpolate(
                sem_feat, (H, W), mode='bilinear', align_corners=False,
            )

        app_seq = app_feat.flatten(2).permute(0, 2, 1)
        sem_seq = sem_feat.flatten(2).permute(0, 2, 1)

        q = self.q_proj(self.norm_app(app_seq))
        k = self.k_proj(self.norm_sem(sem_seq))
        v = self.v_proj(self.norm_sem(sem_seq))

        q = q.view(N, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(N, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(N, -1, self.nhead, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        sem_ctx = (attn @ v).transpose(1, 2).reshape(N, -1, C)
        sem_ctx = self.out_proj(sem_ctx)

        g = self.gate(torch.cat([app_seq, sem_ctx], dim=-1))
        fused = app_seq + g * sem_ctx

        fused = fused + self.ffn(self.norm_ffn(fused))

        return fused.permute(0, 2, 1).view(N, C, H, W)
