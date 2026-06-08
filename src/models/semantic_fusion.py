#!/usr/bin/env python
"""
Semantic Feature Extraction and Guided Fusion Module.

Provides scale-invariant semantic priors for overlap estimation by:
1. Extracting high-level semantic features from a frozen pretrained backbone
   (ImageNet ResNet-50 layer4, OR a self-supervised foundation model DINOv2)
2. Fusing them with appearance features via gated cross-attention

For air-ground UAV-UGV matching with extreme scale differences, DINOv2
features are strongly preferred over ImageNet ResNet features: DINOv2 is
trained on LVD-142M which covers diverse viewpoints including substantial
aerial imagery, yielding scale- and viewpoint-invariant semantic priors.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DINOv2SemanticExtractor(nn.Module):
    """Frozen DINOv2 ViT as a cross-view, scale-invariant semantic prior.

    DINOv2 is trained self-supervised on LVD-142M, which covers diverse
    viewpoints (ground-level, aerial, oblique) and a far wider distribution
    than ImageNet. This makes it markedly better as a semantic prior for
    cross-view air-ground matching than ImageNet ResNet features or
    supervised semantic segmentation backbones (Cityscapes / ADE20K), which
    are dominated by ground-level urban imagery.

    The DINOv2 ViT is kept frozen (no_grad forward, eval mode). Only the
    1x1 projection from DINOv2's feature dim to ``d_model`` is trainable,
    so this module is light in optimization cost but heavy in inductive
    bias — exactly the trade-off appropriate for a semantic prior branch.

    Notes:
        * The first call to ``torch.hub.load`` downloads pretrained weights
          (cached afterwards). Requires internet access on first run.
        * DINOv2 ViTs use patch_size=14. Inputs are resized to the nearest
          14-divisible spatial shape before feature extraction; output
          spatial stride is therefore approximately 14 (not 32).
        * Inputs are expected in the same HWC, [0, 1] layout as the rest of
          the OETR backbones; ImageNet normalization is applied internally.
    """

    _PATCH_SIZE = 14
    _OUT_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    @staticmethod
    def _load_dino_model(model_name):
        """Prefer a cached local DINOv2 repo to avoid mandatory GitHub access."""
        hub_dir = Path(torch.hub.get_dir())
        candidates = (
            hub_dir / 'facebookresearch_dinov2_main',
            Path.home() / '.cache' / 'torch' / 'hub' / 'facebookresearch_dinov2_main',
        )

        seen = set()
        for repo_dir in candidates:
            repo_dir = repo_dir.resolve()
            if repo_dir in seen:
                continue
            seen.add(repo_dir)
            if repo_dir.exists():
                return torch.hub.load(
                    str(repo_dir), model_name, source='local', verbose=False,
                )

        return torch.hub.load(
            'facebookresearch/dinov2', model_name, verbose=False,
        )

    def __init__(self, d_model=256, model_name='dinov2_vitb14'):
        super().__init__()
        if model_name not in self._OUT_DIMS:
            raise ValueError(
                f'Unsupported DINOv2 model: {model_name}. '
                f'Choose from {list(self._OUT_DIMS.keys())}.'
            )
        self.model_name = model_name
        out_dim = self._OUT_DIMS[model_name]

        self.dino = self._load_dino_model(model_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        self.register_buffer(
            'mean',
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_dim, d_model, 1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

    def train(self, mode=True):
        """Keep DINOv2 in eval mode regardless of parent train/eval state."""
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def _extract(self, x_chw):
        """Run frozen DINOv2 on a CHW, ImageNet-normalized input.

        Args:
            x_chw: [N, 3, H, W] in [0, 1] range, CHW format.
        Returns:
            patch features [N, C, h, w] with spatial stride ~= 14.
        """
        x = (x_chw - self.mean) / self.std

        h, w = x.shape[-2:]
        new_h = max(self._PATCH_SIZE, round(h / self._PATCH_SIZE) * self._PATCH_SIZE)
        new_w = max(self._PATCH_SIZE, round(w / self._PATCH_SIZE) * self._PATCH_SIZE)
        if new_h != h or new_w != w:
            x = F.interpolate(
                x, size=(new_h, new_w),
                mode='bilinear', align_corners=False,
            )

        feats = self.dino.forward_features(x)
        patch_tokens = feats['x_norm_patchtokens']  # [N, L, C]
        ph = new_h // self._PATCH_SIZE
        pw = new_w // self._PATCH_SIZE
        N, _, C = patch_tokens.shape
        return patch_tokens.permute(0, 2, 1).reshape(N, C, ph, pw)

    def forward(self, image_hwc):
        """
        Args:
            image_hwc: [N, H, W, 3] in [0, 1] range, matching the OETR
                backbone's image convention.
        Returns:
            semantic features [N, d_model, h, w] at approximately stride 14.
        """
        x = image_hwc.permute(0, 3, 1, 2).contiguous()
        feat = self._extract(x)
        return self.proj(feat)
