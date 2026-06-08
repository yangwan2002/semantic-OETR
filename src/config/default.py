from yacs.config import CfgNode as CN

_CN = CN()
_CN.OUTPUT = ''

# OETR Pipeline
_CN.OETR = CN()
# Strict checkpoint (must match current model architecture).
_CN.OETR.CHECKPOINT = None
# Legacy OETR checkpoint (partial load, used for warm-starting from the
# original single-scale OETR model). Ignored if CHECKPOINT is also set.
_CN.OETR.LEGACY_CHECKPOINT = None
_CN.OETR.NORM_INPUT = True

# 1. OETR-backbone (multi-scale ResNet) config
_CN.OETR.BACKBONE = CN()
_CN.OETR.BACKBONE.NUM_LAYERS = 50

# 2. OETR-neck module config
_CN.OETR.NECK = CN()
_CN.OETR.NECK.MAX_SHAPE = (100, 100)  # max feature-map shape

# 3. OETR-enhanced module config (multi-scale + semantic + scale-adaptive PE)
_CN.OETR.ENHANCED = CN()
_CN.OETR.ENHANCED.FPN_DIM = 256
_CN.OETR.ENHANCED.SEMANTIC_ENABLE = True
# Source of the semantic prior:
#   'layer4' - reuse the trainable backbone's layer4 (cheap, ImageNet-domain)
#   'dinov2' - frozen DINOv2 ViT (recommended for cross-view air-ground)
_CN.OETR.ENHANCED.SEMANTIC_SOURCE = 'dinov2'
# DINOv2 variant when SEMANTIC_SOURCE='dinov2'. Options:
# 'dinov2_vits14' (21M), 'dinov2_vitb14' (86M), 'dinov2_vitl14' (300M).
_CN.OETR.ENHANCED.DINOV2_MODEL = 'dinov2_vitb14'
_CN.OETR.ENHANCED.SCALE_PE = True
_CN.OETR.ENHANCED.SEM_LOSS_WEIGHT = 0.5
_CN.OETR.ENHANCED.SEM_LOSS_MODE = 'bbox_consistency'
_CN.OETR.ENHANCED.SEM_BG_WEIGHT = 0.0
_CN.OETR.ENHANCED.SEM_BG_MARGIN = 0.0
# Scale-consistency loss: directly supervises the scale estimate produced
# by ScaleAdaptivePositionEncoding using GT bbox area ratios, and enforces
# that predicted bbox area ratios are consistent with the GT scale ratio.
# Set 0 to disable (e.g. when SCALE_PE is False).
_CN.OETR.ENHANCED.SCALE_LOSS_WEIGHT = 0.5

# 4. OETR-loss module config
_CN.OETR.LOSS = CN()
_CN.OETR.LOSS.OIOU = False
_CN.OETR.LOSS.CYCLE_OVERLAP = False
_CN.OETR.LOSS.MASK_AUX_WEIGHT = 0.0
_CN.OETR.LOSS.MASK_LOSS_WEIGHT = 0.0
_CN.OETR.LOSS.MASK_BCE_WEIGHT = 1.0
_CN.OETR.LOSS.MASK_DICE_WEIGHT = 1.0
_CN.OETR.LOSS.MASK_POS_WEIGHT_MAX = 1.0
_CN.OETR.LOSS.MASK_USE_GT_BOX_PRIOR = False
_CN.OETR.LOSS.MASK_PRIOR_INPUT_WEIGHT = 1.0
_CN.OETR.LOSS.MASK_PRIOR_LOGIT_WEIGHT = 1.0

# Dataset
_CN.DATASET = CN()
_CN.DATASET.DATA_ROOT = None

# training and validating
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.DATA_SOURCE = 'megadepth'
_CN.DATASET.TRAIN.LIST_PATH = 'assets/train_scenes.txt'
_CN.DATASET.TRAIN.PAIRS_LENGTH = None
_CN.DATASET.TRAIN.WITH_MASK = None
_CN.DATASET.TRAIN.TRAIN = True
_CN.DATASET.TRAIN.VIZ = False
_CN.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
_CN.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]
_CN.DATASET.TRAIN.AUGMENT = False
_CN.DATASET.TRAIN.AUGMENT_BRIGHTNESS = 0.2
_CN.DATASET.TRAIN.AUGMENT_CONTRAST = 0.2
_CN.DATASET.TRAIN.AUGMENT_GAMMA = 0.15
_CN.DATASET.TRAIN.AUGMENT_RGB_SHIFT = 0.04
_CN.DATASET.TRAIN.AUGMENT_BLUR_PROB = 0.2

_CN.DATASET.VAL = CN()
_CN.DATASET.VAL.DATA_SOURCE = 'megadepth'
_CN.DATASET.VAL.LIST_PATH = 'assets/val_scenes.txt'
_CN.DATASET.VAL.PAIRS_LENGTH = None
_CN.DATASET.VAL.WITH_MASK = False
_CN.DATASET.VAL.OIOU = True
_CN.DATASET.VAL.TRAIN = False
_CN.DATASET.VAL.VIZ = True
_CN.DATASET.VAL.IMAGE_SIZE = [640, 640]
_CN.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the OETR project."""
    return _CN.clone()

