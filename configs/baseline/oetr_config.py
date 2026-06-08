from src.config.default import _CN as cfg

cfg.OUTPUT = 'oetr'

cfg.DATASET.DATA_ROOT = './dataset/megadepth/'

cfg.OETR.NORM_INPUT = True
cfg.OETR.CHECKPOINT = None

# 1. OETR-backbone (multi-scale ResNet)
cfg.OETR.BACKBONE.NUM_LAYERS = 50

# 2. Enhanced module config
cfg.OETR.ENHANCED.FPN_DIM = 256
cfg.OETR.ENHANCED.SEMANTIC_ENABLE = True
# DINOv2 ViT-B/14 as a frozen cross-view, scale-invariant semantic prior.
# Switch to 'layer4' to fall back to the lightweight backbone-reuse variant.
cfg.OETR.ENHANCED.SEMANTIC_SOURCE = 'dinov2'
cfg.OETR.ENHANCED.DINOV2_MODEL = 'dinov2_vitb14'
cfg.OETR.ENHANCED.SCALE_PE = True
cfg.OETR.ENHANCED.SEM_LOSS_WEIGHT = 0.5
cfg.OETR.ENHANCED.SCALE_LOSS_WEIGHT = 0.5

# 3. Dataset
cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.TRAIN.LIST_PATH = './dataset/megadepth/assets/megadepth_train_pairs.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000
cfg.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
cfg.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]
cfg.DATASET.TRAIN.VIZ = False

cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.VAL.LIST_PATH = './dataset/megadepth/assets/megadepth_validation_scale.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None
cfg.DATASET.VAL.IMAGE_SIZE = [640, 640]
cfg.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]
cfg.DATASET.VAL.VIZ = False

# 4. Loss
cfg.OETR.LOSS.OIOU = False
cfg.OETR.LOSS.CYCLE_OVERLAP = True
