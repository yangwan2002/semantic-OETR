from src.config.default import _CN as cfg

cfg.OUTPUT = 'carla_fullimg_ugvrelay_soft_planar_finetune_pe'

cfg.DATASET.DATA_ROOT = (
    r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/'
    r'paper_eval_l2_sem_rich_20260531_174329'
)

# Warm-start from the strongest shared mask baseline, then adapt to UGV soft+planar GT.
cfg.OETR.CHECKPOINT = (
    r'/root/autodl-tmp/semantic-OETR/OUTPUT/OETR/checkpoints/'
    r'carla_fullimg_uavrelay_1280_clean_f005_maskhead_semdino_maskalign_v2_best/'
    r'06-03-17:02/model_epoch_11.pth'
)
cfg.OETR.LEGACY_CHECKPOINT = None
cfg.OETR.NORM_INPUT = True

cfg.OETR.BACKBONE.NUM_LAYERS = 50

cfg.OETR.ENHANCED.SEMANTIC_ENABLE = True
cfg.OETR.ENHANCED.SEMANTIC_SOURCE = 'dinov2'
cfg.OETR.ENHANCED.DINOV2_MODEL = 'dinov2_vitb14'
cfg.OETR.ENHANCED.SCALE_PE = True
cfg.OETR.ENHANCED.SEM_LOSS_WEIGHT = 0.1
cfg.OETR.ENHANCED.SEM_LOSS_MODE = 'mask_align'
cfg.OETR.ENHANCED.SEM_BG_WEIGHT = 0.25
cfg.OETR.ENHANCED.SEM_BG_MARGIN = 0.05
cfg.OETR.ENHANCED.SCALE_LOSS_WEIGHT = 0.0

cfg.DATASET.TRAIN.DATA_SOURCE = 'carla_fullimage_pairs'
cfg.DATASET.TRAIN.LIST_PATH = (
    r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/'
    r'paper_eval_l2_sem_rich_20260531_174329/'
    r'oetr_fullimage_ugvrelay_soft_planar_clean_f005/pairs_train.jsonl'
)
cfg.DATASET.TRAIN.PAIRS_LENGTH = None
cfg.DATASET.TRAIN.WITH_MASK = True
cfg.DATASET.TRAIN.IMAGE_SIZE = [1280, 720]
cfg.DATASET.TRAIN.VIZ = False
cfg.DATASET.TRAIN.AUGMENT = True
cfg.DATASET.TRAIN.AUGMENT_BRIGHTNESS = 0.2
cfg.DATASET.TRAIN.AUGMENT_CONTRAST = 0.2
cfg.DATASET.TRAIN.AUGMENT_GAMMA = 0.15
cfg.DATASET.TRAIN.AUGMENT_RGB_SHIFT = 0.04
cfg.DATASET.TRAIN.AUGMENT_BLUR_PROB = 0.2

cfg.DATASET.VAL.DATA_SOURCE = 'carla_fullimage_pairs'
cfg.DATASET.VAL.LIST_PATH = (
    r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/'
    r'paper_eval_l2_sem_rich_20260531_174329/'
    r'oetr_fullimage_ugvrelay_soft_planar_clean_f005/pairs_val.jsonl'
)
cfg.DATASET.VAL.PAIRS_LENGTH = None
cfg.DATASET.VAL.WITH_MASK = True
cfg.DATASET.VAL.IMAGE_SIZE = [1280, 720]
cfg.DATASET.VAL.OIOU = False
cfg.DATASET.VAL.VIZ = False

cfg.OETR.LOSS.OIOU = False
cfg.OETR.LOSS.CYCLE_OVERLAP = False
cfg.OETR.LOSS.MASK_LOSS_WEIGHT = 1.0
cfg.OETR.LOSS.MASK_BCE_WEIGHT = 1.0
cfg.OETR.LOSS.MASK_DICE_WEIGHT = 1.0
cfg.OETR.LOSS.MASK_POS_WEIGHT_MAX = 6.0
cfg.OETR.LOSS.MASK_USE_GT_BOX_PRIOR = False
cfg.OETR.LOSS.MASK_PRIOR_INPUT_WEIGHT = 0.5
cfg.OETR.LOSS.MASK_PRIOR_LOGIT_WEIGHT = 0.25
