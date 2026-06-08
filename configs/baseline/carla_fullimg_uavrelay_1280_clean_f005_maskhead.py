from src.config.default import _CN as cfg

cfg.OUTPUT = 'carla_fullimg_uavrelay_1280_clean_f005_maskhead'

cfg.DATASET.DATA_ROOT = r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_sem_rich_20260531_174329'

# Warm-start from the verified bbox baseline, then add explicit mask supervision.
cfg.OETR.CHECKPOINT = r'/root/autodl-tmp/semantic-OETR/OUTPUT/OETR/checkpoints/carla_fullimg_uavrelay_1280_clean_f005/06-02-14:52/model_epoch_18.pth'
cfg.OETR.LEGACY_CHECKPOINT = None
cfg.OETR.NORM_INPUT = True

cfg.OETR.BACKBONE.NUM_LAYERS = 50

cfg.OETR.ENHANCED.SEMANTIC_ENABLE = False
cfg.OETR.ENHANCED.SEMANTIC_SOURCE = 'layer4'
cfg.OETR.ENHANCED.SCALE_PE = False
cfg.OETR.ENHANCED.SEM_LOSS_WEIGHT = 0.0
cfg.OETR.ENHANCED.SCALE_LOSS_WEIGHT = 0.0

cfg.DATASET.TRAIN.DATA_SOURCE = 'carla_fullimage_pairs'
cfg.DATASET.TRAIN.LIST_PATH = r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_sem_rich_20260531_174329/oetr_fullimage_uavrelay_clean_f005/pairs_train.jsonl'
cfg.DATASET.TRAIN.PAIRS_LENGTH = None
cfg.DATASET.TRAIN.WITH_MASK = True
cfg.DATASET.TRAIN.IMAGE_SIZE = [1280, 720]
cfg.DATASET.TRAIN.VIZ = False

cfg.DATASET.VAL.DATA_SOURCE = 'carla_fullimage_pairs'
cfg.DATASET.VAL.LIST_PATH = r'/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_sem_rich_20260531_174329/oetr_fullimage_uavrelay_clean_f005/pairs_val.jsonl'
cfg.DATASET.VAL.PAIRS_LENGTH = None
cfg.DATASET.VAL.WITH_MASK = True
cfg.DATASET.VAL.IMAGE_SIZE = [1280, 720]
cfg.DATASET.VAL.OIOU = False
cfg.DATASET.VAL.VIZ = False

cfg.OETR.LOSS.OIOU = False
cfg.OETR.LOSS.CYCLE_OVERLAP = False
cfg.OETR.LOSS.MASK_LOSS_WEIGHT = 0.3
cfg.OETR.LOSS.MASK_BCE_WEIGHT = 1.0
cfg.OETR.LOSS.MASK_DICE_WEIGHT = 1.0
