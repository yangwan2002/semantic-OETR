#!/usr/bin/env python
"""
@File    :   oetr.py
@Time    :   2021/07/28 17:20:43
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import torch

from dloc.core.utils.base_model import BaseModel  # noqa: E402
from src.config.default import get_cfg_defaults
from src.model import build_detectors


class OETR(BaseModel):
    default_conf = {
        'num_layers': 50,
        'weights': 'oetr.pth',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def build_cfg(self, conf):
        cfg = get_cfg_defaults()
        cfg.OETR.BACKBONE.NUM_LAYERS = conf['num_layers']
        return cfg

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        self.cfg = self.build_cfg(self.conf)
        self.net = build_detectors(self.cfg.OETR)
        model_file = model_path / self.conf['weights']
        self.net.load_state_dict(torch.load(model_file))

    def _forward(self, data):
        box1, box2 = self.net.forward_dummy(
            data['image0'],
            data['image1'],
            data.get('mask0'),
            data.get('mask1'),
        )
        return box1, box2
