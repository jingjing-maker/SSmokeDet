#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine.module as M

from .yolo_head import SSmokeNetHead
from .yolo_pafpn import YOLOPAFPN


class SSmokeNet(M.Module):
    """
    SSmokeNet model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = SSmokeNetHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        assert not self.training
        outputs = self.head(fpn_outs)

        return outputs
