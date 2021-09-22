# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Changqian Yu (y-changqian@outlook.com)

from .builder import (build_backbone, build_head, build_loss, build_segmentor)
from .decode_heads import *  # noqa: F401,F403

__all__ = ['build_backbone', 'build_head', 'build_loss', 'build_segmentor']
