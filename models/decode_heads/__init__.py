# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Changqian Yu (y-changqian@outlook.com)

from .cond_aspp_head import CondASPPHead
from .cond_sep_aspp_head import CondDepthwiseSeparableASPPHead
from .cond_uper_head import CondUPerHead

__all__ = ['CondASPPHead', 'CondDepthwiseSeparableASPPHead', 'CondUPerHead']
