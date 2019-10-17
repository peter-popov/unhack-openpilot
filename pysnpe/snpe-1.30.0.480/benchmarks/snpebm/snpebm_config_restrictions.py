#==============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import absolute_import
from .snpebm_constants import *

CONFIG_VALID_DEVICEOSTYPES = [
    CONFIG_DEVICEOSTYPES_ANDROID_ARM32,
    CONFIG_DEVICEOSTYPES_ANDROID_AARCH64,
    CONFIG_DEVICEOSTYPES_LE,
    CONFIG_DEVICEOSTYPES_LE64_GCC49,
    CONFIG_DEVICEOSTYPES_LE_OE_GCC82,
    CONFIG_DEVICEOSTYPES_LE64_OE_GCC82,
    CONFIG_DEVICEOSTYPES_QNX64_GCC54
]

CONFIG_VALID_MEASURMENTS = [
    MEASURE_TIMING,
    MEASURE_MEM
]

