#
# Copyright (c) 2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#


SUPPORTED_TARGETS = [
    'arm-android-clang6.0',
    'aarch64-android-clang6.0',
    'arm-linux-gcc4.9sf',
    'aarch64-oe-linux-gcc8.2',
    'arm-oe-linux-gcc8.2hf',
    'aarch64-qnx-gcc5.4'
]

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'

RUNTIME_STR_MAP = {
    'CPU': 'cpu_float32',
    'GPU': 'gpu_float32_16_hybrid',
    'GPU_FP16': 'gpu_float16',
    'DSP': 'dsp_fixed8_tf',
    'AIP': 'aip_fixed8_tf'
}
