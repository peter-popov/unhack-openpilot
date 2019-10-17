#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

CONFIG_NAME_KEY = "Name"
CONFIG_HOST_ROOTPATH_KEY = "HostRootPath"
CONFIG_HOST_RESULTSDIR_KEY = "HostResultsDir"
CONFIG_DEVICE_PATH_KEY = "DevicePath"
CONFIG_DEVICES_KEY = "Devices"
CONFIG_HOST_NAME_KEY = "HostName"
CONFIG_RUNS_KEY = "Runs"
CONFIG_MODEL_KEY = "Model"
CONFIG_MODEL_NAME_SUBKEY = "Name"
CONFIG_MODEL_DLC_SUBKEY = "Dlc"
CONFIG_MODEL_DATA_SUBKEY = "Data"
CONFIG_MODEL_INPUTLIST_SUBKEY = "InputList"
CONFIG_MODEL_RANDOMINPUT_SUBKEY = "RandomInput"
CONFIG_MODEL_INPUTS = "InputDimensions"
CONFIG_RUNTIMES_KEY = "Runtimes"
CONFIG_ARCHITECTURES_KEY = "Architectures"
CONFIG_COMPILER_KEY = "Compiler"
CONFIG_STL_LIBRARY_KEY = "C++ Standard Library"
CONFIG_MEASUREMENTS_KEY = "Measurements"
CONFIG_PERF_PROFILE_KEY = "PerfProfile"
CONFIG_CPU_FALLBACK_KEY = "CpuFallback"
CONFIG_BUFFERTYPES_KEY = "BufferTypes"
CONFIG_PROFILING_LEVEL_KEY = "ProfilingLevel"

CONFIG_ARTIFACTS_KEY = "Artifacts"
CONFIG_ARTIFACTS_COMPILER_KEY_HOST = "x86_64-linux-clang"
CONFIG_ARTIFACTS_COMPILER_KEY_TARGET_ANDROID_OSPACE = [
    "arm-android-gcc4.9_s",
    "arm-android-clang6.0_s",
    "aarch64-android-gcc4.9_s",
    "aarch64-android-clang6.0_s"
]

CONFIG_JSON_ROOTKEYS = [CONFIG_NAME_KEY, CONFIG_HOST_ROOTPATH_KEY,
                        CONFIG_HOST_RESULTSDIR_KEY, CONFIG_DEVICE_PATH_KEY,
                        CONFIG_DEVICES_KEY, CONFIG_HOST_NAME_KEY, CONFIG_RUNS_KEY,
                        CONFIG_MODEL_KEY, CONFIG_RUNTIMES_KEY,
                        CONFIG_MEASUREMENTS_KEY, CONFIG_PERF_PROFILE_KEY, CONFIG_CPU_FALLBACK_KEY,
                        CONFIG_BUFFERTYPES_KEY, CONFIG_PROFILING_LEVEL_KEY]
CONFIG_JSON_MODEL_COMMON_SUBKEYS = [CONFIG_MODEL_NAME_SUBKEY,
                             CONFIG_MODEL_DLC_SUBKEY]
CONFIG_JSON_MODEL_DEFINED_INPUT_SUBKEYS = [CONFIG_MODEL_INPUTLIST_SUBKEY,
                             CONFIG_MODEL_DATA_SUBKEY]
CONFIG_JSON_MODEL_RANDOM_INPUT_SUBKEYS = [CONFIG_MODEL_RANDOMINPUT_SUBKEY]
