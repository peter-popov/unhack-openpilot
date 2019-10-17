# Copyright (c) 2017 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

NDK_TOOLCHAIN_VERSION := clang

APP_PLATFORM := android-16
APP_ABI := armeabi-v7a arm64-v8a
APP_STL := c++_shared
APP_CPPFLAGS += -std=c++11 -fexceptions -frtti
APP_LDFLAGS = -nodefaultlibs -lc -lm -ldl -lgcc
