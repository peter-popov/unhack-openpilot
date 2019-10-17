#
# Copyright (c) 2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

class ConfigError(Exception):
    def __str__(self):
        return ('\nConfiguration Error: ' + repr(self) + '\n')

class AdbShellCmdFailedException(Exception):
    def __str__(self):
        return('\nadb shell command Error: ' + repr(self) + '\n')
