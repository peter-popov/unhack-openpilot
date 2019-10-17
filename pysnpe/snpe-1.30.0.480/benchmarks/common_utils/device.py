#
# Copyright (c) 2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

from .adb import Adb
from .env import EnvHelper

from os import remove, path
import logging

logger = logging.getLogger(__name__)

class Device:
    def __init__(self, device_id, archs, device_root):
        self.device_id = device_id
        self.adb_helper = Adb('adb', device_id)
        self.env_helper = None
        self.device_root = device_root
        self.soc = self.get_soc()
        self.archs = archs

    def init_env(self, artifacts_dir, is_sdk):
        self.env_helper = EnvHelper(self.adb_helper, self.archs, artifacts_dir, self.device_root, is_sdk)

    def setup_device(self):
        self.env_helper.init()

    def push_data(self, src, dst):
        ret, _, err = self.adb_helper.push(src, dst)
        if ret != 0:
            logger.error('[{}] Failed to push: {}'.format(self.device_id, src))
            logger.error('[{}] stderr: {}'.format(self.device_id, err))

    def get_soc(self):
        ret, out, err = self.adb_helper.shell('getprop', ['ro.board.platform'])
        if ret != 0:
            logger.error('[{}] Failed to get SOC'.format(self.device_id))
            soc = 'n/a'
        else:
            soc = ''.join(out).upper()
        return soc

    def __str__(self):
        return '{}-{}'.format(self.soc, self.device_id)
