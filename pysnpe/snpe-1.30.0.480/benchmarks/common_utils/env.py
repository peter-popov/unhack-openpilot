#
# Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

from .adb import Adb

from os import path, remove, walk
import logging

logger = logging.getLogger(__name__)

class EnvHelper:
    _ENVSETUP_SCRIPT = 'env.sh'

    def __init__(self, adb_obj, arch_list, artifacts_dir, device_root, is_sdk=False):
        assert(adb_obj.__class__ == Adb)
        self._adb = adb_obj
        self._arch_list = arch_list
        self._artifacts_dir = artifacts_dir
        self._device_root = device_root
        self._is_sdk = is_sdk

    def init(self):
        self._clean_env()
        self._adb.shell('mkdir {}'.format(self._device_root))
        self._push_artifacts()
        self._setup_env()

    def _clean_env(self):
        logger.debug('[{}] Cleaning device root: {}'.format(self._adb._adb_device, self._device_root))
        self._adb.shell('rm', ['-rf', self._device_root])

    def _push_artifacts(self):
        if self._is_sdk:
            dsp_artifacts = path.join(self._artifacts_dir, 'lib', 'dsp')
            self._adb.push(dsp_artifacts, self._device_root)
            for arch in self._arch_list:
                self._adb.shell('mkdir', [path.join(self._device_root, arch)])
                logger.debug('[{}] Pushing artifacts for {}'.format(self._adb._adb_device, arch))
                lib_dir = path.join(self._artifacts_dir, 'lib', arch)
                bin_dir = path.join(self._artifacts_dir, 'bin', arch)
                # the following workaround is needed for android p
                self._adb.push(lib_dir, path.join(self._device_root, arch))
                self._adb.shell('mv', [path.join(self._device_root, arch, arch), path.join(self._device_root, arch, 'lib')])
                self._adb.push(bin_dir, path.join(self._device_root, arch))
                self._adb.shell('mv', [path.join(self._device_root, arch, arch), path.join(self._device_root, arch, 'bin')])
        else:
            dsp_artifacts = path.join(self._artifacts_dir, 'dsp')
            self._adb.push(dsp_artifacts, self._device_root)
            for arch in self._arch_list:
                artifacts = path.join(self._artifacts_dir, arch)
                logger.debug('[{}] Pushing artifacts for {}'.format(self._adb._adb_device, arch))
                # not able to simply push artifacts folder because
                # adb push fails for pushing libsymphony* symlinks
                for root, _, file_list in walk(artifacts):
                    for file_name in file_list:
                        file_path = path.join(root, file_name)
                        prefix = path.commonprefix([artifacts, file_path])
                        dst = path.join(self._device_root, arch, path.relpath(file_path, prefix))
                        self._adb.push(file_path, dst)

    def _setup_env(self):
        logger.debug('[{}] Pushing envsetup scripts'.format(self._adb._adb_device))
        if self._is_sdk:
            dsp_dir = path.join(self._device_root, 'dsp')
        else:
            dsp_dir = path.join(self._device_root, 'dsp', 'lib')
        for arch in self._arch_list:
            lib_dir = path.join(self._device_root, arch, 'lib')
            bin_dir = path.join(self._device_root, arch, 'bin')
            commands = [
                'export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH'.format(lib_dir),
                'export ADSP_LIBRARY_PATH="{};/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp"'.format(dsp_dir),
                'export PATH={}:$PATH'.format(bin_dir)
            ]
            script_name = '{}_{}_{}'.format(self._adb._adb_device, arch, self._ENVSETUP_SCRIPT)
            with open(script_name, 'w') as f:
                f.write('\n'.join(commands) + '\n')
            self._adb.push(script_name, self._device_root)
            remove(script_name)
