#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import absolute_import
import os
import sys
import logging
from subprocess import check_output
from .snpebm_constants import ERRNUM_MD5CHECKSUM_CHECKSUM_MISMATCH
from .snpebm_constants import ERRNUM_MD5CHECKSUM_UNKNOWN_ERROR
from common_utils.exceptions import AdbShellCmdFailedException

logger = logging.getLogger(__name__)
MD5_HOST_BINARY = 'md5sum'


def _execute_cmd(cmd_str, shell=False):
    try:
        logger.debug('Executing {%s}' % cmd_str)
        cmd_handle = check_output(cmd_str, shell=True).decode()
        logger.debug('Command Output: \n"%s"' % cmd_handle)
        return cmd_handle
    except Exception as e:
        logger.error('Could not execute the cmd {%s}' % (cmd_str),
                     exc_info=True)
        logger.error(repr(e))
        raise


def _gen_md5_for_one_file(path):
    return _execute_cmd(' '.join([MD5_HOST_BINARY, path])).split()[0]


def _exit_on_md5_mismatch(device, md5_binary_on_target, host_file, device_file):
    try:
        host_md5 = _gen_md5_for_one_file(host_file)
        device_md5 = _get_md5_for_one_file_on_target(device, md5_binary_on_target, device_file)
        if  host_md5 != device_md5:
            logger.error('Abort during checksum check:\n%s\n and its copy at \n%s\n have different checksums'%(host_file,device_file))
            sys.exit(ERRNUM_MD5CHECKSUM_CHECKSUM_MISMATCH)
    except Exception as e:
        logger.error('Unknown error during checksum check between \n\n%s\n and \n%s\n' % (host_file, device_file))
        sys.exit(ERRNUM_MD5CHECKSUM_UNKNOWN_ERROR)


def _exit_on_md5_mismatch_after_attempt_copy_once(device, md5_binary_on_target, host_file, device_file):
    try:
        host_md5 = _gen_md5_for_one_file(host_file)
        device_md5 = _get_md5_for_one_file_on_target(device, md5_binary_on_target, device_file)
        if  host_md5 != device_md5:
            # do copy here, then call _exit_on_md5_mismatch
            logger.info('md5 does not match for %s, copy from host again' % device_file)
            device.adb.push(host_file, device_file)
            _exit_on_md5_mismatch(device, md5_binary_on_target, host_file, device_file)
    except Exception as e:
        logger.error('Unknown error during checksum check between \n\n%s\n and \n%s, error msg = %s\n' % (host_file, device_file, repr(e)))
        sys.exit(ERRNUM_MD5CHECKSUM_UNKNOWN_ERROR)


def _find_md5_binary_on_target(device):
    md5_path = '/system/bin/md5'
    result = False
    try:
        result = device.adb.check_file_exists(md5_path)
    except:
        result = False

    if result == False:
        md5_path = '/system/bin/md5sum'
        try:
            result = device.adb.check_file_exists(md5_path)
        except:
            result = False

    if result == False:
        md5_cmd = 'which md5'
        md5_path = str()
        ret, out, err = device.adb.shell(md5_cmd)
        if ret != 0:
            md5_path = ''
        else:
            md5_path = out[0]

        if md5_path == '' or "not found" in md5_path:
            md5_cmd = 'which md5sum'
            ret, out, err = device.adb.shell(md5_cmd)
            if ret != 0:
                md5_path = ''
            else:
                md5_path = out[0]
        if md5_path == '' or "not found" in md5_path:
            logger.error('Could not find md5 checksum binary on device.')
            md5_path = ''
    logger.info('md5 command to be used: %s' % md5_path.rstrip())

    return md5_path.rstrip()


def _get_md5_for_one_file_on_target(device, md5_binary_on_target, path):
    ret, out, err = device.adb.shell(md5_binary_on_target, [path])
    if ret != 0:
        raise AdbShellCmdFailedException(err)
    return out[0].split()[0]


def perform_md5_check(device, artifacts):
    # for snpebm and dnn_mode artifacts,  loop through and compare their md5 chechsums
    md5_binary_on_target = _find_md5_binary_on_target(device)
    logger.info('Perform MD5 check on files on device')
    for _host_path, _dev_dir in artifacts:
        if os.path.isfile(_host_path):
            dev_file = '/'.join([_dev_dir,os.path.basename(_host_path)])
            if md5_binary_on_target == '':
                device.adb.push(_host_path, dev_file)
            elif device.adb.check_file_exists(dev_file):
                _exit_on_md5_mismatch_after_attempt_copy_once(device, md5_binary_on_target, _host_path, dev_file)
            else:
                logger.info(
                    '%s not present on device at \n\t%s, copying' % (os.path.basename(_host_path), _dev_dir))
                device.adb.push(_host_path, dev_file)
                _exit_on_md5_mismatch(device, md5_binary_on_target, _host_path, dev_file)
        elif os.path.isdir(_host_path):
            for _root, _dirs, _files in os.walk(_host_path):
                for _file in _files:
                    dev_file = '/'.join([_dev_dir, _file])
                    if not device.adb.check_file_exists(dev_file):
                        logger.info(
                            '%s not present on device at \n\t%s, copying' % (_file, _dev_dir))
                        device.adb.push(os.path.join(_root, _file), dev_file)
                    if md5_binary_on_target != '':
                        _exit_on_md5_mismatch_after_attempt_copy_once(device, md5_binary_on_target, os.path.join(_root, _file), dev_file)
        else:
            # if neither a dir or file, ignore
            pass

