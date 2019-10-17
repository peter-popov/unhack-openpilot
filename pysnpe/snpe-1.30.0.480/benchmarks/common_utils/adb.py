#
# Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

from .common import execute, Timeouts

from functools import wraps
import logging
import re
import time
import subprocess
import json
import os

logger = logging.getLogger(__name__)

UNKNOWN = 'unknown'
REGX_GET_PROP = re.compile('\[(.+)\]: \[(.+)\]')
getprop_list = ['ro.product.name',
                'ro.serialno',
                'ro.product.model',
                'ro.product.board',
                'ro.product.brand',
                'ro.product.device',
                'ro.product.manufacturer',
                'ro.product.cpu.abi',
                'ro.build.au_rev',
                'ro.build.description',
                'ro.build.version.sdk']

def retry(tries=3, delay=1, backoff=2):
    def retry_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            l_tries, l_delay = tries, delay
            while l_tries > 0:
                logger.debug('Try {}'.format(tries - l_tries))
                code, out, err = func(*args, **kwargs)
                if code == 0:
                    return code, out, err
                logger.debug('Failed. Retrying')
                l_tries -= 1
                l_delay *= backoff
                time.sleep(l_delay)
            return func(*args, **kwargs)
        return wrapper
    return retry_decorator

def check_adb_version(adb_path):
    adb_cmd = "%s version" % adb_path
    recommended_adb_version = "1.0.39"
    p = subprocess.Popen(adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    cmd_out, cmd_err = p.communicate()
    returncode = p.returncode
    if returncode is not 0:
        logger.error('%s failed with stderr of: %s' % (adb_cmd, cmd_out + cmd_err))
        raise Exception("Failed to perform adb version check using command : %s" % adb_cmd)
    adb_version = cmd_out.split('\n')[0].split(' ')[-1]
    if adb_version > recommended_adb_version:
        logger.warning('The version of adb (%s) found at %s has not been validated. Recommended to use known stable adb version %s' % (adb_version, adb_path, recommended_adb_version))
    elif not adb_version == recommended_adb_version:
        raise ValueError("sdk-tests require adb version %s. Found adb version %s at %s" % (recommended_adb_version, adb_version, adb_path))

class Adb(object):
    def __init__(self, adb_executable, device, master_id=None, hostname='localhost'):
        if not os.path.exists(adb_executable):
            exists = False
            for path in os.environ['PATH'].split(os.pathsep):
                if os.path.exists(os.path.join(path, 'adb')):
                    exists = True
                    break
            if not exists:
                logger.error('Invalid path for adb: %s.' % adb_executable)
                raise RuntimeError('adb not in PATH')
        self.__adb_executable = adb_executable
        self.__master_id = master_id
        self._adb_device = device
        self._hostname = 'localhost' if hostname is None else hostname

    @retry()
    def push(self, src, dst, cwd='.'):
        dst_dir_exists = False
        if (self._execute('shell', ['[ -d %s ]' % dst], cwd=cwd)[0] == 0):
            dst_dir_exists = True
        else:
            if (os.path.basename(src) == os.path.basename(dst)) and not os.path.isdir(src):
                dir_name = os.path.dirname(dst)
            else:
                dir_name = dst
            ret, _, err = self._execute('shell', ['mkdir', '-p', dir_name])
            if ret != 0:
                logger.warning('mkdir failed for parent folder')
        code, out, err = self._execute('push', [src, dst], cwd=cwd)
        if code == 0:
            # Check if push was successful
            if src[-1] == '/':
                src = src[:-1]
            file_name = src.split('/')[-1]
            # check if destination directory exists
            # if it exists, then append file name to dst
            # otherwise, adb will rename src dir to dst
            if dst_dir_exists:
                dst = (dst + file_name) if dst[-1] == '/' else (dst + '/' + file_name)
            code, out, error = self._execute('shell',['[ -e %s ]' % dst], cwd=cwd)
        return code, out, err

    @retry()
    def pull(self, src, dst, cwd='.'):
        return self._execute('pull', [src, dst], cwd=cwd)

    @retry()
    def shell(self, command, args=[]):
        shell_args = ["{} {}; echo '\n'$?".format(command, ' '.join(args))]
        logger.debug("Executing on the android device")
        logger.debug(shell_args)
        code, out, err = self._execute('shell', shell_args)
        if code == 0:
            if len(out) > 0:
                try:
                    code = int(out[-1])
                    out = out[:-1]
                except ValueError as ex:
                    code = -1
                    out.append(str(ex))
            else:
                code = -1

            if code != 0 and len(err) == 0:
                err = out
        else:
            code = -1
        return code, out, err

    @retry()
    def install(self, apk_path, package_name):
        code, out, err = self._execute('install', ['-r', apk_path])
        if code == 0:
            # We confirm the installation as we can't rely on the result code above
            code, out, err = self.shell('pm', ['list', 'packages', '|', 'grep', package_name])
        return code, out, err

    @retry()
    def uninstall(self, package_name):
        code, out, err = self._execute('uninstall', [package_name])
        if code == 0:
            # We need to validate the output as adb install will return always 0
            code = 0 if out[-1] == 'Success' else -1
        return code, out, err

    def _execute(self, command, args, cwd='.'):
        adb_command_args = ['-H', self._hostname, '-s', self._adb_device, command] + args
        (return_code, output, error) = execute(self.__adb_executable, adb_command_args, cwd=cwd, timeout=Timeouts.ADB_DEFAULT_TIMEOUT)
        # when the process gets killed, it will return -9 code; Logging this error for debug purpose
        if return_code == -9:
            logger.error("adb command didn't execute within the timeout. Is device in good state?")

        if (self._adb_device) and ("error: device offline" in error or "error: device \'%s\' not found" % (self._adb_device) in error):
            self.recover_device()
            # Adding info log to print the last failed command
            logger.info("Retrying the command after device recovery: %s %s", self.__adb_executable, adb_command_args)
            (return_code, output, error) = execute(self.__adb_executable, adb_command_args, cwd=cwd, timeout=Timeouts.ADB_DEFAULT_TIMEOUT)

        return (return_code, output, error)

    @retry()
    def get_devices(self):
        code, out, err = self._execute('devices', [])
        if code != 0:
            logger.error("Could not retrieve list of adb devices connected, following error "
                         "occured: {0}".format("\n".join(err)))
            return code, out, err

        devices = []
        for line in out:
            # Checking the connected adb devices with serial id and ip address
            match_obj = re.match("^([a-zA-Z0-9.]+(:[0-9]+)?)\s+device", line, re.M)
            if match_obj:
                devices.append(match_obj.group(1))
        return code, devices, err

    @retry()
    def get_device_info(self, fatal=True):
        _info = {}
        ret, out, err = self._execute('shell', ['getprop'])
        if ret != 0:
            if fatal != True:
                logger.warning('Non fatal get prop call failure, is the target os not Android?')
                return ret, [], err
        if out:
            for line in out:
                line = line.strip()
                m = REGX_GET_PROP.search(line)
                if m:
                    _info[m.group(1)] = m.group(2)
        dev_info = []
        for prop_key in getprop_list:
            if not prop_key in _info:
                dev_info.append([prop_key, UNKNOWN])
            else:
                dev_info.append([prop_key, _info[prop_key]])
        return ret, dev_info, err

    def check_file_exists(self, file_path):
        """
        Returns 'True' if the file exists on the target
        Using 'ls' instead of 'test' cmd as 'test' was behaving abnormally on LE
        """
        ret, out, err = self._execute('shell', ['ls', file_path, '| wc -l'])
        if 'No such file or directory' in ''.join(err) or 'No such file or directory' in ''.join(out):
            ret = 1
        return ret == 0

    def is_device_online(self):
        code, out, err = self._execute('wait-for-device', [])
        return code == 0

    def getmetabuild(self):
        try:
            from .device_utils import get_metabuild
            os_type, metabuild = get_metabuild(self, logger)
            return os_type, metabuild
        except Exception as e:
            return UNKNOWN, UNKNOWN

    def recover_device(self):
        logger.warning("Trying to recover the device %s from bad state" % (self._adb_device))
        try:
            from . import device_module
            if not self.__master_id:
                logger.warning("Device master_id is not provided. Cannot recover the device. Retrieving from json file.")
                with open(os.path.join(os.path.dirname(device_module.__file__), 'device_info.json')) as master_data_file:
                    master_data = json.load(master_data_file)
                    if self._adb_device in master_data:
                        self.__master_id = master_data[self._adb_device]
                    else:
                        logger.warning("Device master_id is not provided in the json file. Cannot recover the device.")
                        return

            device_handle = device_module.device_master.Device_instance(self.__master_id)
            device_handle.device_controller.PowerOff()
            time.sleep(5)
            device_handle.device_controller.PowerOn()
            time.sleep(5)
            if self.is_device_online():
                logger.info("Device successfully recovered.")
            else:
                logger.warning("Device not recovered from the bad state")
            # TODO want to root and remount after the reboot?
        except ImportError:
            logger.warning("Failed to import device_module. Skipping device recovery.")
        except:
            logger.warning("Not able to recover the device state. Device master module might be missing.")
            import traceback
            traceback.print_exc()
