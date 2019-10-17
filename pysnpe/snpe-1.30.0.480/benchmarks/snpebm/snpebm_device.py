#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import absolute_import
from multiprocessing import Process
from .snpebm_constants import *
import os
import time
import re
from common_utils.adb import Adb
from common_utils.exceptions import AdbShellCmdFailedException
import logging

logger = logging.getLogger(__name__)

REGX_SPACES = re.compile('[\s]+')
ONE_HOUR_IN_SEC = 1 * 60 * 60.0


class DeviceFactory(object):
    @staticmethod
    def make_device(device_id, config):
        assert device_id, "device id is required"
        assert config, "config is required"
        return BenchmarkDevice(device_id, device_id, config.device_path, config.platform, config.host_rootpath, config.hostname)


class BenchmarkDevice(object):
    def __init__(self, device_name, serial_no, device_root_dir, platform, host_output_dir, host_name='localhost'):
        assert device_root_dir, "device root directory is required"
        self._device_name = device_name
        self._comm_id = serial_no
        self._device_root_dir = device_root_dir
        self._host_output_dir = host_output_dir
        self._host_name = host_name
        self._mem_proc = None
        self._power_proc = None
        self._platform = platform
        self.adb = Adb('adb', serial_no, hostname=host_name)

        if(self._platform == PLATFORM_OS_ANDROID):
            self._device_type = DEVICE_TYPE_ARM_ANDROID
        elif (self._platform == PLATFORM_OS_LINUX):
            self._device_type = DEVICE_TYPE_ARM_LINUX
        elif (self._platform == PLATFORM_OS_QNX):
            self._device_type = DEVICE_TYPE_ARM_QNX
        else:
            raise Exception("device: Invalid platform !!!", platform)

        return

    def __str__(self):
        return (('[Device Name:%s ' % self._device_name) +
                ('Device ID:%s ' % self._comm_id) +
                ('HOST NAME:%s ' % self.host_name) +
                ('Device DIR:%s]' % self._device_root_dir))

    @property
    def device_name(self):
        return self._device_name

    @property
    def host_name(self):
        return self._host_name

    @property
    def comm_id(self):
        return self._comm_id

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_root_dir(self):
        return self._device_root_dir

    @property
    def host_output_dir(self):
        return self._host_output_dir

    def __mem_log_file(self):
        return os.path.join(self._device_root_dir, MEM_LOG_FILE_NAME)

    def __capture_mem_droid(self, exe_name):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        while time_out > (time.time() - t0):
            ret, version_output, err = self.adb.shell('getprop', ['ro.build.version.release'])
            android_version = version_output[0].strip().split()[0]
            if android_version >= "8.0.0":
                ret, ps_output, err = self.adb.shell('ps', ['-A', '|', 'grep', ps_name])
            else:
                ret, ps_output, err = self.adb.shell('ps', ['|', 'grep', ps_name])
            if ps_output:
                ps_pid = REGX_SPACES.split(ps_output[0].strip())[1]
                logger.debug(ps_output)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break
            if ps_pid is not None:
                break

        assert ps_pid, "ERROR: Could not find the Process ID of {0}".format(exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug("Capturing memory usage of {0} with PID {1}".format(exe_name, ps_pid))
        logger.debug("Time required to determine the PID:{0}".format((time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug("Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            self.adb.shell('dumpsys', ['meminfo', ps_pid, create_or_append, mem_log_file])
            num_of_samples += 1
        return

    def __capture_mem_le(self, exe_name):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        while time_out > (time.time() - t0):
            ret, output_pid, err = self.adb.shell('ps', ['-A', '|', 'grep', ps_name])
            if output_pid:
                ps_pid = REGX_SPACES.split(output_pid[0].strip())[0]
                logger.debug(output_pid)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break

        assert ps_pid, "Could not find the Process ID of {0}".format(exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug("Capturing memory usage of {0} with PID {1}".format(exe_name, ps_pid))
        logger.debug("Time required to determine the PID:{0}".format((time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug("Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            self.adb.shell('cat', ['/proc/{}/smaps'.format(ps_pid), '|', 'cat', '>>', mem_log_file])
            self.adb.shell('echo', ['====', create_or_append, mem_log_file])
            num_of_samples += 1
        return

    def __capture_mem_qnx(self, exe_name):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        while time_out > (time.time() - t0):
            ret, output_pidin, err = self.adb.shell('pidin', ['a', '|', 'grep', ps_name, '| grep -v grep'])
            if output_pidin:
                ps_pid = REGX_SPACES.split(output_pidin[0].strip())[0]
                logger.debug(output_pidin)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break
            if ps_pid is not None:
                break

        assert ps_pid, "Could not find the Process ID of {0}".format(exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug("Capturing memory usage of {0} with PID {1}".format(exe_name, ps_pid))
        logger.debug("Time required to determine the PID:{0}".format((time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug("Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            self.adb.shell('cat', ["/proc/%s/pmap | awk --non-decimal-data -F ',' "
                    "'BEGIN{sum=0} {sum+=$2}END {print sum}' %s %s" % (ps_pid, create_or_append, mem_log_file)])
            num_of_samples += 1
        return

    def execute(self, commands):
        functions = {
            'shell': self.adb.shell,
            'push': self.adb.push,
            'pull': self.adb.pull
        }
        for b_cmd in commands:
            ret, out, err = functions[b_cmd.function](*b_cmd.params)
            if ret != 0:
                logger.error(out)
                logger.error(err)
                raise AdbShellCmdFailedException
        return

    def start_measurement(self, benchmark):
        if benchmark.measurement.type == MEASURE_MEM:
            if self._mem_proc is None:
                logger.info("starting memory capture in a parallel process")
                if(self._platform == PLATFORM_OS_ANDROID):
                    logger.info("Android platform")
                    self._mem_proc = Process(target=self.__capture_mem_droid, args=(benchmark.exe_name,))
                    self._mem_proc.start()
                elif(self._platform == PLATFORM_OS_LINUX):
                    logger.info("Linux Embedded")
                    self._mem_proc = Process(target=self.__capture_mem_le, args=(benchmark.exe_name,))
                    self._mem_proc.start()
                elif(self._platform == PLATFORM_OS_QNX):
                    logger.info("QNX platform")
                    self._mem_proc = Process(target=self.__capture_mem_qnx, args=(benchmark.exe_name,))
                    self._mem_proc.start()
                else:
                    raise Exception("start_measurement: Invalid platform !!!", self.platform)
            else:
                logger.info("memory capture is already started")
        return

    def stop_measurement(self, benchmark):
        if benchmark.measurement.type == MEASURE_MEM:
            if self._mem_proc is not None:
                self._mem_proc.terminate()
                self._mem_proc = None
                logger.info("memory capture is terminated")
                self.adb.pull(self.__mem_log_file(), benchmark.host_result_dir)
        return
