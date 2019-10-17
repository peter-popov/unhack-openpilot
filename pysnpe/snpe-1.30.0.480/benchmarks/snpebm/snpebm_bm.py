#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from .snpebm_config_restrictions import *
from .snpebm_parser import LogParserFactory
from collections import OrderedDict
import time
import os
import logging

logger = logging.getLogger(__name__)

class BenchmarkStat(object):
    def __init__(self, log_parser, stat_type, caching_enabled=None):
        self._stats = []
        self._log_parser = log_parser
        self._type = stat_type
        self._cache = caching_enabled

    def __iter__(self):
        return self._stats.__iter__()

    @property
    def stats(self):
        return self._stats

    @property
    def type(self):
        return self._type

    def _process(self, input_dir):
        data_frame = self._log_parser.parse(input_dir)
        self._stats.append(data_frame)

    @property
    def average(self):
        avg_dict = OrderedDict()
        # ignoring the first run if caching is enabled
        if self._cache:
            stats = self._stats[1:]
        # considering the all runs if caching is disabled
        else:
            stats = self._stats
        for stat in stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in avg_dict:
                    avg_dict[channel][0] += _sum
                    avg_dict[channel][1] += _len
                else:
                    avg_dict[channel] = [_sum, _len]
        avgs = OrderedDict()
        for channel in avg_dict:
            avgs[channel] = int(avg_dict[channel][0] / avg_dict[channel][1])
        return avgs

    @property
    def max(self):
        max_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in max_dict:
                    max_dict[channel] = max(max_dict[channel], _max)
                else:
                    max_dict[channel] = _max
        return max_dict

    @property
    def min(self):
        min_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel in min_dict:
                    min_dict[channel] = min(min_dict[channel], _min)
                else:
                    min_dict[channel] = _min
        return min_dict

    @property
    def runtime(self):
        runtime_dict = OrderedDict()
        for stat in self._stats:
            for channel, _sum, _len, _max, _min, _runtime in stat:
                if channel not in runtime_dict:
                    runtime_dict[channel] = _runtime
        return runtime_dict

class BenchmarkCommand(object):
    def __init__(self, function, params):
        self.function = function
        self.params = params

class BenchmarkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def make_benchmarks(config):
        assert config, "config is required"
        assert config.measurement_types_are_valid(), "You asked for %s, but only these types of measurements are supported: %s"%(config.measurements,CONFIG_VALID_MEASURMENTS)
        host_result_dirs = {}
        for arch in config.architectures:
            if arch == ARCH_AARCH64 or arch == ARCH_ARM:
                if 'droid' not in host_result_dirs:
                    host_result_dirs['droid'] = \
                        SnapDnnCppDroidBenchmark.create_host_result_dir(config.host_resultspath)

        benchmarks = []
        for runtime, flavor in config.return_valid_run_flavors():
            for measurement in config.measurements:
                dev_bin_path = config.get_device_artifacts_bin(runtime)
                dev_lib_path = config.get_device_artifacts_lib(runtime)
                exe_name = config.get_exe_name(runtime)
                parser = LogParserFactory.make_parser(measurement, config)
                cache = False
                # to check for each variant and run with cache option if cache is set and runtime is DSP or AIP
                if config.enable_init_caching and runtime.startswith(ENABLE_CACHE_SUPPORTED_RUNTIMES):
                    cache = True
                benchmark = SnapDnnCppDroidBenchmark(
                    dev_bin_path,
                    dev_lib_path,
                    exe_name,
                    config.dnn_model.device_rootdir,
                    os.path.basename(config.dnn_model.dlc),
                    config.dnn_model.input_list_name,
                    config.userbuffer_mode,
                    config.perfprofile,
                    config.profilinglevel,
                    config.cpu_fallback,
                    config.host_rootpath,
                    cache
                )
                benchmark.measurement = BenchmarkStat(parser, measurement, cache)
                benchmark.runtime = runtime
                benchmark.host_output_dir = host_result_dirs['droid']
                benchmark.name = flavor
                benchmarks.append(benchmark)

        return benchmarks, host_result_dirs['droid']


class SnapDnnCppDroidBenchmark(object):
    @staticmethod
    def create_host_result_dir(host_output_dir):
        # Create results output dir, and a "latest_results" that links to it
        _now = time.localtime()[0:6]
        _host_output_datetime_dir = os.path.join(host_output_dir, SNPE_BENCH_OUTPUT_DIR_DATETIME_FMT % _now)
        os.makedirs(_host_output_datetime_dir)
        sim_link_path = os.path.join(host_output_dir, LATEST_RESULTS_LINK_NAME)
        if os.path.islink(sim_link_path):
            os.remove(sim_link_path)
        os.symlink(os.path.relpath(_host_output_datetime_dir,host_output_dir), sim_link_path)
        return _host_output_datetime_dir

    def __init__(self, exe_dir, dep_lib_dir, exe_name, model_dir, container_name, input_list_name, userbuffer_mode, perfprofile, profilinglevel, cpu_fallback, host_rootpath, cache=None):
        assert model_dir, "model dir is required"
        assert container_name, "container is required"
        assert input_list_name, "input_list is required"
        self._exe_dir = exe_dir
        self._model_dir = model_dir
        self._dep_lib_dir = dep_lib_dir
        self._exe_name = exe_name
        self._container = container_name
        self._input_list = input_list_name
        self.output_dir = 'output'
        self.host_output_dir = None
        self.host_result_dir = None
        self.debug = False
        self.runtime = RUNTIME_CPU
        self.name = None
        self.run_number = 0
        self.measurement = None
        self.sh_path ='/system/bin/sh'
        self.userbuffer_mode = userbuffer_mode
        self.perfprofile = perfprofile
        self.profilinglevel = profilinglevel
        self.cpu_fallback = cpu_fallback
        self.host_rootpath = host_rootpath
        self.cache = cache

    @property
    def runtime_flavor_measure(self):
        if self.name != '':
            return '{}_{}_{}'.format(self.runtime, self.name, self.measurement.type)
        else:
            return '{}_{}'.format(self.runtime, self.measurement.type)

    @property
    def exe_name(self):
        return SNPE_BATCHRUN_EXE

    def __create_script(self):
        cmds = ['export LD_LIBRARY_PATH=' + self._dep_lib_dir + ':$LD_LIBRARY_PATH',
                'export ADSP_LIBRARY_PATH=\"' + self._dep_lib_dir + \
                '/../../dsp/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\"',
                'cd ' + self._model_dir,
                'rm -rf ' + self.output_dir]
        run_cmd = "{0} --container {1} --input_list {2} --output_dir {3}" \
            .format(os.path.join(self._exe_dir, self._exe_name), self._container, self._input_list, self.output_dir)
        # add runtime arg
        run_cmd += RUNTIMES[self.runtime]
        # add option userbuffer mode
        if self.name in BUFFER_MODES:
            run_cmd += BUFFER_MODES[self.name]
        if self.debug:
            run_cmd += " --debug"
        if self.perfprofile:
            run_cmd += " --perf_profile " + self.perfprofile
        if self.profilinglevel:
            run_cmd += " --profiling_level " + self.profilinglevel
        if self.cpu_fallback:
            run_cmd += " --enable_cpu_fallback"
        if self.cache:
            run_cmd += " --enable_init_cache"
        cmds.append(run_cmd)
        cmd_script_path = os.path.join(self.host_rootpath, SNPE_BENCH_SCRIPT)
        if os.path.isfile(cmd_script_path):
            os.remove(cmd_script_path)
        with open(cmd_script_path, 'w') as cmd_script:
            cmd_script.write('#!' + self.sh_path + '\n')
            for ln in cmds:
                cmd_script.write(ln + '\n')
        os.chmod(cmd_script_path, 0o555)
        return cmd_script_path

    @property
    def pre_commands(self):
        self.host_result_dir = os.path.join(
            self.host_output_dir,
            self.measurement.type, '_'.join(filter(''.__ne__, (self.runtime, self.name))),
            "Run" + str(self.run_number)
        )
        os.makedirs(self.host_result_dir)
        cmd_script = self.__create_script()
        diag_rm_files = os.path.join(self._model_dir, SNPE_BENCH_DIAG_REMOVE)
        return [BenchmarkCommand('shell', ['rm', ['-f', diag_rm_files]]),
                BenchmarkCommand('push', [cmd_script, self._exe_dir])]

    @property
    def commands(self):
        return [BenchmarkCommand('shell', ['sh', [os.path.join(self._exe_dir, SNPE_BENCH_SCRIPT)]])]

    @property
    def post_commands(self):
        if self.host_output_dir is None:
            return []
        device_output_dir = os.path.join(self._model_dir, self.output_dir)
        # now will also pull the script file used to generate the results
        return [BenchmarkCommand('shell', ['chmod', ['777', device_output_dir]]),
                BenchmarkCommand('pull', [os.path.join(self._exe_dir,SNPE_BENCH_SCRIPT),
                                                       self.host_result_dir]),
                BenchmarkCommand('pull', [os.path.join(device_output_dir,SNPE_BENCH_DIAG_OUTPUT_FILE),
                                                       self.host_result_dir])]
    def get_snpe_version(self, config):
        snpe_version_parser = LogParserFactory.make_parser(MEASURE_SNPE_VERSION, config)
        return snpe_version_parser.parse(self.host_result_dir)

    def process_results(self):
        assert os.path.isdir(self.host_result_dir), "ERROR: no host result directory"
        self.measurement._process(self.host_result_dir)

