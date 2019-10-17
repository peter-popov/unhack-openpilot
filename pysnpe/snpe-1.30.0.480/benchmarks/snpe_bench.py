#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import print_function
import logging
import sys
import os
from time import sleep


if os.path.isdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../lib/benchmarks/')):
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../lib/benchmarks/'))
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../lib'))
else:
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'snpebm'))
from snpebm import snpebm_parser, snpebm_config, snpebm_bm, snpebm_constants,  snpebm_md5, snpebm_writer, snpebm_device
from common_utils import exceptions
from common_utils.constants import LOG_FORMAT

logger = None


def _find_shell_binary_on_target(device):
    sh_path = '/system/bin/sh'
    if device.adb.check_file_exists(sh_path) is False:
        sh_cmd = 'which sh'
        sh_path = ''
        ret, out, err = device.adb.shell(sh_cmd)
        sh_path = out[0]
        if ret != 0:
            sh_path = ''
        if sh_path == '' or "not found" in sh_path:
            logger.error('Could not find md5 checksum binary on device.')
            sh_path = ''
    return sh_path.rstrip()

def _config_logger(debug, device_id=None):
    global logger
    log_prefix = snpebm_constants.SNPE_BENCH_NAME + ('_'+device_id if device_id else "")
    logger = logging.getLogger(log_prefix)
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=LOG_FORMAT)


def snpe_bench(program_name, args_list, device_msm_os_dict=None):
    try:
        args_parser = snpebm_parser.ArgsParser(program_name,args_list)
        if args_parser.device_id_override:
            _config_logger(args_parser.debug_enabled, args_parser.device_id_override[0])
        else:
            _config_logger(args_parser.debug_enabled)
        logger.info("Running {0} with {1}".format(snpebm_constants.SNPE_BENCH_NAME, args_parser.args))
        config = snpebm_config.ConfigFactory.make_config(args_parser.config_file_path, args_parser.output_basedir_override, args_parser.device_id_override, args_parser.host_name,
                                           args_parser.run_on_all_connected_devices_override, args_parser.device_os_type_override,
                                           args_parser.userbuffer_mode, args_parser.perfprofile, args_parser.profilinglevel, args_parser.enable_init_cache)
        if config is None:
            return
        logger.info(config)

        # Dictionary is {"cpu_arm_all_SNPE_BENCH_NAMEMemory":ZdlSnapDnnCppDroidBenchmark object}
        benchmarks, results_dir = snpebm_bm.BenchmarkFactory.make_benchmarks(config)
        # Now loop through all the devices and run the benchmarks on them
        for device_id in config.devices:
            device = snpebm_device.DeviceFactory.make_device(device_id, config)
            if not device.adb.is_device_online():
                raise exceptions.AdbShellCmdFailedException('Could not run simple command on device %s' % device.comm_id)
            # don't need to capture retcode/err since error handling is
            # done in the fuction and behaviour depends on 'fatal' argument
            _, device_info, _ = device.adb.get_device_info(fatal=((args_parser.device_os_type_override != 'le' and args_parser.device_os_type_override != 'le64')))
            logger.debug("Perform md5 checksum on %s" % device_id)
            snpebm_md5.perform_md5_check(device, [item for sublist in config.artifacts.values() for item in sublist]
                                         + config.dnn_model.artifacts)
            logger.debug("Artifacts on %s passed checksum" % device_id)
            sh_path = _find_shell_binary_on_target(device)

            benchmarks_ran = []
            # Run each benchmark on device, and pull results
            for bm in benchmarks:
                matches = [value for key, value in snpebm_constants.RUNTIMES.items()
                           if bm.runtime_flavor_measure.startswith(key)]
                if matches:
                    logger.info('Running on {}'.format(bm.runtime_flavor_measure))
                    bm.sh_path = sh_path
                    # running iterations of the same runtime.  Two possible failure cases:
                    # 1. Say GPU runtime is not available
                    # 2. Transient failure
                    # For now, for either of those cases, we will mark the whole runtime
                    # as bad, so I break out of for loop as soon as a failure is detected

                    # if init caching is enabled run for one extra run for DSP and AIP runtime flavors
                    if args_parser.enable_init_cache and bm.runtime_flavor_measure.startswith(snpebm_constants.ENABLE_CACHE_SUPPORTED_RUNTIMES):
                        iterations = config.iterations + 1
                    # if init caching is disabled maintain the same number of runs
                    else:
                        iterations = config.iterations
                    for i in range(iterations):
                        logger.info("Run " + str(i + 1))
                        bm.run_number = i + 1
                        try:
                            device.execute(bm.pre_commands)
                            device.start_measurement(bm)
                            # Sleep to let things cool off
                            if args_parser.sleep != 0:
                                logger.debug("Sleeping: " + str(args_parser.sleep))
                                sleep(args_parser.sleep)
                            device.execute(bm.commands)
                            device.stop_measurement(bm)
                            device.execute(bm.post_commands)
                        except exceptions.AdbShellCmdFailedException as e:
                            logger.warning('Failed to perform benchmark for %s.' % bm.runtime_flavor_measure)
                            break
                        finally:
                            device.stop_measurement(bm)

                        bm.process_results()
                    else:  # Ran through iterations without failing
                        benchmarks_ran.append((bm.runtime_flavor_measure, bm))
                else:
                    logger.error("The specified runtime with  %s is not a supported runtime,"
                                 " benchmarks will not be running with this runtime" % bm.runtime_flavor_measure)

            if len(benchmarks_ran) == 0:
                logger.error('None of the selected benchmarks ran, therefore no results reported')
                sys.exit(snpebm_constants.ERRNUM_NOBENCHMARKRAN_ERROR)
            else:
                os_type, device_meta = device.adb.getmetabuild()
                metabuild_id = ('Meta_Build_ID', device_meta)
                device_info.append(metabuild_id)

                os_type = ('OS_Type', os_type)
                device_info.append(os_type)

                if device_msm_os_dict is not None:
                    chipset = ('Chipset', device_msm_os_dict[device_id][1])
                    if device_msm_os_dict[device_id][2] == '':
                        OS = ('OS', device_msm_os_dict[device_id][3])
                    else:
                        OS = ('OS', device_msm_os_dict[device_id][2])
                    device_info.append(chipset)
                    device_info.append(OS)

                snpe_version = benchmarks_ran[0][1].get_snpe_version(config)
                basewriter = snpebm_writer.Writer(snpe_version, benchmarks_ran, config, device_info, args_parser.sleep)

                if args_parser.generate_json:
                    basewriter.writejson(os.path.join(results_dir, "benchmark_stats_{0}.json".format(config.name)))
                basewriter.writecsv(os.path.join(results_dir, "benchmark_stats_{0}.csv".format(config.name)))

    except exceptions.ConfigError as ce:
        print(ce)
        sys.exit(snpebm_constants.ERRNUM_CONFIG_ERROR)
    except exceptions.AdbShellCmdFailedException as ae:
        print(ae)
        sys.exit(snpebm_constants.ERRNUM_ADBSHELLCMDEXCEPTION_ERROR)
    except Exception as e:
        print(e)
        sys.exit(snpebm_constants.ERRNUM_GENERALEXCEPTION_ERROR)


if __name__ == "__main__":
    snpe_bench(sys.argv[0], sys.argv[1:])
