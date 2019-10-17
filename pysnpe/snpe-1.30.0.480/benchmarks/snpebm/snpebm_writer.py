#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import absolute_import
import json
import csv
import logging
from collections import OrderedDict
from .snpebm_constants import MEASURE_TIMING, MEASURE_MEM

logger = logging.getLogger(__name__)

class Writer(object):
    SPACE = ' '
    NOT_AVAILABLE = "N/A"
    SDK_VERSION_HEADER = "SNPE SDK version:"
    CONFIG_HEADER = "Configuration used:"
    DEVICE_INFO_HEADER = "Device Info:"
    UNITS = {MEASURE_TIMING: "us", MEASURE_MEM: "kB"}

    def __init__(self, snpe_sdk_version, benchmarks, config, device_info, sleeptime):
        self._snpe_sdk_version = snpe_sdk_version
        self._tables = {}
        self._config = config
        self._device_info = device_info
        self._sleeptime = sleeptime
        for run_flavor_measure, bm in benchmarks:
            if bm.measurement.type not in self._tables:
                self._tables.update({bm.measurement.type: []})
            self._tables[bm.measurement.type].append(bm)

    def __write_csv_metadata(self, writer):
        writer.writerow([self.SDK_VERSION_HEADER, self._snpe_sdk_version])
        writer.writerow([])
        writer.writerow([self.CONFIG_HEADER])
        rows = self._config.csvrows
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow([self.DEVICE_INFO_HEADER])
        writer.writerows(self._device_info)
        writer.writerow([])
        return

    def writecsv(self, csv_file_path):
        csv_file = open(csv_file_path, 'wt')
        try:
            writer = csv.writer(csv_file)
            self.__write_csv_metadata(writer)
            for measure_type, bms in self._tables.items():
                header_row = [self.SPACE]
                header_row_2 = [self.SPACE]
                data_rows = OrderedDict()
                bmcount = 0
                for bm in bms:
                    if (self._sleeptime == 0):
                        header_row += ["{}({} runs)".format(bm.runtime_flavor_measure, self._config.iterations), self.SPACE, self.SPACE, self.SPACE]
                    else:
                        header_row += ["{}({} runs, {}s sleep)".format(bm.runtime_flavor_measure, self._config.iterations, self._sleeptime),
                                       self.SPACE, self.SPACE, self.SPACE]
                    unit = self.UNITS[measure_type]
                    header_row_2 += ["avg ({0})".format(unit), "max ({0})".format(unit), "min ({0})".format(unit), "runtime"]
                    avg_dict = bm.measurement.average
                    max_dict = bm.measurement.max
                    min_dict = bm.measurement.min
                    runtime_dict = bm.measurement.runtime
                    for channel, raw_data in avg_dict.items():
                        if channel not in data_rows:
                            data_rows[channel] = [channel]
                            for i in range(0, bmcount):
                                #Add padding as needed.  One pad for "avg", one for "max", one for "min", one for "runtime"
                                data_rows[channel] += [self.SPACE]
                                data_rows[channel] += [self.SPACE]
                                data_rows[channel] += [self.SPACE]
                                data_rows[channel] += [self.SPACE]
                        data_rows[channel] += [raw_data]

                    for channel, raw_data in max_dict.items():
                        if channel not in data_rows:
                            logger.error("Error: invalid data")
                            return
                        data_rows[channel] += [raw_data]

                    for channel, raw_data in min_dict.items():
                        if channel not in data_rows:
                            logger.error("Error: invalid data")
                            return
                        data_rows[channel] += [raw_data]

                    for channel, raw_data in runtime_dict.items():
                        if channel not in data_rows:
                            logger.error("Error: invalid data")
                            return
                        data_rows[channel] += [raw_data]

                    #Add padding for unreported measurement when setProfiling is set to basic
                    #Remove when setProfiling is implemented on all runtimes
                    for channel in data_rows:
                        if channel not in avg_dict:
                            data_rows[channel] += [self.SPACE]
                        if channel not in max_dict:
                            data_rows[channel] += [self.SPACE]
                        if channel not in min_dict:
                            data_rows[channel] += [self.SPACE]
                        if channel not in runtime_dict:
                            data_rows[channel] += [self.SPACE]

                    bmcount = bmcount + 1
                writer.writerow([""] + [""] + header_row)
                writer.writerow([""] + [""] + header_row_2)
                for channel in data_rows.keys():
                    writer.writerow([""] + [""] + data_rows[channel])
                writer.writerow([])

        finally:
            csv_file.close()

    def __write_json_metadata(self, writer):
        writer.update({self.SDK_VERSION_HEADER: self._snpe_sdk_version})
        for item in self._device_info:
            writer.update({item[0]: item[1]})
        jsonrows = self._config.jsonrows
        writer.update(jsonrows)
        return

    def writejson(self, json_file_path):
        bm_data = {}
        run_data = {}
        try:
            self.__write_json_metadata(bm_data)
        except Exception as e:
            logger.error(e, "\nError:Could not write meta data of benchmarks in raw_json")
            return
        for measure_type, bms in self._tables.items():
            if self.UNITS[measure_type] == "us":
                suffix = "Time"
            else:
                suffix = "Size"
            try:
                for bm in bms:
                    runtime = bm.runtime_flavor_measure.split('_timing')[0]
                    run_data_individual = {}
                    avg_dict = bm.measurement.average
                    max_dict = bm.measurement.max
                    min_dict = bm.measurement.min
                    runtime_dict = bm.measurement.runtime
                    for channel, raw_data in avg_dict.items():
                        if channel in runtime_dict:
                            actual_runtime = runtime_dict[channel]
                        else:
                            logger.error("Error:Actual runtime information is not present for the layer")
                            return
                        if channel in max_dict:
                            max_val_channel = max_dict[channel]
                        else:
                            logger.error("Error: Maximum amount of inference time is not present for the layer")
                            return
                        if channel in min_dict:
                            min_val_channel = min_dict[channel]
                        else:
                            logger.error("Error: Minimum amount of inference time is not present for the layer")
                            return
                        run_data_individual.update({channel: {"Runtime": actual_runtime, "Avg_"+suffix: raw_data, "Max_"+suffix: max_val_channel, "Min_"+suffix: min_val_channel}})
                    run_data[runtime] = run_data_individual
                bm_data['Execution_Data'] = run_data
                bm_data['Units'] = self.UNITS[measure_type]
            except Exception as e:
                logger.error(e)
                logger.error("Error:Could not write benchmark scores in raw_json")
                return
        with open(json_file_path, 'w') as outfile:
            json.dump(bm_data, outfile, indent=4)
