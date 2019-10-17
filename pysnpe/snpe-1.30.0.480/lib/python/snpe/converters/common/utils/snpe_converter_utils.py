#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy
import sys


# -----------------------------------------------------------------------------------------------------
#   Common Functions
# -----------------------------------------------------------------------------------------------------

def sanitize_args(args, args_to_ignore=[]):
    sanitized_args = []
    for k, v in list(vars(args).items()):
        if k in args_to_ignore:
            continue
        sanitized_args.append('{}={}'.format(k, v))
    return "{} {}".format(sys.argv[0].split('/')[-1], ' '.join(sanitized_args))


def get_string_from_txtfile(filename):
    if not filename:
        return filename
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r') as myfile:
                file_data = myfile.read()
            return file_data
        except Exception as e:
            logger.error("Unable to open file %s: %s" % (filename, e))
            sys.exit(-1)
    else:
        logger.error("File %s: must be a text file." % filename)
        sys.exit(-1)


# -----------------------------------------------------------------------------------------------------
#   Logging
# -----------------------------------------------------------------------------------------------------
# @deprecated
# TODO: remove once cleanup of converters is done to use method below instead
logger = logging.getLogger(__name__)


def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- end of deprecated ---


class SNPEUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
            # Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)


LOGGER = None
LOG_LEVEL = logging.INFO

# Custom Logging
logging.DEBUG_3 = DEBUG_LEVEL_IR_TO_BACKEND = 11
logging.DEBUG_2 = DEBUG_LEVEL_IR_OPTIMIZATION = 12
logging.DEBUG_1 = DEBUG_LEVEL_CONVERTER_TO_IR = 13

# add the custom log-levels
logging.addLevelName(DEBUG_LEVEL_IR_TO_BACKEND, "DEBUG_3")
logging.addLevelName(DEBUG_LEVEL_IR_OPTIMIZATION, "DEBUG_2")
logging.addLevelName(DEBUG_LEVEL_CONVERTER_TO_IR, "DEBUG_1")


def setup_logging(args):
    global LOGGER
    global LOG_LEVEL

    if args.debug == -1:  # --debug is not set
        LOG_LEVEL = logging.INFO
    elif args.debug == 0:  # --debug is set with no specific level. i.e: print every debug message.
        LOG_LEVEL = logging.DEBUG
    elif args.debug == 1:
        LOG_LEVEL = logging.DEBUG_1
    elif args.debug == 2:
        LOG_LEVEL = logging.DEBUG_2
    elif args.debug == 3:
        LOG_LEVEL = logging.DEBUG_3
    else:
        log_assert("Unknown debug level provided. Got {}", args.debug)

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(LOG_LEVEL)
    LOGGER.addHandler(handler)


def log_assert(cond, msg, *args):
    assert cond, msg.format(*args)


def log_debug(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))


def log_debug1(msg, *args):
    def debug1(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_1):
            LOGGER._log(logging.DEBUG_1, msg, args, kwargs)
    debug1(msg.format(*args))


def log_debug2(msg, *args):
    def debug2(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_2):
            LOGGER._log(logging.DEBUG_2, msg, args, kwargs)
    debug2(msg.format(*args))


def log_debug3(msg, *args):
    def debug3(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_3):
            LOGGER._log(logging.DEBUG_3, msg, args, kwargs)
    debug3(msg.format(*args))


def log_error(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))


def log_info(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))


def log_warning(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))


def get_log_level():
    global LOG_LEVEL
    return LOG_LEVEL


def is_log_level_debug():
    global LOG_LEVEL
    return LOG_LEVEL == logging.DEBUG


# -----------------------------------------------------------------------------------------------------
#   Translation Helpers
# -----------------------------------------------------------------------------------------------------
"""
Following functions sanitize op/layer type names and attach converter type for registering translations
"""


def get_op_info(type_name):
    """Return the op name and version, if specified"""
    op_data = str(type_name).split('-')
    if len(op_data) > 1:
        return [op_data[0], int(op_data[1])]
    op_data.append(0)
    return op_data


def op_type(type_name):
    """Return the actual onnx op name"""
    data = get_op_info(type_name)
    return data[0]


def converter_type(type_name, src_converter):
    """Convert an src converter type name string to a namespaced format"""
    return src_converter + '_' + (op_type(type_name)).lower()
