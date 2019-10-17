#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#  All contributions by the University of California:
#  Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
#  All rights reserved.
#
#  All other contributions:
#  Copyright (c) 2014, 2015, the respective contributors
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import collections
import copy  # deep copy
import logging
import math
import numpy
import pprint  # pretty print for dicts
import random
import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
import yaml
from functools import reduce
from google.protobuf import text_format
from os import path

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2


from snpe.converters.common.utils import code_to_message
# Importing axis tracking types
from snpe.converters.common.utils import snpe_axis_transformer
from snpe.converters.common.utils import snpe_converter_utils
from snpe.converters.common.utils.snpe_converter_utils import SNPEUtils

AxisAnnotation = snpe_axis_transformer.AxisAnnotation

permute_id = 0
def get_permute_id():
    global permute_id
    temp = permute_id
    permute_id += 1
    return temp


snpeUtils = SNPEUtils()


def getArgs():
    logger = logging.getLogger()
    logger.debug("Parsing the arguments")

    parser = argparse.ArgumentParser(
        description=
        'Script to convert caffe protobuf configuration into a DLC file.')
    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--caffe_txt', type=str, required=True,
                          help='Input caffe proto txt configuration file')

    optional = parser.add_argument_group('optional arguments')


    optional.add_argument('-b','--caffe_bin', type=str,
                          help='Input caffe binary file containing the weight data')
    optional.add_argument('-d', '--dlc', type=str,
                          help='Output DLC file containing the model. If not specified, the data will be written to a file with same name as the caffetxt file with a .dlc extension')
    optional.add_argument('--copyright_file', type=str,
                          help='Path to copyright file. If provided, the content of the file will be added to the dlc.')
    # The "omit_preprocessing" argument populates a variable called "enable_preprocessing" with its opposite value, so that
    # we avoid "double-negatives" all over the code when using it.
    optional.add_argument('--omit_preprocessing', dest="enable_preprocessing", action="store_const", const=False, default=True,
                          help="If specified, converter will disable preprocessing specified by a data layer transform_param or any preprocessing command line options")
    optional.add_argument('--encoding', type=str, choices=['argb32', 'rgba', 'nv21', 'bgr'], default='bgr',
                          help='Image encoding of the source images. Default is bgr if not specified')
    optional.add_argument('--input_size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
                          help='Dimensions of the source images for scaling, if different from the network input.')
    optional.add_argument('--model_version', type=str,
                          help='User-defined ASCII string to identify the model, only first 64 bytes will be stored')
    optional.add_argument('--disable_batchnorm_folding', dest="disable_batchnorm_folding", action="store_true",
                          help="If not specified, converter will try to fold batchnorm into previous convolution layer")
    optional.add_argument('--in_layer', type=str, action='append', dest='input_layers',
                          help='Name of the input layer')
    optional.add_argument('--in_type', type=str, choices=['default', 'image', 'opaque'], action='append', dest='input_types',
                          help='Type of data expected by input layer. Type is default if not specified.')
    optional.add_argument('--validation_target', nargs=2, metavar=('RUNTIME_TARGET','PROCESSOR_TARGET'), default = [], action=snpe_validation_utils.ValidateTargetArgs,
                          help="A combination of processor and runtime target against which model will be validated."
                               "Choices for RUNTIME_TARGET: {cpu, gpu, dsp}."
                               "Choices for PROCESSOR_TARGET: {snapdragon_801, snapdragon_820, snapdragon_835}."
                               "If not specified, will validate model against {snapdragon_820, snapdragon_835} across all runtime targets.")
    optional.add_argument('--strict', dest="enable_strict_validation", action="store_true", default=False,
                          help="If specified, will validate in strict mode whereby model will not be produced if it violates constraints of the specified validation target."
                               "If not specified, will validate model in permissive mode against the specified validation target.")
    optional.add_argument("--verbose", dest="verbose", action="store_true",
                          help="Verbose printing", default = False)

    args = parser.parse_args()
    if args.dlc is None:
        filename, fileext = os.path.splitext(os.path.realpath(args.caffe_txt))
        args.dlc = filename + ".dlc"

    return args

#------------------------------------------------------------------------------
#   Specify caffe layers' ordered axes
#   A layer type, not listed here, will assume axis order = N, C, H, W
#------------------------------------------------------------------------------
default_caffe_axes = [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH]
caffe_layer_axes = snpe_axis_transformer.LayerOrderedAxes("Caffe",default_caffe_axes)

# Layers with NONTRIVIAL input/output axis order
caffe_layer_axes.add_axis_order('RESHAPE', [AxisAnnotation.NONTRIVIAL])
caffe_layer_axes.add_axis_order('FLATTEN', [AxisAnnotation.NONTRIVIAL])
caffe_layer_axes.add_axis_order('SSDDETECTIONOUTPUT', [AxisAnnotation.NONTRIVIAL])
caffe_layer_axes.add_axis_order('DETECTIONOUTPUT', [AxisAnnotation.NONTRIVIAL])

# Layers with ANY input/output axis order

# 1D/3D Batchnorm support
caffe_layer_axes.add_axis_order('BATCHNORM', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('BATCHNORM', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])

# Add both 4d and 3d axis order for concat layer
caffe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

caffe_layer_axes.add_axis_order('LSTM',
                                [AxisAnnotation.TIME, AxisAnnotation.BATCH, AxisAnnotation.FEATURE],
                                [AxisAnnotation.TIME, AxisAnnotation.BATCH, AxisAnnotation.FEATURE])
caffe_layer_axes.add_axis_order('LSTM',
                                [AxisAnnotation.TIME, AxisAnnotation.BATCH],
                                [AxisAnnotation.TIME, AxisAnnotation.BATCH, AxisAnnotation.FEATURE])

# Add both 4d and 3d axis order for tile layer
caffe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

# Add 3d axis order for slice layer
caffe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

# Add 3d axis order for slice layer
caffe_layer_axes.add_axis_order('SLICE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('SLICE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)

# Add 3d axis order for permute layer
caffe_layer_axes.add_axis_order('SSDPERMUTE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('SSDPERMUTE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('PERMUTE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('PERMUTE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('PERMUTE', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

caffe_layer_axes.add_axis_order('INNERPRODUCT',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('INNERPRODUCT',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

caffe_layer_axes.add_axis_order('FCRISTRETTO',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('FCRISTRETTO',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])



# FIXME: A workaround for unsupported feature: 'axis' parameter support of Softmax.
#        Softmax should be published as AxisAnnotation.ANY once 'axis' parameter of Softmax is supported.
caffe_layer_axes.add_axis_order('SOFTMAX', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('SOFTMAX', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])

caffe_layer_axes.add_axis_order('RELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('RELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('PRELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('PRELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('SIGMOID', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('SIGMOID', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('TANH', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('TANH', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('ELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('ELU', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

caffe_layer_axes.add_axis_order('ELTWISE', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('ELTWISE', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])

#------------------------------------------------------------------------------
#   Specify SNPE layers' ordered axes
#   A layer type, not listed here, will assume axis order = B, H, W, C
#------------------------------------------------------------------------------
default_snpe_axes = [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL]
snpe_layer_axes = snpe_axis_transformer.LayerOrderedAxes("SNPE", default_snpe_axes)

# Layers with NONTRIVIAL input/output axis order
snpe_layer_axes.add_axis_order('RESHAPE', [AxisAnnotation.NONTRIVIAL])
snpe_layer_axes.add_axis_order('FLATTEN', [AxisAnnotation.NONTRIVIAL])
snpe_layer_axes.add_axis_order('SSDDETECTIONOUTPUT', [AxisAnnotation.NONTRIVIAL])
snpe_layer_axes.add_axis_order('DETECTIONOUTPUT', [AxisAnnotation.NONTRIVIAL])

# Layers with ANY input/output axis order

# 1D/3D Batchnorm support
snpe_layer_axes.add_axis_order('BATCHNORM', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('BATCHNORM',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])

# Add from 4D thru 1D axis order for concat layer
snpe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)
snpe_layer_axes.add_axis_order('CONCAT', [AxisAnnotation.ANY] * 1, [AxisAnnotation.ANY] * 1)

snpe_layer_axes.add_axis_order('LSTM',
                                [AxisAnnotation.BATCH, AxisAnnotation.TIME, AxisAnnotation.FEATURE],
                                [AxisAnnotation.BATCH, AxisAnnotation.TIME, AxisAnnotation.FEATURE])
snpe_layer_axes.add_axis_order('LSTM',
                                [AxisAnnotation.BATCH, AxisAnnotation.TIME],
                                [AxisAnnotation.BATCH, AxisAnnotation.TIME, AxisAnnotation.FEATURE])

# Add from 4D thru 1D axis order for tile layer
snpe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)
snpe_layer_axes.add_axis_order('TILE', [AxisAnnotation.ANY] * 1, [AxisAnnotation.ANY] * 1)

# Add from 4D thru 1D axis order for scale layer
snpe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)
snpe_layer_axes.add_axis_order('SCALE', [AxisAnnotation.ANY] * 1, [AxisAnnotation.ANY] * 1)

# Add 4d axis order for slice layer
snpe_layer_axes.add_axis_order('SLICE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)

# Add 4d axis order for permute layer
snpe_layer_axes.add_axis_order('SSDPERMUTE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('PERMUTE', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)

snpe_layer_axes.add_axis_order('INNERPRODUCT',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL],
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('INNERPRODUCT',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

snpe_layer_axes.add_axis_order('FCRISTRETTO',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL],
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('FCRISTRETTO',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

# FIXME: A workaround for unsupported feature: 'axis' parameter support of Softmax.
#        Softmax should be published as AxisAnnotation.ANY once 'axis' parameter of Softmax is supported.
snpe_layer_axes.add_axis_order('SOFTMAX', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('SOFTMAX', [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])

snpe_layer_axes.add_axis_order('RELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('RELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('PRELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('PRELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('SIGMOID',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('SIGMOID',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('TANH',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('TANH',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('ELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('ELU',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

snpe_layer_axes.add_axis_order('ELTWISE', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('ELTWISE', [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#   Simple Layer wrapper class needed for implicit scaling layer
#------------------------------------------------------------------------------
class LayerAdapter(object):
    def __init__(self, name, typ, bottom, top):
        self.name = name
        self.type = typ
        self.bottom = bottom
        self.top = top

#------------------------------------------------------------------------------
#   BufferProxy
#------------------------------------------------------------------------------
class BufferProxy(object):
    def __init__(self):
        # proxy buffers (proxy_buffer, buf)
        self._output_buffer_proxy = {}
        # pending input buffer proxies
        self._pending_input_buffer_proxy = {}
        # output Buffer seen so far
        self._output_buffers = []
        # Some layers needs to be dropped/excluded
        # create mapping but dont register buffers
        # dropout --> 6
        self._excluded_layers_types = [ 'dropout', 'scale', '6' ]
        self._logger = logging.getLogger()

    def dump(self):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_BUFFER_DUMP')(str(len(self._output_buffers))))
        for v in self._output_buffers:
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_BUFFER_PRINT')(str(v)))
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_INPUT_BUFFER_DUMP')(str(len(list(self._pending_input_buffer_proxy.keys())))))
        for k,v in list(self._pending_input_buffer_proxy.items()):
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_KEY_VALUE_PRINT')(str(k), str(v)))
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DUMP')(str(len(list(self._output_buffer_proxy.keys())))))
        for k,v in list(self._output_buffer_proxy.items()):
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_KEY_VALUE_PRINT')(str(k), str(v)))

    def _snapshot_output_buffer_proxy(self):
        self._pending_input_buffer_proxy.clear() # not mandatory
        self._pending_input_buffer_proxy = copy.deepcopy(self._output_buffer_proxy)

    def _gen_alias(self, buf, layername):
        alias = str(layername) + "." + buf
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFER_ALIAS_GEN')(buf, alias))
        return alias

    def _handle_excluded_layers(self, l, model, layer_type):
        # no special handling for scale.
        # scale is skipped. It is only allowed to come after conv.
        # there is another scale in setup_preprocessing() but
        # it is not handled by BufferProxy logic

        # special case for dropout
        # There are 2 options for dropout
        # 1. dropout w/o inplace. in this case
        #    we need to proxy the output buffer of dropout
        #    to the input buffer of the dropout.
        # 2. Dropout that actually have in-place
        #    in this case, this is a classic ignore of dropout
        #    since it does not specify a new output buffer name
        #    so whatever it is reading from, it is writing to. perfect ignore
        if layer_type.lower() == 'dropout':
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFER_DROPOUT_HANDLE'))
            if len(l.top) != len(l.bottom):
                raise RuntimeError(code_to_message.get_error_message('ERROR_CAFFE_NUM_BOTTOM_NOT_EQ_TO_NUM_TOP'))
            idx = 0;
            for t in l.top:
                outname = str(t)
                inpname = str(l.bottom[idx])
                ++idx
                # classic ignore - if it is a true in-place dropout
                if outname.lower() == inpname.lower():
                    continue
                dims =  model.get_buffer_dims(inpname)
                model.register_buffer(outname, list(dims))
                self._output_buffer_proxy[outname] = inpname
                self._pending_input_buffer_proxy[outname] = inpname

    def add_implicit_scale_layer(self, l, model):
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFER_ADD_IMPLICIT_SCALE_LAYER'))
        self.dump()
        # keep a copy of self._handle_excluded_layers
        excluded_layers_copy = copy.deepcopy(self._excluded_layers_types)

        # call add_layer with the modified one
        self._excluded_layers_types.remove('scale')
        self.add_layer(l, model, l.type)

        # create mapping between bottom -> top
        # we only deal with one top/bottom
        self._output_buffer_proxy[l.bottom[0]] = l.top[0]

        # restore it
        self._excluded_layers_types = copy.deepcopy(excluded_layers_copy)
        self.dump()

    def install_buffer_proxy(self, original_buffer, proxy_buffer):
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_INSTALLATION')(proxy_buffer, original_buffer))
        self.dump()

        # create mapping
        self._output_buffer_proxy[original_buffer] = proxy_buffer
        self._pending_input_buffer_proxy[original_buffer] = proxy_buffer

        self.dump()

    def add_layer(self, l, model, layer_type):
        layername = str(l.name)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADDING_LAYER')(layername))

        # for exclude ones, we will register the buffers
        # if there are alias, we will override it
        if layer_type.lower() in self._excluded_layers_types:
            self._handle_excluded_layers(l, model, layer_type)
            return

        # keep a current snapshot before we do anything
        self._snapshot_output_buffer_proxy()

        for t in l.top:
            bufname = str(t)
            self._logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER')(layername, bufname))
            if t in self._output_buffers:
                alias = self._gen_alias(bufname, layername)
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_ALIAS_TO')(layername, bufname, alias))
                # must update model so we can have it registered in the C++ domain
                dims = model.get_buffer_dims(bufname)
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_DESCR')(bufname, str(dims), str(list(dims))))
                model.register_buffer(str(alias), list(dims))

                # need to add proxy
                self._output_buffer_proxy[bufname] = alias
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_PROXY_TO')(layername, bufname, alias))
                # override bufname with alias, this would
                # be get into the _output_buffers below
                bufname = alias

            self._logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER')(layername, bufname))
            self._output_buffers.append(bufname)
            self.dump()

    def input_proxy(self):
        return self._pending_input_buffer_proxy

    def output_proxy(self):
        return self._output_buffer_proxy

    def get_output_proxy_buffer(self, bufname):
        return self._output_buffer_proxy[str(bufname)]

#Map of blob dependencies
#Needed to handle any layer optimizations we might choose to do
class BlobConnectivityMap(object):
    def __init__(self):
        self._blobs = {}
        self._layers_operating_on_blob = {}
        self._dependent_blobs = {}
        self._layers = {}
        self._logger = logging.getLogger()
        self._optimized_out_layers = []
        #Map of all blobs that have been folded into prev blobs and their replacements
        self._optimized_blob_replacements = {}
        self._concat_squash = []

    #fold one layer with single top into another layer with a single top
    def fold_layer(self, folded_layer, folded_into_layer):
        assert(folded_layer is not None and folded_into_layer is not None)
        self._optimized_out_layers.append(folded_layer)
        folded_layer_top = folded_layer.top[0]
        folded_into_layer_top = folded_into_layer.top[0]
        if folded_layer_top != folded_into_layer_top:
            self._optimized_blob_replacements[folded_layer_top] = folded_into_layer_top

    def replace_optimized_blobs(self, layer):
        replaced_bottom = layer.bottom
        for name in self._optimized_blob_replacements:
            replaced_bottom = [self._optimized_blob_replacements[name] if x == name else x for x in replaced_bottom]
        return replaced_bottom

    def add_layer(self, layer):
        #Populate blobs and layer information
        self._layers[layer.name] = {}
        self._layers[layer.name]['layer'] = layer

        # Get input blobs
        for name in layer.bottom:
            if name not in self._blobs:
                self._blobs[name] = {}
                self._blobs[name]['input_to_layers'] = []
                self._blobs[name]['input_to_layers_name'] = []
                self._blobs[name]['output_of_layers'] = []
                self._blobs[name]['output_of_layers_name'] = []
                self._blobs[name]['dependent_blobs'] = []

            self._blobs[name]['input_to_layers'].append(layer)
            self._blobs[name]['input_to_layers_name'].append((layer.name, layer.type))
            # Get output blobs
            for output_blob in layer.top:
                self._blobs[name]['dependent_blobs'].append(output_blob)

        # Get output blobs
        for name in layer.top:
            if name not in self._blobs:
                self._blobs[name] = {}
                self._blobs[name]['input_to_layers'] = []
                self._blobs[name]['input_to_layers_name'] = []
                self._blobs[name]['output_of_layers'] = []
                self._blobs[name]['output_of_layers_name'] = []
                self._blobs[name]['dependent_blobs'] = []

            self._blobs[name]['output_of_layers'].append(layer)
            self._blobs[name]['output_of_layers_name'].append(layer.name)

    def dump_blob_map(self):
        for name in self._blobs:
            print(code_to_message.get_progress_message('INFO_CAFFE_BLOB_NAME_IS_A_DEPENDEE_OF')(name, str(self._blobs[name]['dependent_blobs'])))
            print(code_to_message.get_progress_message('INFO_CAFFE_BLOB_NAME_IS_OPERATED_ON_BY_AND_USED_BY')(name, str(self._blobs[name]['output_of_layers_name']), str(self._blobs[name]['input_to_layers_name'])))

    def get_optimized_out_layers(self):
        return self._optimized_out_layers;

    def check_s_folding(self, layer):
        typ = str(layer.type).upper()
        assert(typ == 'SCALE')

        prev_layer_output_blob = layer.bottom[0]

        bn_layer_seq = None
        scale_layer_seq = layer

        # If a bn layer is an input to the scale layer return it
        output_layer = self._blobs[prev_layer_output_blob]['output_of_layers'][0]
        typ = str(output_layer.type).upper()

        # Check if the previous layer was a batchnorm. If so, scale should only be folded
        # if it is the only layer that depends on that output. Return bn layer in that case
        # other wise return None to force the scale layer to be added
        layers_depending_on_bn_blob = self._blobs[prev_layer_output_blob]['input_to_layers']
        if typ == 'BATCHNORM' and len(layers_depending_on_bn_blob) == 1:
            bn_layer_seq = output_layer

        return (bn_layer_seq, scale_layer_seq)

    def check_bs_folding(self, layer):
        typ = str(layer.type).upper()
        assert(typ == 'BATCHNORM')

        bn_output_blob = layer.top[0]

        bn_layer_seq = layer
        scale_layer_seq = None

        layers_writing_into_bn_blob = self._blobs[bn_output_blob]['output_of_layers']
        index_bn_layer = layers_writing_into_bn_blob.index(bn_layer_seq)
        n1_entry = None
        n1_entry_typ = ''
        if len(layers_writing_into_bn_blob) > (index_bn_layer+1):
            n1_entry = layers_writing_into_bn_blob[index_bn_layer + 1]
            n1_entry_typ = str(n1_entry.type).upper()

        if n1_entry_typ == 'SCALE':
            scale_layer_seq = n1_entry

        if scale_layer_seq is not None:
            if len(scale_layer_seq.bottom) > 1:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
            #Bn + scale consecutive seq found and write into same output buffer
            #scale can be folded into prev bn layer
            self.fold_layer(scale_layer_seq, bn_layer_seq)
            return (bn_layer_seq, scale_layer_seq)
        else:
            #Case where there is no scale layer writing into same bn blob
            layers_depending_on_bn_blob = self._blobs[bn_output_blob]['input_to_layers']
            #Check if there is more than one layer depending on
            #bn blob as that would mean that we can't fold scale without impacting
            #other layer
            if len(layers_depending_on_bn_blob) > 1:
                #More than two layers depend on bn blob
                #Even if one of them is scale we can't fold it into  bn
                #without impacting the other layer
                return (bn_layer_seq, None)
            else:
                #Only one layer depends on bn blob and
                #if that is scale that can be folded into bn
                scale_layer_cand = None
                for input_layer in layers_depending_on_bn_blob:
                    typ = str(input_layer.type).upper()
                    if typ == 'SCALE':
                        scale_layer_cand = input_layer

                if scale_layer_cand is not None:
                    scale_layer_seq = scale_layer_cand
                    if len(scale_layer_seq.bottom) > 1:
                        raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
                    #Only one layer and is scale
                    #can be folded
                    self.fold_layer(scale_layer_seq, bn_layer_seq)
                    return (bn_layer_seq, scale_layer_seq)
                else:
                    #There is no scale layer candidate
                    #No folding necessary
                    return (bn_layer_seq, None)


    def check_cbs_folding(self, layer):
        typ = str(layer.type).upper()
        # Temporary work-around to use this existing code for FC - Batchnorm folding as well
        assert((typ == 'CONVOLUTION' or typ == 'CONVOLUTIONRISTRETTO' or typ == 'INNERPRODUCT' or typ == 'FCRISTRETTO'))
        #Get output blob of convolution layer
        output_blob = layer.top[0]

        layer_seq = layer
        bn_layer_seq = None
        scale_layer_seq = None

        #First step is to check if this blob is written into by convolution + batchnorm + scale
        #Get the output of layers name and check within it
        layers_writing_into_blob = self._blobs[output_blob]['output_of_layers']

        index_layer = layers_writing_into_blob.index(layer_seq)
        n1_entry = n2_entry = None
        n1_entry_typ = n2_entry_typ = 'NONE'
        if len(layers_writing_into_blob) > (index_layer + 1):
            n1_entry = layers_writing_into_blob[index_layer + 1]
            n1_entry_typ = str(n1_entry.type).upper()
        if len(layers_writing_into_blob) > (index_layer + 2):
            n2_entry = layers_writing_into_blob[index_layer + 2]
            n2_entry_typ = str(n2_entry.type).upper()

        if n1_entry_typ == 'BATCHNORM' and Utils.check_use_global_stats(n1_entry):
            bn_layer_seq = n1_entry
        if n2_entry_typ == 'SCALE':
            scale_layer_seq = n2_entry

        if bn_layer_seq is not None and scale_layer_seq is not None:
            if len(scale_layer_seq.bottom) > 1:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
            #FC/Conv + bn + scale consecutive seq found and write into same output buffer
            #The batchnorm and scale layers can be folded into the prev FC/conv layer
            self.fold_layer(bn_layer_seq, layer_seq)
            self.fold_layer(scale_layer_seq, layer_seq)
            return (layer_seq, bn_layer_seq, scale_layer_seq)
        elif bn_layer_seq is not None:
            #FC/Conv + bn consecutive seq found and write into the same buffer
            #Check if there is a scale layer that depends on the FC/conv blob
            #and if the scale layer is the first and only other dependent layer after the
            #bn blob that depends on it
            #Get layers that depend on this blob
            layers_depending_on_blob = self._blobs[output_blob]['input_to_layers']
            index_bn_layer = layers_depending_on_blob.index(bn_layer_seq)
            n1_entry = None
            n1_entry_typ = ''
            # index_bn_layer + 2 as index starts from 0
            if len(layers_depending_on_blob) == (index_bn_layer + 2):
                n1_entry = layers_depending_on_blob[index_bn_layer + 1]
                n1_entry_typ = str(n1_entry.type).upper()

            if n1_entry_typ == 'SCALE':
                #The only dependent blob on FC1/conv1 other than bn is scale
                scale_layer_seq = n1_entry
                if len(scale_layer_seq.bottom) > 1:
                    raise ValueError(
                        code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
                self.fold_layer(bn_layer_seq, layer_seq)
                self.fold_layer(scale_layer_seq, layer_seq)
                return (layer_seq, bn_layer_seq, scale_layer_seq)
            else:
                #No scale layer dependent on conv after bn
                #The batchnorm layer can be folded into the prev fc/conv layer
                self.fold_layer(bn_layer_seq, layer_seq)
            return (layer_seq, bn_layer_seq, None)
        else:
            #Check for sequence #2
            #Get layers that depend on this blob
            layers_depending_on_blob = self._blobs[output_blob]['input_to_layers']
            #Early Out: Check if there is more than two layers depend on fc/conv output
            #in which case there is no way that batchnorm and scale could be folded
            #as both depend on fc/conv output as its input
            if len(layers_depending_on_blob) > 1:
                #More than two layers depend on the conv output for their input
                #Cannot fold anything into this layer as a result
                return (layer_seq, None, None)

            bn_layer_cand = None
            for input_layer in layers_depending_on_blob:
                typ = str(input_layer.type).upper()
                if typ == 'BATCHNORM' and Utils.check_use_global_stats(input_layer):
                    bn_layer_cand = input_layer

            if bn_layer_cand is not None:
                #len(layers_depending_on_blob) == 1:
                #Only one layer depends on fc/conv_output_blob and that is bn_layer_cand
                bn_output_blob = bn_layer_cand.top[0]

                layers_writing_into_bn_blob = self._blobs[bn_output_blob]['output_of_layers']
                layers_depending_on_bn_blob = self._blobs[bn_output_blob]['input_to_layers']
                index_bn_layer = layers_writing_into_bn_blob.index(bn_layer_cand)
                #if another layer writes into the same blob as batchnorm
                if len(layers_writing_into_bn_blob) > (index_bn_layer + 1):
                    n1_entry = layers_writing_into_bn_blob[index_bn_layer + 1]
                    n1_entry_typ = str(n1_entry.type).upper()
                    #if that layer is scale we can fold fc/conv, bn and scale
                    if n1_entry_typ == 'SCALE':
                        bn_layer_seq = bn_layer_cand
                        scale_layer_seq = n1_entry
                        if len(scale_layer_seq.bottom) > 1:
                            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
                        self.fold_layer(bn_layer_seq, layer_seq)
                        self.fold_layer(scale_layer_seq, layer_seq)
                        return (layer_seq, bn_layer_seq, scale_layer_seq)
                    else:
                        #Some other layer apart from scale is writing into batchnorm
                        #We can't fold batchnorm in this case
                        #TODO:This could be folded.. needs work
                        return (layer_seq, None, None)
                else:
                    #If no other layer writes into the batchnorm blob
                    #Early out: If there are more than one layers that depend on the batchnorm blob
                    #we can't fold the batchnorm layer
                    if len(layers_depending_on_bn_blob) > 1:
                        #Cannot fold in any scale layer that might exist after
                        #since at this point we know that the scale layer
                        #does not write into same bn blob and there is
                        #another layer apart from scale that depends on bn
                        bn_layer_seq = bn_layer_cand
                        self._optimized_out_layers.append(bn_layer_seq)
                        self._optimized_blob_replacements[bn_output_blob] = output_blob
                        return (layer_seq, bn_layer_seq, None)
                    elif len(layers_depending_on_bn_blob) == 1:
                        #Only one other layer depends on bn
                        #If this layer turns out to be scale
                        #we can fold fc/conv+bn+scale
                        n1_entry = layers_depending_on_bn_blob[0]
                        n1_entry_typ = str(n1_entry.type).upper()
                        if n1_entry_typ == 'SCALE':
                            bn_layer_seq = bn_layer_cand
                            scale_layer_seq = n1_entry
                            if len(scale_layer_seq.bottom) > 1:
                                raise ValueError(
                                    code_to_message.get_error_message('ERROR_CAFFE_SCALE_FOLDING_TOO_MANY_INPUTS')(scale_layer_seq.name, str(scale_layer_seq.bottom)))
                            self.fold_layer(bn_layer_seq, layer_seq)
                            self.fold_layer(scale_layer_seq, layer_seq)
                            return (layer_seq, bn_layer_seq, scale_layer_seq)
                        else:
                            #only fold the bn into fc/conv
                            bn_layer_seq = bn_layer_cand
                            self.fold_layer(bn_layer_seq, layer_seq)
                            return (layer_seq, bn_layer_seq, None)
                    else:
                        #len(layers_depending_on_bn_blob) == 0
                        #fc/Conv layer is depended on by only bn
                        #Bn layer is not depended on by anybody
                        #only fold the bn into conv
                        bn_layer_seq = bn_layer_cand
                        self.fold_layer(bn_layer_seq, layer_seq)
                        return (layer_seq, bn_layer_seq, None)
            else:
                return (layer_seq, None, None)

    def squash_concats(self, spec):
        for layer in spec.layer:
            if layer not in self._optimized_out_layers and str(layer.type).upper() == 'CONCAT':
                self.check_concat_folding(layer)

    def check_concat_folding(self, layer):
        # Only layers to be processed should end up calling this function. Concat layers that
        # are part of a pre-calculated squash sequence are added to the layer 'skip' list in
        # the main loop and will be skipped over. Only the 3 nodes should enter this function:
        # 1. Standalone concat layers
        # 2. The 'first' node in a concat squash sequences (meaning the first discovered in the protobuf)
        # 3. The last node in a squash sequence

        # If the current layer is listed as the last node in a concat squash return as it has already been processed
        if layer.name in self._concat_squash:
            return

        # Find the last concat layer in a sequence of foldable concat layers. A concat
        # layer is foldable if it has the same axes and doesn't output to more than 1 layer
        last_concat_layer = self.find_last_foldable_concat(layer)

        concat_input_seq = self.build_ordered_concat_input_list([], last_concat_layer)
        if last_concat_layer.name == layer.name:
            return

        # Add the squash enpoint layer to the squash list. Other layers in the sequence will be skipped
        # but this layer should not be skipped, but also should not be checked for folding again
        self._concat_squash.append(last_concat_layer.name)

        # Rewrite the inputs using the 'squashed inputs'.
        cur_inputs = []
        cur_inputs.extend(last_concat_layer.bottom)
        for in_name in cur_inputs:
            last_concat_layer.bottom.remove(in_name)
        last_concat_layer.bottom.extend(concat_input_seq)

        return

    def find_last_foldable_concat(self, layer):
        # Get the next layer(s) in the sequence
        layers_depending_on_concat_blob = self._blobs[layer.top[0]]['input_to_layers']

        # TODO? Should we try to handle concat to multiple concats that come back to a single concat?
        # If more than 1 layer depends on this output (or none do) or it's not a concat layer, bail
        if len(layers_depending_on_concat_blob) != 1:
            return layer

        next_layer = layers_depending_on_concat_blob[0]
        if str(next_layer.type).upper() != 'CONCAT':
            return layer

        if next_layer.concat_param.axis != layer.concat_param.axis:
            return layer

        return self.find_last_foldable_concat(next_layer)

    # Builds an ordered input list for squashable concats, and squashes all included concat layers
    # 'layer' starts as the last layer in the list and works backwards to get the real inputs
    def build_ordered_concat_input_list(self, concat_input_seq, layer):
        if str(layer.type).upper() != 'CONCAT':
            return concat_input_seq

        # For each input get the layer that produced it. If it's not a concat append the input
        # to the input list, otherwise traverse to the concat layer and continue input processing.
        for input_name in layer.bottom:
            layer_producing_blob = self._blobs[input_name]['output_of_layers']
            # The layer is squashable if it was produced by only 1 layer, is a concat,
            # has only 1 dependency, and shares the same concat axis
            if len(layer_producing_blob) == 1:
                prev_layer = layer_producing_blob[0]
                if (str(prev_layer.type).upper() == 'CONCAT' and
                    len(self._blobs[input_name]['input_to_layers']) == 1 and
                    prev_layer.concat_param.axis == layer.concat_param.axis):

                    # Squashable, fold the previous layer into the current layer
                    self.fold_layer(prev_layer, layer)
                    concat_input_seq = self.build_ordered_concat_input_list(concat_input_seq, prev_layer)
                    continue

            concat_input_seq.append(input_name)

        return concat_input_seq

class NetworkTopology(object):
    def __init__(self, modeltools):
        self._model = modeltools
        # convinient way to register internally all layers
        # will be easier to query for example during get_output_buffers
        # rather than going into C++ land in modeltools
        self._layers = {}
        # Input/Output buffer Proxies
        self._proxy = BufferProxy()
        self._logger = logging.getLogger()

    def add_implicit_scale_layer(self, l, model):
        self._proxy.add_implicit_scale_layer(l, model)
        # update self._layers
        self.__add_layer(l)

    def install_buffer_proxy(self, original_buffer, proxy_buffer):
        self._proxy.install_buffer_proxy(original_buffer, proxy_buffer)

    def add_layer(self, l, layer_type):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_ADD_LAYER')(l.name, layer_type))
        # handle input/output proxy
        self._proxy.add_layer(l, self._model, layer_type)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_DONE_ADDING'))
        self._proxy.dump()
        # update self._layers
        self.__add_layer(l)

    # Given a layername, get a list of the output buffers
    # and consult the proxy buffers - using get_output_buffers for each
    # buffer in the layer
    # This is used in implicit crop/preprocessing logic
    # Also take advantage that we keep all layers info in self._layers
    def get_output_buffers(self, layername):
        self._logger.debug("NetworkTopology get_output_buffers " +layername)
        if layername not in self._layers:
            raise RuntimeError("NetworkTopology get_output_buffers " + layername + " does not exist")
        layer = self._layers[layername]
        outputs = []
        for t in layer.top:
            tstr = str(t)
            outputs.append(self.get_output_buffer_name(tstr))
        return outputs

    def get_input_buffer_name(self, bufname):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME')(bufname))
        ret = self.__get_buffer_name(bufname, self._proxy.input_proxy())
        # the bufname is in the first element of the tuple
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME_RET')(ret))
        return ret

    def get_output_buffer_name(self, bufname):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_OUTPUT_BUFFER_NAME')(bufname))
        ret = self.__get_buffer_name(bufname, self._proxy.output_proxy())
        # the bufname is in the first element of the tuple
        return ret

    # update self._layers
    def __add_layer(self, l):
        if l.name in self._layers:
             raise RuntimeError("NetworkTopology add_layer " + l.name + " already exists")
        self._layers[l.name] = l;

    def __get_buffer_name(self, bufname, proxy_array):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_BUFFER_NAME')(bufname))
        ret_bufname = str(bufname)
        if bufname in proxy_array:
            ret_bufname = proxy_array[bufname]
        # no proxy (thus no alias either), just return the same name
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_BUFFER_NAME_RET')(ret_bufname))
        return (str(ret_bufname))

#------------------------------------------------------------------------------
#   Weight Providers
#------------------------------------------------------------------------------
class BlobWeightProvider(object):
    def __init__(self, weights_map):
        self.weights_map = weights_map

    def get_bn_weights(self, layer):
        # SegNet BatchNorm:
        #
        # blob 0 -> weights
        # blob 1 -> bias
        # network must be set to INFERENCE mode by the BN global statistics script
        # before running this script
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])
        return c_weights, c_bias

    def get_batch_norm_weights(self, layer, input_depth_prev):
        # Mainline BatchNorm:
        #
        # blob 0 -> unscaled mean
        # blob 1 -> unscaled variance
        # blob 2 -> scale_factor (1-element array)
        #
        # weights = 1 / sqrt(variance+epsilon)
        # bias = (-1 * mean) / sqrt(variance+epsilon)
        # input_depth_prev is not used. It's only in place to stay in sync
        # with RandomWeightProvider
        scale_factor = snpeUtils.blob2arr(self.weights_map[layer.name][2])[0]
        mean = snpeUtils.blob2arr(self.weights_map[layer.name][0]) / scale_factor
        variance = snpeUtils.blob2arr(self.weights_map[layer.name][1]) / scale_factor
        eps = layer.batch_norm_param.eps
        stddev = numpy.sqrt(variance+eps)
        c_weights = 1 / stddev
        c_bias = (-1 * mean) / stddev
        return c_weights, c_bias

    def get_normalize_weights(self, layer):
        # SSD Normalize
        #
        # blob 0 -> scale factors
        #
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        return c_weights

    def get_conv_weights(self, layer, bias_term):
        # weights are stored as [N,C,H,W], need [H,W,C,N]
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        c_weights = numpy.rollaxis(c_weights, 0, 4) # [C,H,W,N]
        c_weights = numpy.rollaxis(c_weights, 0, 3)
        c_weights = numpy.ascontiguousarray(c_weights, dtype=numpy.float32)

        if bias_term:
            c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])
        else:
            c_bias = numpy.require([0] * layer.convolution_param.num_output, dtype=numpy.float32)
        return c_weights, c_bias

    def get_deconv_weights(self, layer, bias_term):
        # deconv weights are ordered as [C,N,H,W]
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        c_weights = numpy.rollaxis(c_weights, 0, 4)
        c_weights = numpy.rollaxis(c_weights, 0, 4)
        c_weights = numpy.ascontiguousarray(c_weights, dtype=numpy.float32)
        if bias_term:
            c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])
        else:
            c_bias = numpy.require([0] * layer.convolution_param.num_output, dtype=numpy.float32)
        return c_weights, c_bias

    def get_fc_weights(self, layer, input_depths, bias_term):
        if bias_term:
            c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][-1])
            weights_blob = self.weights_map[layer.name][:-1]
        else:
            c_bias = numpy.require([0] * layer.inner_product_param.num_output, dtype=numpy.float32)
            weights_blob = self.weights_map[layer.name]

        weights_list = []
        for input_depth, blob in zip(input_depths, weights_blob):
            c_weights = snpeUtils.blob2arr(blob)
            output_size, input_size = c_weights.shape
            if input_depth == input_size:
                weights_list.append(c_weights)
            else: # need to re-order because activations go C,H,W -> H,W,C
                c_weights = numpy.reshape(c_weights, (output_size, input_depth, input_size//input_depth))
                c_weights = numpy.rollaxis(c_weights, 1, 3)
                c_weights = numpy.reshape(c_weights, (output_size,input_size))
                weights_list.append(c_weights)
        return weights_list, c_bias

    def get_lstm_weights(self, layer):
        # Caffe stores weights NxK, we want KxN
        c_x_weights = numpy.ascontiguousarray(snpeUtils.blob2arr(self.weights_map[layer.name][0]).transpose(), dtype=numpy.float32)
        c_bias =      snpeUtils.blob2arr(self.weights_map[layer.name][1])
        c_h_weights = numpy.ascontiguousarray(snpeUtils.blob2arr(self.weights_map[layer.name][2]).transpose(), dtype=numpy.float32)
        return c_x_weights, c_bias, c_h_weights

    def get_prelu_weights(self, layer):
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        return [ float(f) for f in c_weights ]

    def get_scale_weights(self, layer, bias_term, input_depth):
        # input_depth ignored when bias provided
        # Weights are not present or passed when scale layer has 2 inputs
        c_weights = None
        c_bias = numpy.require([0] * input_depth, dtype=numpy.float32)
        if len(layer.bottom) == 2:
            if bias_term:
                c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        else:
            c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
            if bias_term:
                c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])

        return c_weights, c_bias

class RandomWeightProvider(object):
    def __init__(self, model, network_topology, blob_connectivity_map):
        self.model = model
        self._network_topology = network_topology
        self._blob_connectivity_map = blob_connectivity_map

    def make_weights(self, shape, macs_per_output = 1):
        """
        Create random weights with a given shape, which will
        keep the power of a laye's activations the same as that of
        it's input.

        parameters:
        shape           -- tuple of int, the shape the weights should have.
        macs_per_output -- The number of macs which will contribute to a single
                           activation. Used for normalization.
        """
        weights = numpy.random.random(shape)*2-1
        weights /= macs_per_output
        return numpy.require(weights, dtype=numpy.float32)

    def get_bn_weights(self, layer):
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_layer_name = replaced_bottom[0]
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[3]
        c_weights = self.make_weights( (input_depth,) )
        return c_weights, c_weights # use weights for bias

    def get_batch_norm_weights(self, layer, input_depth_prev):
        # For RandomWeightProvider previous layer might not be regaisterd
        # by the time the batchnorm layer weights are requested since
        # batchnorm weights might be needed in the conv layer itself
        # if it is being folded in. In this case the prev input depth
        # is provided as a parameter
        if input_depth_prev is None:
            replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
            input_layer_name = replaced_bottom[0]
            bufname = self._network_topology.get_output_buffer_name(input_layer_name)
            input_depth = self.model.get_buffer_dims(bufname)[-1]
        else:
            input_depth = input_depth_prev
        c_weights = self.make_weights( (input_depth,) )
        return c_weights, c_weights # use weights for bias

    def get_normalize_weights(self, layer):
        if (layer.norm_param.channel_shared):
            input_depth = 1
        else:
            input_layer_name = layer.bottom[0]
            bufname = self._network_topology.get_output_buffer_name(input_layer_name)
            input_depth = self.model.get_buffer_dims(bufname)[3]
        c_weights = self.make_weights( (input_depth,) )
        return c_weights

    def get_conv_weights(self, layer, bias_term):
        convParam = layer.convolution_param

        kx = 0
        ky = 0
        if convParam.kernel_h and convParam.kernel_w:
            kx = convParam.kernel_w
            ky = convParam.kernel_h
        if isinstance(convParam.kernel_size, int):
            kx = convParam.kernel_size
            ky = convParam.kernel_size
        else:
            if len(convParam.kernel_size) > 0:
                kx = convParam.kernel_size[0]
                ky = convParam.kernel_size[0]
            if len(convParam.kernel_size) > 1:
                kx = convParam.kernel_size[1]
        if kx == 0  or ky == 0:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_FIELDS')(str(layer.name)))

        output_depth = convParam.num_output
        groups = convParam.group
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_layer_name = replaced_bottom[0]
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[3]
        weights_shape = (ky, kx, input_depth/groups, output_depth)
        macs_per_output = ky*ky*input_depth/groups
        c_weights = self.make_weights(weights_shape, macs_per_output)
        c_bias = self.make_weights( (output_depth,))
        return c_weights, c_bias

    def get_deconv_weights(self, layer, bias_term):
        convParam = layer.convolution_param

        kx = 0
        ky = 0
        if convParam.kernel_h and convParam.kernel_w:
            kx = convParam.kernel_w
            ky = convParam.kernel_h
        if isinstance(convParam.kernel_size, int):
            kx = convParam.kernel_size
            ky = convParam.kernel_size
        else:
            if len(convParam.kernel_size) > 0:
                kx = convParam.kernel_size[0]
                ky = convParam.kernel_size[0]
            if len(convParam.kernel_size) > 1:
                kx = convParam.kernel_size[1]
        if kx == 0  or ky == 0:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_FIELDS')(str(layer.name)))

        output_depth = convParam.num_output
        groups = convParam.group
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_layer_name = replaced_bottom[0]
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[3]

        weights_shape = (ky, kx, input_depth, output_depth/groups)
        macs_per_output = ky*kx*input_depth/groups
        c_weights = self.make_weights(weights_shape, macs_per_output)
        c_bias = self.make_weights( (output_depth,))
        return c_weights, c_bias

    def get_fc_weights(self, layer, input_depths, bias_term):
        # input_depths unused
        fc_parm = layer.inner_product_param
        output_depth = fc_parm.num_output
        c_weights_list = []
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        for name in replaced_bottom:
            bufname = self._network_topology.get_output_buffer_name(name)
            # Strip batch dimension for input size calculation
            input_dims = self.model.get_buffer_dims(bufname)[1:]
            input_size = reduce(lambda a,b: a*b, input_dims)
            w = self.make_weights( (output_depth, input_size), input_size)
            c_weights_list.append(w)
        if bias_term:
            c_bias = self.make_weights( (output_depth,) )
        else:
            c_bias = numpy.zeros((output_depth,), dtype=numpy.float32)
        return c_weights_list, c_bias

    def get_lstm_weights(self, layer):
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_layer_name = replaced_bottom[0]
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[-1]
        output_depth = layer.recurrent_param.num_output
        c_x_weights = self.make_weights( (input_depth, output_depth * 4), input_depth )
        c_bias = self.make_weights( (output_depth * 4,) )
        c_h_weights = self.make_weights( (output_depth, output_depth * 4), input_depth )
        return c_x_weights, c_bias, c_h_weights

    def get_prelu_weights(self, layer):
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_layer_name = replaced_bottom[0]
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[3]
        return [ (2*random.random())-1 for i in range(input_depth) ]

    def get_scale_weights(self, layer, bias_term, input_depth):
        # Bias should always be created. Weights are only created when scale takes a
        # single input
        c_bias = self.make_weights( (input_depth,) )

        # Only create weights when there isn't a second input (which ARE the weights)
        c_weights = None
        if len(layer.bottom) == 1:
            c_weights = c_bias
        return c_weights, c_bias


#------------------------------------------------------------------------------
#   Converter Class
#------------------------------------------------------------------------------
class CaffeSnapDnnConverter(object):
    def __init__(self):
        self.model = modeltools.Model()

        self.layer_id_map = {}
        self.pool_parms_map = {}
        self.logger = logging.getLogger()
        self.preprocessing_is_setup = False
        # define empty dict
        self._udl_factory_func = {}

        # Instansiate network topology to be used by the converter
        self._network_topology = NetworkTopology(self.model)

        # Instantiate blob connectivity map needed for optimizations
        self._blob_connectivity_map = BlobConnectivityMap()

        # This variable indicates the presence of fixed point layers
        self.fixed_point_layers_present = False

        # Instantiate two axis trackers, one each for Caffe and SNPE
        self._caffe_axis_tracker = snpe_axis_transformer.AxisTracker("Caffe")
        self._snpe_axis_tracker = snpe_axis_transformer.AxisTracker("SNPE")

        # Instantiate one axis transformer with two axis trackers and two layered order axes
        self._axis_transformer = snpe_axis_transformer.AxisTransformer(caffe_layer_axes, self._caffe_axis_tracker,
                                                                                snpe_layer_axes, self._snpe_axis_tracker)
        # Maintains a list of layers not requiring to go through axis transformer
        self._axis_skip_layers = ['DATA', 'DUMMYDATA', 'INPUT', 'DROPOUT', 'SSDPERMUTE', 'PERMUTE']

        # Maintains a list of layers that are not transformed to any layer of dlc and thus skipping
        # its processing
        self._skip_layers = ['SILENCE', 'ACCURACY', 'SOFTMAXWITHLOSS', 'ARGMAX', 'SSDPRIORBOX', 'PRIORBOX']

        # Global SSD priorbox parameters
        self.total_prior_box_output = {}

        # Dictionary mapping for add_layer function
        self.add_layer_dict = {
            'INPUT'                     : self.add_input_layer,
            'BN'                        : self.add_bn_layer,
            'CONCAT'                    : self.add_concat_layer,
            'TILE'                      : self.add_tile_layer,
            'CROP'                      : self.add_crop_layer,
            'CUDNNCROSSCORRELATION'     : self.add_cross_correlation_layer,
            'SHUFFLECHANNEL'            : self.add_channel_shuffle_layer,
            'DATA'                      : self.add_data_layer,
            'DUMMYDATA'                 : self.add_data_layer,
            'DECONVOLUTION'             : self.add_deconvolution_layer,
            'DECONVOLUTIONRISTRETTO'    : self.add_deconvolution_layer,
            'DROPOUT'                   : self.add_dropout_layer,
            'ELTWISE'                   : self.add_elementwise_layer,
            'INNERPRODUCT'              : self.add_fc_layer,
            'FCRISTRETTO'               : self.add_fc_layer,
            'LRN'                       : self.add_rnorm_layer,
            'LRNRISTRETTO'              : self.add_rnorm_layer,
            'LSTM'                      : self.add_lstm_layer,
            'NORMALIZE'                 : self.add_normalize_layer,
            'POOLING'                   : self.add_pooling_layer,
            'SSDPERMUTE'                : self.add_permute_layer,
            'PERMUTE'                   : self.add_permute_layer,
            'PRELU'                     : self.add_prelu_layer,
            'PYTHON'                    : self.add_python_layer,
            'POWER'                     : self.add_power_layer,
            'RESHAPE'                   : self.add_reshape_layer,
            'FLATTEN'                   : self.add_reshape_layer,
            'RELU'                      : self.add_relu_layer,
            'ELU'                       : self.add_elu_layer,
            'ROIPOOLING'                : self.add_roipooling_layer,
            'SIGMOID'                   : self.add_logistic_layer,
            'SOFTMAX'                   : self.add_softmax_layer,
            'TANH'                      : self.add_tanh_layer,
            'UPSAMPLE'                  : self.add_upsample_layer,
            'SLICE'                     : self.add_slice_layer,
            'BATCHNORM'                 : self.add_batch_norm_layer,
            'CONVOLUTION'               : self.add_conv_layer,
            'CONVOLUTIONRISTRETTO'      : self.add_conv_layer,
            'SCALE'                     : self.add_scale_layer,
            'SSDOUTPUT'                 : self.add_ssd_detection_output_layer,
            'DETECTIONOUTPUT'           : self.add_ssd_detection_output_layer,
            'SILENCE'                   : self.add_silence_layer
        }

        # Dictionary mapping for add_layer function for convert_caffe_old
        self.add_layer_old_dict = {
            caffe.proto.caffe_pb2.V1LayerParameter.CONCAT           : self.add_concat_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.CONVOLUTION      : self.add_conv_layer_old,
            caffe.proto.caffe_pb2.V1LayerParameter.DECONVOLUTION    : self.add_deconvolution_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.DROPOUT          : self.add_dropout_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.INNER_PRODUCT    : self.add_fc_layer_old,
            caffe.proto.caffe_pb2.V1LayerParameter.LRN              : self.add_rnorm_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.POOLING          : self.add_pooling_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.RELU             : self.add_relu_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.SIGMOID          : self.add_logistic_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.SOFTMAX          : self.add_softmax_layer,
            caffe.proto.caffe_pb2.V1LayerParameter.TANH             : self.add_tanh_layer
        }

    def set_udls(self, obj):
        if not type (obj) is dict:
            self.logger.error (code_to_message.get_error_message('ERROR_CAFFE_UDL_SET_IS_NOT_DICT'))
            return False
        # Extract every udl object
        for layer_type, layer_obj in list(obj.items()):
            up = layer_type.upper()

            # Register udl function
            self._udl_factory_func[up] = layer_obj.getLayerCallback()

            # Register all of its target axis orders
            input_axis_orders, output_axis_orders =  layer_obj.getAxisOrder()
            for i_axis_order, o_axis_order in zip(input_axis_orders, output_axis_orders):
                snpe_layer_axes.add_axis_order(up, i_axis_order, o_axis_order)

            # Register all of its source axis orders
            src_input_axis_orders, src_output_axis_orders =  layer_obj.getSrcAxisOrder()
            for i_axis_order, o_axis_order in zip(src_input_axis_orders, src_output_axis_orders):
                caffe_layer_axes.add_axis_order(up, i_axis_order, o_axis_order)

        ppstr = pprint.pformat(self._udl_factory_func, indent=4)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_PRINT_UDL_FACTORY_FUNCS')(ppstr))

        return True

    def convert(self,
                caffe_prototext_path,
                caffe_model_path,
                dlc_output_path,
                copyright_file,
                encoding,
                input_size,
                input_layers,
                input_types,
                enable_preprocessing=False,
                model_version=None,
                disable_batchnorm_folding=False,
                converter_command='N/A',
                validation_target = [],
                enable_strict_validation=False):

        self.copyright_str = snpe_converter_utils.get_string_from_txtfile(copyright_file)
        self.enable_preprocessing = enable_preprocessing
        self.encoding = encoding
        self.input_size = input_size
        self.input_dim = list()
        self.network_dim = []
        self.disable_batchnorm_folding = disable_batchnorm_folding
        self.validation_target = validation_target
        self.enable_strict_validation = enable_strict_validation
        self.caffe_prototext_path = caffe_prototext_path

        # add validation target
        if len(self.validation_target) == 0:
            self.logger.debug("no validation target specified.")
            self.model.add_validation_targets(self.model.get_validation_targets())
        else:
            self.logger.debug("validation target :" + str(tuple(self.validation_target)))
            self.model.add_validation_targets(tuple(self.validation_target))

        # set valiation mode
        if self.enable_strict_validation == True:
            self.logger.debug("strict validation is enabled.")
            self.model.set_strict_validation(True)

        self.input_types_map = {}
        if input_layers is not None and input_types is not None:
            if len(input_layers) != len(input_types):
                print(code_to_message.get_error_message('ERROR_CAFFE_INPUT_TYPES_LAYER_NAMES_NOT_IN_PAIRS')(input_types, input_layers))
                sys.exit(1)
            self.input_types_map = {input_layers[i]: input_types[i] for i in range(len(input_layers))}

        caffe.set_mode_cpu()

        # get caffe spec
        try:
            self.spec = caffe_pb2.NetParameter()
            with open(caffe_prototext_path, 'rb') as text_file:
                text_format.Merge(text_file.read(), self.spec)
        except Exception as e:
            print(code_to_message.get_error_message('ERROR_CAFFE_CAFFE_PARSING_ERROR')(caffe_prototext_path, str(e)))
            print(code_to_message.get_progress_message('INFO_CAFFE_CAFFE_INSTALLATION_ERROR')(caffe.__file__))
            sys.exit(1)

        # get weight provider
        if caffe_model_path is None:
            self.weight_provider = RandomWeightProvider(self.model,
                                                        self._network_topology,
                                                        self._blob_connectivity_map)
        else:
            caffenet = caffe.Net(caffe_prototext_path,
                                 caffe_model_path,
                                 caffe.TEST)

            self.weight_provider = BlobWeightProvider(caffenet.params)

        # Rename top/bottom blobs to have unique names before doing any work
        self.preprocess_net(self.spec, self._network_topology)

        # Loop through layers
        if len(self.spec.layers) == 0:
            self.convert_caffe_new(self.spec)
        else:
            self.convert_caffe_old(self.spec)

        if model_version is not None:
            self.model.set_model_version(model_version[:64])

        self.model.set_converter_command(converter_command)
        self.model.set_model_copyright(self.copyright_str)
        self.model.save(dlc_output_path)

    def preprocess_net(self, spec, topology):
        self.preprocess_net_names(spec, topology)

        # Generate layer connectivity map first
        for layer in spec.layer:
            if self._is_in_train_phase(layer):
                #Skip train layers
                continue
            self._blob_connectivity_map.add_layer(layer)

        # Squash concats ahead of layer processing
        self._blob_connectivity_map.squash_concats(spec)

    def preprocess_net_names(self, spec, topology):
        # Loop through all ops and fix op names and inputs and outputs such that if more than one op ouput
        # has the same name subsequent outputs and the following inputs with the same name get renamed
        output_map = {}
        for layer in spec.layer:
            if self._is_in_train_phase(layer):
                #Skip train layers
                continue

            # Process inputs first... they need to use the current output mapping
            for index,i in enumerate(layer.bottom):
               # If a mapping for an outuput exists and it's different remap the input to the new name
               if i in output_map and i != output_map[i]:
                   self.logger.debug(
                       code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_REMAP_INPUT')(layer.name, i, output_map[i]))
                   layer.bottom[index] = output_map[i]

            # Process outputs next, they may remap later inputs
            for index,o in enumerate(layer.top):
               if o not in output_map:
                   output_map[o] = o
               else:
                   o_alias = str(layer.name) + "." + str(o) # works because Caffe enforces unique layer names
                   self.logger.debug(
                       code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_REMAP_OUTPUT')(layer.name, o, o_alias))
                   output_map[o] = o_alias
                   layer.top[index] = o_alias

    def convert_caffe_new( self, spec ):
        # Need to add a data layer
        if len(spec.input_shape) > 0 and len(spec.input_shape[0].dim) == 4:
            # 4-D input_shape used. Treat as (batch, depth, height, width)
            self.input_dim = list(map( int, [spec.input_shape[0].dim[0],
                                        spec.input_shape[0].dim[2],
                                        spec.input_shape[0].dim[3],
                                        spec.input_shape[0].dim[1]] ))
        elif len(spec.input_shape) > 0:
            # input_shape, but not 4-D, just copy native dims and add 1 as batch
            self.input_dim = list(map(int, spec.input_shape[0].dim))
            self.input_dim.insert(0, 1)
        elif len(spec.input_dim) == 4:
            # 4-D input_dim. Treat as (batch, depth, height, width)
            self.input_dim = list(map( int, [spec.input_dim[0],
                                        spec.input_dim[2],
                                        spec.input_dim[3],
                                        spec.input_dim[1]] ))
        elif len(spec.input_dim) > 0:
            # input_dim, but not 4-D, just copy native dims and add 1 as batch
            self.logger.debug("length of input dim " + str(len(spec.input_dim)))
            self.input_dim = list(map(int, spec.input_dim).insert(0,1))

        if self.input_dim and len(spec.input):
            self.setup_preprocessing(str(spec.input[0]), None)

        # If there are additional inputs. create data layers for these. Note that
        # only the input {} input_shape {} syntax is supported here.
        for index in range(1, len(spec.input)):
            data_name = str(spec.input[index])
            if len(spec.input_shape[index].dim) == 4:
                # 4-D input_dim. Treat as (batch, depth, height, width)
                data_dims = list(map( int, [spec.input_shape[index].dim[0],
                                       spec.input_shape[index].dim[2],
                                       spec.input_shape[index].dim[3],
                                       spec.input_shape[index].dim[1]] ))
            elif len(spec.input_shape[index].dim) == 2:
                # 2-D input_dim. Treat as (batch, depth)
                data_dims = list(map( int, [1, 1, spec.input_shape[index].dim[0], spec.input_shape[index].dim[1]] ))
            else:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_UNSUPPORTED_INPUT_DIMS')(str(data_name)))
            input_type = self.input_types_map.get(data_name, "default")
            id_ = self.model.add_data_layer(data_name, data_dims, "bgr", "bgr", input_type)
            self._axis_transformer.update_src_axis_order('DATA', len(self.input_dim), data_name, len(self.input_dim))
            self._axis_transformer.update_target_axis_order('DATA', len(self.input_dim), data_name, len(self.input_dim))
            self.save_axis_order(data_name)
            self.layer_id_map[data_name] = id_ # layer name is blob name

        prev_batch_norm_layer = None

        for layer in spec.layer:

            typ = str(layer.type).upper()
            self.logger.debug("layer type " + typ)

            # SSD specific "hacks" mbox_priorbox is a regular concat
            if ('Concat' in str(layer.type) and
                len(self._blob_connectivity_map._blobs[layer.bottom[0]]['output_of_layers']) > 0 and
                'PriorBox' in self._blob_connectivity_map._blobs[layer.bottom[0]]['output_of_layers'][0].type):
                self.process_ssd_priorbox_concat_layer(layer)
                continue
            if 'DetectionOutput' in str(layer.type) or 'SsdOutput' in str(layer.type):
                # Remove the priorbox input name since we are passing converter generated data to the detection layer
                # Cache the prior box input name for looking up the corresponding data for this specific ssd detect instance
                self.priorbox_input_name = layer.bottom[2]
                del layer.bottom[2]
            if 'PriorBox' in str(layer.type):
                self.process_ssd_priorbox_layer(layer)

            if self._is_in_train_phase(layer) or self._is_in_skip_layers(layer):
                # Skip train and unsupported layers
                continue
            if layer in self._blob_connectivity_map.get_optimized_out_layers():
                #Skip optimized out layers
                continue
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_PRINT_LAYER_TYPE')(typ))

            # Pre layer work - if implicit permute layer is required, it adds it
            self.do_pre_layer_work(layer)
            self._network_topology.add_layer(layer, typ)

            # We would like to allow UDL of (also) known layers.
            # So in order to achieve this, we will consult the UDL map
            # here, before checking for the type and if this works,
            # we will run do_post_layer_work and continue the loop
            # This check for udl is_known_udl_type used to be at the very
            # end, but by moving it up, we will first check for the map
            # and it will use it even if this layer is known to us
            # So check if we have been supplied with UDL factory function
            if self.is_known_udl_type(typ):
                self.add_udl_layer(layer)
                # Post layer work - Axis tracking related
                # need to call if for any layer addon.
                # it is also inovked at the very end of this func
                self.do_post_layer_work(layer)
                continue

            try:
                self.add_layer_dict[typ](layer)
            except:
                print(code_to_message.get_error_message('ERROR_CAFFE_LAYER_TYPE_NOT_SUPPORTED')(str(layer.name), typ))
                sys.exit(1)

            # Post layer work - Axis tracking related
            self.do_post_layer_work(layer)

    def _is_in_train_phase(self, layer):
        if layer.include:
            caffe_phases = {pair[0]: pair[1] for pair in list(caffe_pb2.Phase.items())}
            phases = [state.phase for state in layer.include if state.phase is not None]
            return caffe_phases['TRAIN'] in phases
        return False

    def _is_in_skip_layers(self, layer):
        layer_type = str(layer.type).upper()
        if layer_type in self._skip_layers:
            self.logger.debug("Omitting layer " + layer.name + " of type " + layer_type)
            return True
        else:
            return False

    def convert_caffe_old( self, spec ):
        # Need to add a data Layer
        sx = spec.input_dim[3]
        sy = spec.input_dim[2]
        sz = spec.input_dim[1]
        sb = spec.input_dim[0]

        data_name = str(spec.input[0])
        input_type = self.input_types_map.get(data_name, "default")
        id_ = self.model.add_data_layer( data_name, [sb,sx,sy,sz], "bgr", "bgr", input_type)
        self.layer_id_map[data_name] = id_ # layer name is blob name

        for layer in spec.layers:
            typ = layer.type
            self._network_topology.add_layer(layer, str(layer.type))
            try:
                self.add_layer_old_dict[typ](layer)
            except:
                print(code_to_message.get_error_message('LAYER_TYPE_NOT_SUPPORTED')(layer.name, layer.type))
                print(code_to_message.get_progress_message('INFO_CAFFE_LAYER_TYPE_DEF_ERROR'))
                sys.exit(1)

    # Check if the input needs to be permuted
    def check_input_axes(self, layer_type, input_dims, original_input_name, new_input_name):
        permute = False
        no_permute_order = numpy.arange(len(input_dims)).tolist()
        if not self._axis_transformer.has_target_axis_order(new_input_name):
            return True, []

        permute_order = self._axis_transformer.get_permute_order( str(layer_type).upper(), len(input_dims),
                                                                          original_input_name, new_input_name )
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_NO_PERMUTE_ORDER')(str(no_permute_order), str(permute_order)))

        # If the current input permute ordering doesn't match we need to permute
        if len(permute_order) and permute_order != list(range(len(permute_order))):
            permute = True

        return permute, permute_order

    # Figure out which input name should be used for a layer by determining if permuting is required or not
    def pick_input_names(self, input_names, layer_type):
        ret_names = []
        for name in input_names:
            new_name = self._network_topology.get_input_buffer_name(name)

            # There are two inputs... potentially the same or an input could have been proxiedto a new name
            # If the inputs are the same do nothing. Otherwise check the proxy input first and if it does
            # not require a permute add it. Otherwise check the original input name. If that doesn't require
            # a permute take that name. This tackles proxy cases and skip connection cases where one branch
            # requires a permute but the others don't.
            if name == new_name:
                ret_names.append(name)
                continue

            input_dims = self.model.get_buffer_dims(str(new_name))
            permute, _ = self.check_input_axes(layer_type, input_dims, name, new_name)
            if permute:
                if name != new_name:
                    input_dims = self.model.get_buffer_dims(str(name))
                    permute, _ = self.check_input_axes(layer_type, input_dims, name, name)
                    if not permute:
                        ret_names.append(name)
                        continue
            ret_names.append(new_name)

        return ret_names

    # Function to conduct preparatory work before layer is added
    # It includes adding implicit permute layer.
    def do_pre_layer_work(self, layer):

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_PREWORK_OF_LAYER')(layer.name))

        if str(layer.type).upper() in self._axis_skip_layers:
            return

        # Get both caffe and snpe buffer names and its dims.
        target_input_names = self.get_input_names(layer)

        #FIXME: Due to bug where 2nd outpout (index-based upsampling pool mask) of pooling layer
        #is not registerd with buffer info, its output dim is not availabe and get_output_dims complains about it.
        #Thus, for pooling layer, only one output is considered.
        input_dims = []
        src_input_names = []
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        if str(layer.type).upper() == 'UPSAMPLE' and len(replaced_bottom) > 1:
            src_input_names.append(replaced_bottom[0])
            input_dims.append(self.get_input_dim(layer))
        else:
            replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
            src_input_names = list(replaced_bottom)
            input_dims = self.get_input_dims(layer)
        assert len(src_input_names)
        assert( len(src_input_names) == len(input_dims) )
        assert( len(src_input_names) == len(target_input_names) )

        # For each of its bottom buffer, get its permute order
        for idx in range(len(src_input_names)):
            # No permute order = [0,1,2] for 3d input or [0,1] for 2d
            # First check if the src/target input names are the same. For skip connections where a permute may be required in
            # one path, but not the other, first try the original name, then only if it fails try the permute/mapped name
            if src_input_names[idx] != target_input_names[idx]:
                permute, permute_order = self.check_input_axes(layer.type, input_dims[idx], str(src_input_names[idx]), str(src_input_names[idx]))
                if not permute:
                    # No permute required using original input. Continue analyzing pther inputs
                    continue

            # Test the proxy name
            permute, permute_order = self.check_input_axes(layer.type, input_dims[idx], src_input_names[idx], target_input_names[idx])

            # Note: input and output name are the same as to mimic in-place buffer
            if permute:
                self.logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE_IMPLICIT_PERMUTE_LAYER')(str(permute_order), layer.name))
                self.add_implicit_permute_layer( layer.name, permute_order, src_input_names[idx] )

    # Function to conduct finishing work after layer is added
    def do_post_layer_work(self, layer):

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_POSTWORK_OF_LAYER')(layer.name))
        if str(layer.type).upper() in self._axis_skip_layers:
            return

        target_output_names = self.get_output_names(layer)


        #FIXME: Due to bug where 2nd outpout (index-based upsampling pool mask) of pooling layer
        #is not registerd with buffer info, its output dim is not availabe and get_output_dims complains about it.
        #Thus, for pooling layer, only one output is considered.
        output_dims = []
        src_output_names = []
        if str(layer.type).upper() == 'POOLING' and len(layer.top) > 1:
            src_output_names.append(layer.top[0])
            output_dims.append(self.get_output_dim(layer))
        else:
            src_output_names = list(layer.top)
            output_dims = self.get_output_dims(layer)

        # With the strict assumption that in MIMO or MISO usecases, the axis order of all inputs
        # are the same, the input_rank of only one input is passed.
        input_rank = len(self.get_input_dim(layer))
        target_singleout_input_name = self.get_input_name(layer)
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        src_singleout_input_name = replaced_bottom[0]

        # For each of its top buffer, update its Caffe and SNPE axis order
        for idx in range(len(src_output_names)):
            # TBD: +1 might change for reshape/lstm where batch dimension is not disregard.
            # TBD: Figure out Caffe py api that returns per-layer input shape . That way we could partially
            #      get away with +1
            self._axis_transformer.update_src_axis_order(str(layer.type).upper(), len(output_dims[idx]), src_output_names[idx],
                                                         input_rank, src_singleout_input_name)
            self._axis_transformer.update_target_axis_order(str(layer.type).upper(), len(output_dims[idx]), target_output_names[idx],
                                                            input_rank, target_singleout_input_name)
            self.save_axis_order(target_output_names[idx])

    def add_layer(self, layer, id_):
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_ADDING_LAYER')(layer.name))
        for name in layer.top:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_ADDING_LAYER_TOP_NAME_IS_TREATED_DIFFERENTLY')(name))
            self.layer_id_map[name] = id_

    def is_known_udl_type(self, name):
        # empty means it was not provided
        ppstr = pprint.pformat(self._udl_factory_func, indent=4)
        if not type (self._udl_factory_func) is dict:
            self.logger.error(code_to_message.get_error_message('ERROR_CAFFE_UDL_FACTORY_FUNCS_NOT_SUPPLIED')(ppstr))
            return False
        # have a UDL list, so see if name layer exists in UDL list
        if name not in self._udl_factory_func:
            # note that this function can be called for native layers as well as UDL layers to override native
            # layers,so it's not necessarily an error if the name doesn't exist in the UDL map
            return False
        return True

    def add_udl_layer(self, layer):
        typ = str(layer.type)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_UDL')(typ, str(layer.name)))
        name = str(layer.name)
        # guaranteed to be available but in UPPERCASE:
        # since this script has UPPERCASE and lowercase mixed
        # at least in UDL we treat all as UPPERCASE
        func = self._udl_factory_func[typ.upper()]
        # FIXME should be list of lists
        inputDims = self.get_input_dims(layer)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_UDL_INPUT_DIMS')(str(inputDims)))
        blob_output = func(layer, inputDims)
        blob = blob_output.getBlob()
        # we need a list of lists.
        # i.e. a list of dimensions. each dimensions is a list
        output_dims = []
        for idx in range(len(layer.top)):
            # FIXME do we need list() here?
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS_IDX')(str(idx)))
            dim = blob_output.getOutputDims(idx)
            assert(isinstance(dim, list))
            output_dims.append(dim)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_UDL_OUTPUT_DIMS')(str(output_dims)))
        if blob.getSize() == 0:
            self.logger.error (code_to_message.get_error_message('ERROR_CAFFE_UDL_BLOB_SIZE_IS_ZERO')(name))
            sys.exit(1)
        # need list(output_dims) since it is tuple, and the function expect list of int
        inputsList = self.get_input_names(layer)
        outputList = self.get_output_names(layer)
        # we cache typ again since  blob_output = func(layer, inputDims)
        # might change the layer type (we allow this)
        # make sure typ is to upper
        typ = str(layer.type)
        id_ = self.model.add_user_defined_layer(name,
                                                typ.upper(),
                                                inputsList,
                                                outputList,
                                                output_dims,
                                                blob.getBlob())
        self.add_layer(layer, id_);

    def add_bn_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), str(layer.name)))
        name = str(layer.name)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_BATCH_NORMALIZATION_LAYER')(name))
        weights, bias = self.weight_provider.get_bn_weights(layer)
        id_ = self.model.add_batchnorm_layer(name,
                                             weights,
                                             bias,
                                             compute_statistics = False,
                                             use_mu_sigma = False,   # unused
                                             across_spatial = False, # unused
                                             input_name = str(self.get_input_name(layer)),
                                             output_name = str(self.get_output_name(layer)))
        self.add_layer(layer, id_)

    def add_batch_norm_layer(self, layer):
        layer_seq = None
        if not self.disable_batchnorm_folding:
            layer_seq = self._blob_connectivity_map.check_bs_folding(layer)
            bn_layer_seq = layer_seq[0]
            assert (bn_layer_seq is not None)
            scale_layer_seq = layer_seq[1]
            self.add_batch_norm_layer_util(bn_layer_seq, scale_layer_seq)
        else:
            self.add_batch_norm_layer_util(layer, None)

    def add_batch_norm_layer_util(self, layer_batch_norm, layer_scale):
        name = str(layer_batch_norm.name)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_BATCH_NORMALIZATION_LAYER')(name))
        # from the batch_norm layer we get weights W1 and bias B1:
        # y  = W1.x + B1
        # from the scaling layer (if present), we get weights W2 and bias B2:
        # y' = W2.y + B2 = W2(W1.x + B1) + B2 =
        #                = (W2.W1)x + (W2.B1 + B2)
        weights, bias = self.weight_provider.get_batch_norm_weights(layer_batch_norm, None)

        # If use_global_stats is False (Caffe training mode) treat this as instance normalization
        compute_statistics = False
        if not Utils.check_use_global_stats(layer_batch_norm):
            # Reset weights and biases to 1s and 0s
            weights.fill(1)
            bias.fill(0)
            compute_statistics = True

        if layer_scale is not None:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_MERGING_SCALE_LAYER')(str(layer_scale.name)))
            bias_term = getattr(layer_scale, "bias_term", True)
            scale_weights, scale_bias = self.weight_provider.get_scale_weights(layer_scale, bias_term, len(bias))
            weights = weights * scale_weights
            bias = bias * scale_weights + scale_bias
            #Change name to reflect scale folded into batchnorm
            name = name + '.' + str(layer_scale.name)

        id_ = self.model.add_batchnorm_layer(name, weights, bias,
                                             compute_statistics = compute_statistics,
                                             use_mu_sigma = True,   # unused
                                             across_spatial = True, # unused
                                             input_name = str(self.get_input_name(layer_batch_norm)),
                                             output_name = str(self.get_output_name(layer_batch_norm)))
        self.add_layer(layer_batch_norm, id_)
        #if layer_scale is not None:
        #    self.add_layer(layer_scale, id_)

    def add_normalize_layer(self, layer):
        name = str(layer.name)
        self.logger.debug("Converting normalization layer " + name)
        # from the normalize layer we get weights W:
        # if channel_shared is true, there is only a single weight which we will
        # replicate across the input channels.
        weights = self.weight_provider.get_normalize_weights(layer).flatten(order='C')
        if layer.norm_param.channel_shared:
           input_layer_name = layer.bottom[0]
           bufname = self._network_topology.get_output_buffer_name(input_layer_name)
           input_depth = self.model.get_buffer_dims(bufname)[2]
           weights = weights[0] * numpy.ones([input_depth], dtype=numpy.float32)
        # this layer does not support bias values. construct an array of zeros.
        bias = numpy.zeros(shape=[len(weights)],dtype=numpy.float32)
        id_ = self.model.add_batchnorm_layer(name, weights, bias,
                                             compute_statistics = True,
                                             use_mu_sigma = False,  # compute RMS
                                             across_spatial = layer.norm_param.across_spatial,
                                             input_name = str(self.get_input_name(layer)),
                                             output_name = str(self.get_output_name(layer)))
        self.add_layer(layer, id_)

    def add_concat_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), str(layer.name)))

        # When more than one concat is present the axis verification occurs in check_concat_folding
        caffe_axis = layer.concat_param.axis
        if caffe_axis == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CONCAT_BATCH_DIM_ERR')(str(layer.name)))

        target_inames = self.get_input_names(layer)
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        src_inames = list(replaced_bottom)
        assert(len(target_inames) == len(src_inames))

        # For each of input buffer, retrieve snpe axis given a caffe axis
        snpe_axes = []
        for idx in range(len(target_inames)):
            snpe_axes.append(self._axis_transformer.get_target_axis(src_inames[idx], caffe_axis, target_inames[idx]))

        # Sanity Check: for all buffers, the same axis is returned.
        if not all( x == snpe_axes[0] for x in snpe_axes):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CONCAT_AXIS_NOT_ALIGNED')(str(layer.name)))

        id_ = self.model.add_concatenation_layer( name=str(layer.name),
                                                  input_names=target_inames,
                                                  output_name = str(self.get_output_name(layer)),
                                                  axis = snpe_axes[0])

        # Adding only the last layer in a concat squash sequence
        self.add_layer(layer, id_)

    def add_tile_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), str(layer.name)))

        caffe_axis = layer.tile_param.axis
        caffe_tiles = layer.tile_param.tiles

        if caffe_axis > 4:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_TILE_AXIS_NOT_SUPPORTED')(str(layer.name), caffe_axis))

        target_inames = self.get_input_names(layer)
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        src_inames = list(replaced_bottom)
        assert(len(target_inames) == len(src_inames))

        snpe_axis = self._axis_transformer.get_target_axis(src_inames[0], caffe_axis, target_inames[0])
        tile_inputs = target_inames * caffe_tiles

        id_ = self.model.add_concatenation_layer( name = str(layer.name),
                                                  input_names = tile_inputs,
                                                  output_name = str(self.get_output_name(layer)),
                                                  axis = snpe_axis)

        self.add_layer(layer, id_)

    def add_conv_layer(self, layer):
        if not self.disable_batchnorm_folding:
            layer_seq = self._blob_connectivity_map.check_cbs_folding(layer)
            conv_layer_seq = layer_seq[0]
            bn_layer_seq = layer_seq[1]
            scale_layer_seq = layer_seq[2]
            if bn_layer_seq is not None and scale_layer_seq is not None:
                self.add_conv_layer_util(conv_layer_seq, bn_layer_seq, scale_layer_seq)
            elif bn_layer_seq is not None:
                self.add_conv_layer_util(conv_layer_seq, bn_layer_seq, None)
            else:
                self.add_conv_layer_util(conv_layer_seq, None, None)
        else:
            self.add_conv_layer_util(layer, None, None)

    def add_conv_layer_old(self, layer):
        self.add_conv_layer_util(layer, None, None)

    def add_conv_layer_util(self, layer, layer_batch_norm, layer_scale):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        convParam = layer.convolution_param
        groups = 1
        if hasattr(convParam, "group"):
            groups = convParam.group

        # Fetch values from layer.convParam
        conv_params = self.get_conv_params(convParam)

        dilation_x, dilation_y = 1, 1
        if len(convParam.dilation) == 1:
            dilation_x = convParam.dilation[0]
            dilation_y = convParam.dilation[0]
        elif len(convParam.dilation) > 1:
            dilation_x = convParam.dilation[1]
            dilation_y = convParam.dilation[0]

        input_layer_name = self.get_input_name(layer)
        output_depth = convParam.num_output
        input_size = self.model.get_buffer_dims(str(input_layer_name))
        input_height = input_size[0]
        input_width = input_size[1]
        input_depth = input_size[2]

        # def calc_modules(pad, size, k, stride, dilation):
        #     kernel_extent = dilation*(k-1) + 1
        #     return 1 + ((2*pad + size - kernel_extent) // stride)

        # output_width = calc_modules(conv_params.padx,
        #                             input_width,
        #                             conv_params.kx,
        #                             conv_params.stridex,
        #                             dilation_x)
        # output_height = calc_modules(conv_params.pady,
        #                              input_height,
        #                              conv_params.ky,
        #                              conv_params.stridey,
        #                              dilation_y)

        bias_term = getattr(convParam, "bias_term", True)
        c_weights, c_bias = self.weight_provider.get_conv_weights(layer, bias_term)

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_WEIGHT_DIMS')(c_weights.shape))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_INPUT_DIMS')(input_size))
        # self.logger.debug(
        #     code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS')(tuple([output_height, output_width, output_depth])))

        #FIXME: Handle else case where scale exists but not batchnorm.
        #This should be caught in the calling function but need to have a check here for defense
        if layer_batch_norm is not None and layer_scale is not None:
            # from the conv layer we get weights W0 and bias B0:
            # y0 = W0.x + B0
            # from the batch_norm layer we get weights W1 and bias B1:
            # y  = W1.y0 + B1 = W1(W0.x + B0) + B1 =
            # y = (W1.W0)x + (W1.B0 + B1)
            # from the scaling layer (if present), we get weights W2 and bias B2:
            # y' = W2.y + B2 = W2(W1.W0.x + W1.B0 + B1) + B2 =
            #                = W2.W1.W0.x + (W2.W1.B0 + W2.B1 + B2)
            bn_weights, bn_bias = self.weight_provider.get_batch_norm_weights(layer_batch_norm, len(c_bias))
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_MERGING_SCALE_LAYER')(str(layer_scale.name)))
            bias_term = getattr(layer_scale, "bias_term", True)
            scale_weights, scale_bias = self.weight_provider.get_scale_weights(layer_scale, bias_term, len(bn_bias))

            c_weights = c_weights * bn_weights * scale_weights
            c_bias = c_bias * scale_weights * bn_weights + bn_bias * scale_weights + scale_bias
        elif layer_batch_norm is not None:
            # from the conv layer we get weights W0 and bias B0:
            # y0 = W0.x + B0
            # from the batch_norm layer we get weights W1 and bias B1:
            # y  = W1.y0 + B1 = W1(W0.x + B0) + B1 =
            # y = (W1.W0)x + (W1.B0 + B1)
            bn_weights, bn_bias = self.weight_provider.get_batch_norm_weights(layer_batch_norm, len(c_bias))

            c_weights = c_weights * bn_weights
            c_bias = c_bias * bn_weights + bn_bias

        #Convolution layer could have folded batchnorm and scale layers after it
        #Change name to reflect that
        name = str(layer.name)
        if layer_batch_norm is not None and layer_scale is not None:
            name = name + '.' + str(layer_batch_norm.name) + '.' + str(layer_scale.name)
        elif layer_batch_norm is not None:
            name = name + '.' + str(layer_batch_norm.name)

        id_ = self.model.add_conv_layer( name = name,
                                         weights = c_weights,
                                         bias = c_bias,
                                         padx = conv_params.padx,
                                         pady = conv_params.pady,
                                         padding_mode = modeltools.PADDING_ZERO,
                                         padding_size_strategy = modeltools.PADDING_SIZE_EXPLICIT,
                                         stridex = conv_params.stridex,
                                         stridey = conv_params.stridey,
                                         dilationx = dilation_x,
                                         dilationy = dilation_y,
                                         input_name = str(self.get_input_name(layer)),
                                         output_name = str(self.get_output_name(layer)),
                                         groups = groups)

        output_dim = self.model.get_buffer_dims(str(layer.top[0]))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS')(tuple(output_dim)))

        # Add quantization information
        self.add_fxp_layer_encoding(layer)

        self.add_layer(layer, id_)

    def add_crop_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        # FIXME why using get_input_names() here and not get_input_name()?
        input_name, shape_name = self.get_input_names(layer)
        output_name = str(self.get_output_name(layer))
        input_dims = list(self.model.get_buffer_dims(input_name))
        target_dims = list(self.model.get_buffer_dims(shape_name))

        caffe_offset = [int(o) for o in layer.crop_param.offset]
        if len(caffe_offset) == 0:
            caffe_offset = [0]*4
        elif len(caffe_offset) == 1:
            caffe_offset = [caffe_offset[0]]*4

        if len(target_dims) == 4:
            axis = layer.crop_param.axis % 4
            if axis == 0:
                offsets = [caffe_offset[0],caffe_offset[2],caffe_offset[3],caffe_offset[1]]
            elif axis  == 1:
                offsets = [0,caffe_offset[1],caffe_offset[2],caffe_offset[0]]
            elif axis == 2:
                offsets = [0,caffe_offset[0],caffe_offset[1],0]
                target_dims[0] = input_dims[0]
                target_dims[3] = input_dims[3]
            else:
                offsets = [0,0,caffe_offset[0],0]
                target_dims[0] = input_dims[0]
                target_dims[1] = input_dims[1]
                target_dims[3] = input_dims[3]
        elif len(target_dims) == 3:
            axis = layer.crop_param.axis % 4
            if axis == 0:
                offsets = [caffe_offset[2],caffe_offset[3],caffe_offset[1]]
            elif axis  == 1:
                offsets = [caffe_offset[1],caffe_offset[2],caffe_offset[0]]
            elif axis == 2:
                offsets = [caffe_offset[0],caffe_offset[1],0]
                target_dims[2] = input_dims[2]
            else:
                offsets = [0,caffe_offset[0],0]
                target_dims[0] = input_dims[0]
                target_dims[2] = input_dims[2]
        elif len(target_dims) == 1:
            offsets = [caffe_offset[0]]
        else:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CROP_LAYER_OUTPUT_DIM_ERR')(str(layer.name)))

        id_ = self.model.add_crop_layer(str(layer.name), offsets,
                                        target_dims, input_name, output_name)
        self.add_layer(layer, id_)


    def add_cross_correlation_layer(self, layer):
        self.logger.debug("Converting " + str(layer.type) + " layer " + layer.name)
        input_name, filter_name = None, None
        try:
            input_name, filter_name = self.get_input_names(layer)
        except ValueError:
            raise ValueError("Layer %s: expected exactly two input blobs" % layer.name)

        output_name = str(self.get_output_name(layer))
        id_ = self.model.add_cross_correlation_layer(str(layer.name),
                                                     input_name,
                                                     filter_name,
                                                     output_name)
        self.add_layer(layer, id_)

    def add_channel_shuffle_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_CHANNEL_SHUFFLE_LAYER')(str(layer.type), layer.name))

        snpe_groups = 1
        if hasattr(layer, "shuffle_channel_param"):
            shuffle_channel_param = layer.shuffle_channel_param
            if hasattr(shuffle_channel_param, "group"):
                snpe_groups = shuffle_channel_param.group
            else:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_CHANNEL_SHUFFLE_LAYER_MISSING_GROUPS_ARG')(str(layer.type)))

        return self.model.add_channel_shuffle_layer(name=layer.name,
                                                    groups=snpe_groups,
                                                    shuffle_type=modeltools.CHANNEL_SHUFFLE_GROUPED,
                                                    input_name=str(self.get_input_name(layer)),
                                                    output_name=str(self.get_output_name(layer)))

    def add_data_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        if len(layer.include) > 0 and layer.include[0].phase != caffe_pb2.TEST:
            self.logger.warn(code_to_message.get_warning_message('WARNING_CAFFE_OMIT_DATA_LAYER_INCL_TRAIN'))

        if len(self.input_dim) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM')(str(layer.name)))
        self.setup_preprocessing(str(layer.name), getattr(layer, "transform_param", None))

    def add_input_layer(self, layer):
        # As of Feb 2016, the latest baseline caffe version supports a newer mechanism
        # of specifying input data. It uses a separate input type layer.
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_INPUT_LAYER')(layer.name))

        if not hasattr(layer, "input_param"):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_INPUT_PARAM_SPECIFIED')(str(layer.name)))

        input_param = layer.input_param
        self.input_dim = list(map( int, [input_param.shape[0].dim[0],
                                    input_param.shape[0].dim[2],
                                    input_param.shape[0].dim[3],
                                    input_param.shape[0].dim[1]] ))

        self.setup_preprocessing(str(layer.name), getattr(layer, "transform_param", None))

    def add_deconvolution_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        convParam = layer.convolution_param

        # Fetch values from layer.convParam
        padx, pady, stridex, stridey, kx, ky = self.get_conv_params(convParam)
        output_depth = convParam.num_output
        groups = convParam.group if hasattr(convParam, "group") else 1

        bias_term = getattr(convParam, "bias_term", True)
        c_weights, c_bias = self.weight_provider.get_deconv_weights(layer, bias_term)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_WEIGHT_DIMS')(c_weights.shape))
        id_ = self.model.add_deconvolution_layer( name=str(layer.name),
                                                  weights = c_weights,
                                                  bias = c_bias,
                                                  stride = stridex,
                                                  padding_size_strategy = modeltools.PADDING_SIZE_EXPLICIT,
                                                  padding = padx,
                                                  input_name = str(self.get_input_name(layer)),
                                                  output_name = str(self.get_output_name(layer)),
                                                  output_width=-1,
                                                  output_height=-1,
                                                  groups=groups)

        self.add_layer(layer, id_)

    def add_dropout_layer(self, layer):
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OMITTING_DROPOUT_LAYER')(layer.name))
        input_id = self.get_input_id(layer)
        self.layer_id_map[self.get_output_name(layer)] = input_id

        if len(layer.top) != 1:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_DROPOUT_LAYER_WITH_MUL_OUTPUTS_ERR')(str(layer.name)))

        # In case of not in-place  buffer, axis order needs to be notified about src
        # axis order.
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        if len(layer.top) == len(replaced_bottom) and layer.top[0] != replaced_bottom[0]:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_DROPOUT_LAYER_WITHOUT_INPUT_BUFFER'))
            src_input_axis_order = self._axis_transformer.get_src_axis_order(replaced_bottom[0])
            self._axis_transformer.update_src_axis_order(str(layer.type).upper(), len(src_input_axis_order), layer.top[0],
                                                          len(src_input_axis_order), replaced_bottom[0], src_input_axis_order)

    def add_elementwise_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        elementwise_param = layer.eltwise_param
        op = elementwise_param.operation
        input_names = self.get_input_names(layer)
        output_name = str(self.get_output_name(layer))
        if op == elementwise_param.PROD:
            id_ = self.model.add_elementwise_product_layer( str(layer.name),
                                                            input_names,
                                                            output_name)
        elif op == elementwise_param.SUM:
            coeffs = list(elementwise_param.coeff)
            if len(coeffs) < len(input_names):
                self.logger.warn(code_to_message.get_warning_message('WARNING_CAFFE_FEWER_COEFFS_THAN_INPUT_NUM'))
                coeffs.extend( [1.0 for i in range(len(input_names)-len(coeffs))] )
            elif len(coeffs) > len(input_names):
                self.logger.warn(code_to_message.get_warning_message('WARNING_CAFFE_MORE_COEFFS_THAN_INPUT_NUM'))

            id_ = self.model.add_elementwise_sum_layer( str(layer.name),
                                                        coeffs,
                                                        input_names,
                                                        output_name)
        elif op == elementwise_param.MAX:
            id_ = self.model.add_elementwise_max_layer( str(layer.name),
                                                        input_names,
                                                        output_name)
        else:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_UNRECOGNIZED_ELEMENTWISE_OP')(str(layer.name), str(op)))

        self.add_layer(layer, id_)

    def add_fc_layer(self, layer):
        if not self.disable_batchnorm_folding:
            layer_seq = self._blob_connectivity_map.check_cbs_folding(layer)
            fc_layer_seq = layer_seq[0]
            bn_layer_seq = layer_seq[1]
            scale_layer_seq = layer_seq[2]
            if bn_layer_seq is not None and scale_layer_seq is not None:
                self.add_fc_layer_util(fc_layer_seq, bn_layer_seq, scale_layer_seq)
            elif bn_layer_seq is not None:
                self.add_fc_layer_util(fc_layer_seq, bn_layer_seq, None)
            else:
                self.add_fc_layer_util(fc_layer_seq, None, None)
        else:
            self.add_fc_layer_util(layer, None, None)

    def add_fc_layer_old(self, layer):
         self.add_fc_layer_util(layer, None, None)

    def add_fc_layer_util(self, layer, layer_batch_norm, layer_scale):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        c_input_names = self.get_input_names(layer)
        self.logger.debug(str(c_input_names))

        # Compute parameters for fc layer
        input_depths = [self.model.get_buffer_dims(name)[-1] for name in c_input_names]
        fcParam = layer.inner_product_param
        bias_term = getattr(fcParam, "bias_term", True)
        c_weights_list, c_bias = self.weight_provider.get_fc_weights(layer, input_depths, bias_term)
        name = str(layer.name)

        # Condition to check if batch norm or batch norm+scale is available.
        # Then the appropriate math to fold batch_norm and scale layer into FC
        # Math:From the fc layer we get weights W0 and bias B0:
        # y0 = W0.x + B0
        # from the batch_norm layer we get weights W1 and bias B1:
        # y  = W1.y0 + B1 = W1(W0.x + B0) + B1 =
        # y = ((W1^T * W0)^T )* x + (W1.B0 + B1) ("^T" means array transpose)
        # from the scaling layer (if present), we get weights W2 and bias B2:
        # y' = W2.y + B2 = ((W1^T * W0)^T )* x + (W1.B0 + B1) + B2

        if layer_batch_norm is not None:
            bn_weights, bn_bias = self.weight_provider.get_batch_norm_weights(layer_batch_norm, len(c_bias))
            c_weights_list = [(weights.T * bn_weights).T for weights in c_weights_list]
            c_bias = c_bias * bn_weights + bn_bias
            name = str(layer.name) + '.' + str(layer_batch_norm.name)

            if layer_scale is not None:
                self.logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE_MERGING_SCALE_LAYER')(str(layer_scale.name)))
                bias_term = getattr(layer_scale, "bias_term", True)
                scale_weights, scale_bias = self.weight_provider.get_scale_weights(layer_scale, bias_term, len(bn_bias))
                c_weights_list = [(weights.T * scale_weights).T for weights in c_weights_list]
                c_bias = scale_weights * c_bias + scale_bias
                name = name + '.' + str(layer_scale.name)

        for weights in c_weights_list:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_WEIGHTS_SHAPE')(str(weights.shape)))

        id_ = self.model.add_fc_layer(name=name,
                                      weights_list=c_weights_list,
                                      bias=c_bias,
                                      input_names=c_input_names,
                                      output_name=str(self.get_output_name(layer)))

        # Add quantization information
        self.add_fxp_layer_encoding(layer)

        self.add_layer(layer, id_)

    def add_logistic_layer(self, layer):
        id_ = self.model.add_neuron_layer( name = str(layer.name),
                                           func = modeltools.NEURON_LOGISTIC,
                                           input_name = str(self.get_input_name(layer)),
                                           output_name = str(self.get_output_name(layer)),
                                           a = 1.0)
        self.add_layer(layer, id_)

    def add_lstm_layer(self, layer):
        name = str(layer.name)
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(layer.type, name))
        x_weights, bias, h_weights = self.weight_provider.get_lstm_weights(layer)
        # For LSTM, output_names are generated in the Model Tools C++ space
        input_names = self.get_input_names(layer)
        output_names = [name]
        sequence_continuation_name = ''
        if len(input_names) > 1:
            sequence_continuation_name = input_names[1]

        x_static_name = ''
        if len(input_names) == 3 or len(input_names) == 5:
            x_static_name = input_names[2]

        c_0_input_name = ''
        h_0_input_name = ''
        if len(input_names) > 3:
            c_0_input_name = input_names[-2]
            h_0_input_name = input_names[-1]
            output_names.append('{}_c_T'.format(name))
            output_names.append('{}_h_T'.format(name))

        id_ = self.model.add_lstm_layer(name,
                                        x_weights,
                                        bias,
                                        h_weights,
                                        0,     # unclear how Caffe encodes x_static weights
                                        False, # unclear how Caffe encodes "backward"
                                        reset_state_at_time_step_0=False,
                                        input_name=input_names[0],
                                        sequence_continuation_input_name=sequence_continuation_name,
                                        x_static_input_name=x_static_name,
                                        c_0_input_name=c_0_input_name,
                                        h_0_input_name=h_0_input_name,
                                        output_names=output_names)

        self.add_layer(layer, id_)

    def add_silence_layer(self, layer):
        print(code_to_message.get_progress_message('INFO_CAFFE_OMIT_SILENCE_LAYER')(layer.name))

    def add_pooling_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        pool_param = layer.pooling_param

        c_pool_type = modeltools.POOL_MAX
        if pool_param.pool:
            c_pool_type = modeltools.POOL_AVG


        sizex = pool_param.kernel_size
        sizey = sizex
        if pool_param.kernel_h or pool_param.kernel_w:
            sizex = pool_param.kernel_w
            sizey = pool_param.kernel_h

        stridex = pool_param.stride
        stridey = stridex
        if pool_param.stride_h or pool_param.stride_w:
            stridex = pool_param.stride_w
            stridey = pool_param.stride_h

        padx = pool_param.pad
        pady = padx
        if pool_param.pad_h or pool_param.pad_w:
            padx = pool_param.pad_w
            pady = pool_param.pad_h

        include_padding = True
        input_dim = self.get_input_dim(layer)
        if pool_param.global_pooling:
            sizey = input_dim[1]
            sizex = input_dim[2]
            stridex, stridey = 1, 1
            padx, pady = 0, 0
            include_padding = False


        output_height = 1 + int(math.ceil(float(input_dim[0] + 2*pady - sizey)/stridey))
        # don't start a pool beyond the bottom of the image
        if (output_height-1)*stridey - pady >= input_dim[0]:
            output_height -= 1
        output_width = 1 + int(math.ceil(float(input_dim[1] + 2*padx - sizex)/stridex))
        # don't start a pool beyond the right border of the image
        if (output_width-1)*stridex - padx >= input_dim[1]:
            output_width -= 1

        output_dim = [output_height, output_width, input_dim[2]]
        id_ = self.model.add_pooling_layer( str(layer.name),
                                            c_pool_type,
                                            sizex,
                                            sizey,
                                            stridex,
                                            stridey,
                                            padx,
                                            pady,
                                            modeltools.PADDING_SIZE_EXPLICIT,
                                            str(self.get_input_name(layer)),
                                            str(self.get_output_name(layer)),
                                            include_padding)
        output_dim = self.model.get_buffer_dims(str(layer.top[0]))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS')(tuple(output_dim)))

        # if there is a second top, this will be upsampled later.
        if len(layer.top) > 1:
            if sizex != sizey or stridex != stridey or padx != pady:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_INDEX_BASED_UPSAMPLING_DOES_NOT_SUPPORT_RECT_POOL')(str(layer.name)))

            input_name = str(self.get_input_name(layer))
            input_dim = self.model.get_buffer_dims(input_name)
            pool_parms = {}
            pool_parms['size'] = sizex
            pool_parms['stride'] = stridex
            pool_parms['pad'] = padx
            pool_parms['input_dim'] = input_dim
            pool_parms['id'] = id_

            mask_name = str(layer.top[1])
            self.pool_parms_map[mask_name] = pool_parms

        self.add_layer(layer, id_)

    def add_permute_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        if hasattr(layer, 'ssd_permute_param'):
           permute_param = layer.ssd_permute_param
        else:
           permute_param = layer.permute_param
        if not len(permute_param.order):
             raise ValueError(
                 code_to_message.get_error_message('ERROR_CAFFE_PERMUTE_LAYER_MISSING_ORDER_FIELD')(str(layer.name)))

        src_permute_order = permute_param.order
        target_buffer_name = self.get_input_name(layer)
        input_dim = self.model.get_buffer_dims(target_buffer_name)
        no_permute_order = numpy.arange(len(input_dim)).tolist()

        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)

        target_permute_order = self._axis_transformer.get_permute_order(str(layer.type).upper(), len(input_dim),
                                                                        replaced_bottom[0], target_buffer_name, src_permute_order)

        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_SNPE_PERMUTE_ORDER')(str(src_permute_order), str(target_permute_order)))

        if len(target_permute_order) and target_permute_order != no_permute_order:
            id_ = self.model.add_permute_layer( name = str(layer.name),
                                                order = target_permute_order,
                                                input_name = target_buffer_name,
                                                output_name = str(self.get_output_name(layer)) )
            self.add_layer(layer, id_)

            # Update axis order of target buffer
            target_input_axis_order = self._axis_transformer.get_target_axis_order(target_buffer_name)
            target_output_axis_order = []
            for idx in target_permute_order:
                target_output_axis_order.append(target_input_axis_order[idx])

            self._axis_transformer.update_target_axis_order(str(layer.type).upper(), len(target_permute_order), self.get_output_name(layer),
                                                            len(target_permute_order), target_buffer_name, target_output_axis_order)
            self.save_axis_order(self.get_output_name(layer))
        else:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_NO_PERMUTE_REQUIRED_FOR_SNPE'))

            # When permute layer is skipped, use bottom alias for top
            self._network_topology.install_buffer_proxy(layer.top[0], self.get_input_name(layer))

        # Update axis order of src buffer
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        src_input_axis_order = self._axis_transformer.get_src_axis_order(replaced_bottom[0])
        src_output_axis_order = []
        for idx in src_permute_order:
            src_output_axis_order.append(src_input_axis_order[idx])

        self._axis_transformer.update_src_axis_order(str(layer.type).upper(), len(src_permute_order), layer.top[0],
                                                     len(src_permute_order), replaced_bottom[0], src_output_axis_order)

    def add_implicit_permute_layer(self, layer_name, permute_order, layer_input_name):

        # Generate unique implicit layer name by combining the layer name, "permute" and input buffer name
        # Just layer name and "permute" is not sufficient for multiple inputs.
        implicit_permute_layer_name = layer_input_name + "_permute_" + str(get_permute_id())
        implicit_permute_output_name = layer_input_name + ".permute." + layer_name

        # The top and bottom name of the actual layer for which the implicit permute is done, is chosen as input
        # and output for the implicit permute layer. It relies on networktopology+buffer proxy to generate correct alias
        # for the actual layer as it's been doing for in-place buffer.
        # In other words, the implicit layer is added whose buffer names are chosen as in-place buffer names
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_ADDING_IMPLICIT_LAYER')(implicit_permute_layer_name, layer_name))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_PRINT_PERMUTE_ORDER')(str(permute_order)))

        implicit_permute_layer = LayerAdapter(implicit_permute_layer_name, 'SSDPERMUTE', [layer_input_name], [implicit_permute_output_name] )

        # let network_topology add this implicit permute later and deal with BufferProxy mapping
        self._network_topology.add_layer(implicit_permute_layer, 'SSDPERMUTE')

        id_ = self.model.add_permute_layer( name = str(implicit_permute_layer_name),
                                            order = permute_order,
                                            input_name = str(self.get_input_name(implicit_permute_layer)),
                                            output_name = str(self.get_output_name(implicit_permute_layer)) )
        self.add_layer(implicit_permute_layer, id_)

        # Update the axis order based on permute order
        target_input_axis_order = self._axis_transformer.get_target_axis_order(self.get_input_name(implicit_permute_layer))
        target_output_axis_order = []
        for idx in permute_order:
            target_output_axis_order.append(target_input_axis_order[idx])
        self._axis_transformer.update_target_axis_order('SSDPERMUTE', len(permute_order), self.get_output_name(implicit_permute_layer),
                                                        len(permute_order), self.get_input_name(implicit_permute_layer), target_output_axis_order)
        self.save_axis_order(self.get_output_name(implicit_permute_layer))

        # Update the proxy so that wherever the input to this layer is referred, it points to this layer's output
        self._network_topology.install_buffer_proxy(layer_input_name, implicit_permute_output_name)

    def add_prelu_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        prelu_param = layer.prelu_param
        if prelu_param.channel_shared:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_PRELU_NON_CHANNEL_SHARED_SUPPORT_ONLY')(str(layer.name)))
        bias = self.weight_provider.get_prelu_weights(layer)
        id_ = self.model.add_prelu_layer( name = str(layer.name),
                                          coeff = bias,
                                          input_name = str(self.get_input_name(layer)),
                                          output_name = str(self.get_output_name(layer)))
        self.add_layer(layer, id_)

    def add_power_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        power_param = layer.power_param
        scale = 1.0
        shift = 0.0
        power = 1.0

        if (hasattr(power_param, "scale")):
            scale = float(power_param.scale)

        if (hasattr(power_param, "shift")):
            shift = float(power_param.shift)

        if (hasattr(power_param, "power")):
            power = float(power_param.power)

        id_ = self.model.add_power_layer( name = str(layer.name),
                                               scale = scale,
                                               shift = shift,
                                               power = power,
                                               input_name = str(self.get_input_name(layer)),
                                               output_name = str(self.get_output_name(layer)))

        self.add_layer(layer,id_)

    def add_python_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))

        python_param = layer.python_param
        py_layer = python_param.layer
        py_param_str = python_param.param_str

        if py_layer == 'ProposalLayer':
            self.add_proposal_layer(layer)
        else:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_UNSUPPORTED_PYTHON_MODULE')(str(layer.name), py_layer))

    def add_proposal_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))

        python_param = layer.python_param
        py_param_str = python_param.param_str
        if not len(py_param_str):
             raise ValueError(
                 code_to_message.get_error_message('ERROR_CAFFE_PROPOSAL_LAYER_MISSING_PARAM_STR_FIELD')((layer.name)))

        layer_params = yaml.load(py_param_str)
        feat_stride = layer_params['feat_stride']
        scales_ = layer_params.get('scales', (8, 16, 32))
        scales = list(map(float, scales_))
        ratios_ = layer_params.get('ratios', (0.5, 1.0, 2.0))
        ratios = list(map(float, ratios_))
        anchor_base_size = layer_params.get('anchor_base_size', 16)
        min_bbox_size = float(layer_params.get('min_bbox_size', 16.0))
        max_num_proposals = layer_params.get('max_num_proposals', 6000)
        max_num_rois = layer_params.get('max_num_rois', 1) # Output the top 1 ROI if max_num_rois is not specified
        iou_threshold_nms = float(layer_params.get('iou_threshold_nms', 0.7))

        input_dims = self.get_input_dims(layer)

        id_ = self.model.add_proposal_layer( str(layer.name),
                                             feat_stride,
                                             scales,
                                             ratios,
                                             anchor_base_size,
                                             min_bbox_size,
                                             max_num_proposals,
                                             max_num_rois,
                                             iou_threshold_nms,
                                             self.get_input_names(layer),
                                             str(self.get_output_name(layer)))
        output_dim = self.model.get_buffer_dims(str(self.get_output_name(layer)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS')(tuple(output_dim)))
        self.add_layer(layer, id_)

    def add_relu_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        id_ = self.model.add_neuron_layer( name = str(layer.name),
                                           func = modeltools.NEURON_RELU,
                                           input_name = str(self.get_input_name(layer)),
                                           output_name = str(self.get_output_name(layer)))
        self.add_layer(layer, id_)

    def add_elu_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        id_ = self.model.add_neuron_layer( name = str(layer.name),
                                           func = modeltools.NEURON_ELU,
                                           input_name = str(self.get_input_name(layer)),
                                           output_name = str(self.get_output_name(layer)),
                                           a = 1.0)
        self.add_layer(layer, id_)

    def add_rnorm_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        lrn_param = layer.lrn_param

        if lrn_param.norm_region == lrn_param.ACROSS_CHANNELS:
            id_ = self.model.add_cmrn_layer( name = str(layer.name),
                                             window_size = lrn_param.local_size,
                                             # adjust for calculation difference
                                             alpha = float(lrn_param.alpha)/lrn_param.local_size,
                                             beta = lrn_param.beta,
                                             k = lrn_param.k,
                                             input_name = str(self.get_input_name(layer)),
                                             output_name = str(self.get_output_name(layer)))
        else:
            id_ = self.model.add_local_norm_layer( name = str(layer.name),
                                                   window_size = lrn_param.local_size,
                                                   alpha = lrn_param.alpha,
                                                   beta = lrn_param.beta,
                                                   k = float(lrn_param.k),
                                                   input_name = str(self.get_input_name(layer)),
                                                   output_name = str(self.get_output_name(layer)))

        self.add_layer(layer,id_)

    def add_reshape_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        # There are 2 different layers in Caffe that are mapped to the SNPE Reshape layer.
        #  - For "Reshape", the "shape" BlobShape parameter defines the output dimensions, with a 0
        #    indicating an unchanged dimension (to be copied from the corresponding input dimension,
        #    and -1 indicating all remaining dimensionality to be folded into this dimension.
        #    Additionally, Reshape has the axis parameter which specifies the first dimension to be
        #    included in the reshape operation (default 0) and the num_axis parameter which specifies
        #    how many of the dimensions to include (default -1 meaning all)
        #  - For "Flatten", the axis (default 1) and end_axis (default -1 meaning last) are used to
        #    determine which dimensions are to be folded into the single output dimension.
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        bufname = self._network_topology.get_output_buffer_name(replaced_bottom[0])
        input_dims = list(map(int, self.model.get_buffer_dims(bufname)))
        typ = str(layer.type).upper()
        output_dims = []
        if typ == "RESHAPE":
           input_size = reduce(int.__mul__, input_dims)
           output_dims = list(map(int, layer.reshape_param.shape.dim))
           axis = layer.reshape_param.axis
           num_axes = layer.reshape_param.num_axes
           if axis < 0: axis = len(input_dims) + axis
           if num_axes < 0: num_axes = len(input_dims) - axis
           # replace any 0 in the output_dims with the corresponding dimension in the input_dims.
           output_dims = list(map(lambda x: input_dims[x+axis] if output_dims[x]==0 else output_dims[x], range(len(output_dims))))
           # prefix/postfix
           output_dims = input_dims[:axis] + output_dims + input_dims[axis+num_axes:]
           # replace -1 in the output by the remainder of the inputs
           remainder_index = [i for i, j in enumerate(output_dims) if j==-1]
           if len(remainder_index)==1:
               output_size = -1*reduce(int.__mul__, output_dims) # multiply by -1 to make this positive
               output_dims[remainder_index[0]] = input_size / output_size

        if typ == "FLATTEN":
           axis = layer.flatten_param.axis
           end_axis = layer.flatten_param.end_axis
           if axis < 0: axis = len(input_dims) + axis
           if end_axis < 0: end_axis = len(input_dims) + end_axis
           output_dims = [ reduce(int.__mul__, input_dims[axis:end_axis+1]) ]
           output_dims = input_dims[:axis] + output_dims + input_dims[end_axis+1:]

        id_ = self.model.add_reshape_layer( name = str(layer.name),
                                            output_dimensions = output_dims,
                                            input_name = str(self.get_input_name(layer)),
                                            output_name = str(self.get_output_name(layer)))
        self.add_layer(layer, id_)

    def add_roipooling_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        roi_pool_param = layer.roi_pooling_param

        pooled_size_h = roi_pool_param.pooled_h
        pooled_size_w = roi_pool_param.pooled_w
        spatial_scale = roi_pool_param.spatial_scale

        input_dims = self.get_input_dims(layer)

        # The output depth is equal to the input feature map depth. We are assuming that input[0] is the feature map.
        output_dim = [input_dims[0][0], pooled_size_h, pooled_size_w, input_dims[0][3]]

        id_ = self.model.add_roipooling_layer( str(layer.name),
                                               pooled_size_h,
                                               pooled_size_w,
                                               spatial_scale,
                                               output_dim,
                                               self.get_input_names(layer),
                                               str(self.get_output_name(layer)))
        output_dim = self.model.get_buffer_dims(str(self.get_output_name(layer)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS')(tuple(output_dim)))
        self.add_layer(layer, id_)

    def add_scale_layer(self, layer):
        layer_seq = self._blob_connectivity_map.check_s_folding(layer)
        bn_layer_seq = layer_seq[0]
        scale_layer_seq = layer_seq[1]
        # Only add a scale layer when the previous layer is not batchnorm or if
        # batchnorm folding is disabled
        if bn_layer_seq is None or self.disable_batchnorm_folding:
            self.add_scale_layer_util(layer)

    def add_scale_layer_util(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        c_input_names = self.get_input_names(layer)
        self.logger.debug(str(c_input_names))
        input_depths = [ self.model.get_buffer_dims(name)[-1] for name in c_input_names ]

        scaleParam = layer.scale_param
        bias_term = getattr(scaleParam, "bias_term", False)
        axis = scaleParam.axis
        if axis < 0:
            axis = len(input_dim) + axis
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        axis = self._axis_transformer.get_target_axis(replaced_bottom[0], axis, self.get_input_name(layer))
        num_axes = scaleParam.num_axes

        # Get the input depth
        input_layer_name = str(layer.bottom[0])
        bufname = self._network_topology.get_output_buffer_name(input_layer_name)
        input_depth = self.model.get_buffer_dims(bufname)[-1]
        c_weights, c_bias = self.weight_provider.get_scale_weights(layer, bias_term, input_depth)
        if c_weights is not None:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_WEIGHTS_SHAPE')(str(c_weights.shape)))

        id_ = self.model.add_scale_layer( name = str(layer.name),
                                          weights = c_weights,
                                          bias = c_bias,
                                          input_names = c_input_names,
                                          output_name = str(self.get_output_name(layer)),
                                          axis = axis,
                                          num_axes = num_axes)
        self.add_layer(layer, id_)

    def add_softmax_layer(self, layer):
        id_ = self.model.add_softmax_layer( name = str(layer.name),
                                            input_name = str(self.get_input_name(layer)),
                                            output_name = str(self.get_output_name(layer)))

        self.add_layer(layer, id_)

    def add_ssd_detection_output_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))

        # Use the passed priorbox_input_name to lookup the priorbox data destined for this layer as the actual input
        # was removed earlier to prevent input dimension calculation for an input that doesn't actually exist
        priorbox_data = self.total_prior_box_output[self.priorbox_input_name][0]+self.total_prior_box_output[self.priorbox_input_name][1]

        params = layer.detection_output_param if layer.type == 'DetectionOutput' else layer.ssd_detection_output_param
        nms_param = params.nms_param

        if ("keep_top_k" not in str(params)):
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_MISSING_SSD_PARAM')(str(layer.name), 'keep_top_k'))
        if (int(params.keep_top_k) < 0):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')(str(layer.name), 'keep_top_k', int(params.keep_top_k)))

        # 7: [image_batch, label, confidence, xmin, ymin, xmax, ymax]
        # 0 dimension indicates dynamic resizing of # of outputs
        input_dim = self.get_input_dim(layer)
        output_dims = [[input_dim[0], 1, 0, 7]]

        code_map = { caffe_pb2.PriorBoxParameter.CodeType.Value('CORNER') : modeltools.PRIORBOX_TYPE_CORNER,
                     caffe_pb2.PriorBoxParameter.CodeType.Value('CENTER_SIZE') : modeltools.PRIORBOX_TYPE_CENTER_SIZE,
                     caffe_pb2.PriorBoxParameter.CodeType.Value('CORNER_SIZE') : modeltools.PRIORBOX_TYPE_CORNER_SIZE }
        code_type = code_map[params.code_type]

        id_ = self.model.add_ssd_detection_output_layer(name=str(layer.name),
                                                        input_names=self.get_input_names(layer),
                                                        output_names=self.get_output_names(layer),
                                                        output_dims=output_dims,
                                                        num_classes=params.num_classes,
                                                        share_location=params.share_location,
                                                        background_label_id=params.background_label_id,
                                                        nms_threshold=nms_param.nms_threshold,
                                                        nms_top_k=nms_param.top_k,
                                                        nms_eta=nms_param.eta,
                                                        code_type=code_type,
                                                        prior_data=priorbox_data,
                                                        keep_top_k=params.keep_top_k,
                                                        variance_encoded_in_target=params.variance_encoded_in_target,
                                                        confidence_threshold=params.confidence_threshold)
        self.add_layer(layer, id_)

    def add_tanh_layer(self, layer):
        id_ = self.model.add_neuron_layer( name=str(layer.name),
                                           func = modeltools.NEURON_TANH,
                                           input_name = str(self.get_input_name(layer)),
                                           output_name = str(self.get_output_name(layer)),
                                           a=1.0,
                                           b=1.0 )
        self.add_layer(layer, id_)

    def add_upsample_layer(self, layer):
        name = str(layer.name)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))

        upsample_param = layer.upsample_param
        if upsample_param.upsample_mode == upsample_param.DENSE:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_SUPPORT_DENSE_UPSAMPLING')(str(layer.name)))
        # If sparse mode is not enabled, extract params from pooling layer
        elif upsample_param.upsample_mode != upsample_param.SPARSE:
           replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
           pool_name = str(replaced_bottom[1])
           pool_parms = self.pool_parms_map[pool_name]

           id_ = self.model.add_upsample_layer(name,
                                               pool_parms['size'],
                                               pool_parms['stride'],
                                               pool_parms['pad'],
                                               pool_parms['input_dim'][1],
                                               pool_parms['input_dim'][2],
                                               str(self.get_input_name(layer)),
                                               str(self.get_output_name(layer)),
                                               pool_parms['id'])
           self.add_layer(layer, id_)
        # Otherwise, use scale, upsample_h and upsample_w fields from upsample
        # param
        else:
           input_dim = self.model.get_buffer_dims(str(self.get_input_name(layer)))
           if upsample_param.upsample_h != 0:
              output_height = upsample_param.upsample_h
           else:
              output_height = input_dim[0]*upsample_param.scale

           if upsample_param.upsample_w != 0:
              output_weight = upsample_param.upsample_w
           else:
              output_weight = input_dim[1]*upsample_param.scale

           id_ = self.model.add_upsample_layer(name,
                                               upsample_param.scale,
                                               upsample_param.scale,
                                               0,
                                               output_height,
                                               output_weight,
                                               str(self.get_input_name(layer)),
                                               str(self.get_output_name(layer)))

           self.add_layer(layer, id_)

    def add_slice_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))

        input_dim = list(self.model.get_buffer_dims(str(self.get_input_name(layer))))

        # By default, slice_axis is 1
        slice_axis = 1

        if layer.slice_param.HasField('slice_dim'):
           slice_axis = layer.slice_param.slice_dim
           self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_SLICE_DIM'))
        else:
           try:
             slice_axis = layer.slice_param.axis
             self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_AXIS'))
             # Since axis parameter could contain -ve value, let's turn it to +ve
             if slice_axis < 0:
               slice_axis = len(input_dim) + slice_axis
           except AttributeError:
             self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_DEFINE_SLICE_DIM_AXIS_FIELD'))
             self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_AXIS_DEFAULT_FOR_LAYER')(str(layer.type), layer.name))
             pass

        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)

        id_ = self.model.add_slice_layer( name = str(layer.name),
                                          input_name = str(self.get_input_name(layer)),
                                          axis = self._axis_transformer.get_target_axis(replaced_bottom[0], slice_axis, self.get_input_name(layer)),
                                          slice_points = [int(v) for v in layer.slice_param.slice_point],
                                          output_names = self.get_output_names(layer) )
        self.add_layer(layer, id_)

    def get_input_id(self, layer):
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_name = replaced_bottom[0]
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_GET_INPUT_ID_BUFFER')(layer.name, input_name))
        return self.layer_id_map[str(input_name)]

    def get_input_name(self, layer):
        replaced_bottom =  self._blob_connectivity_map.replace_optimized_blobs(layer)
        input_names = self.pick_input_names([replaced_bottom[0]], layer.type)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_GET_INPUT_NAME_BUFFER')(layer.name, replaced_bottom[0], input_names[0]))
        return input_names[0]

    def get_input_id_list(self, layer):
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        for name in replaced_bottom:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_GET_INPUT_ID_LIST_BUFFER')(layer.name, name))
            ret.append(self.layer_id_map[str(name)])
        return ret

    def get_input_names(self, layer):
        ret = []
        replaced_bottom = self._blob_connectivity_map.replace_optimized_blobs(layer)
        ret = self.pick_input_names(replaced_bottom, layer.type)
        for name in ret:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_GET_INPUT_NAMES_BUFFER')(layer.name, name))
        return ret

    def get_output_names(self, layer):
        ret = []
        for name in layer.top:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_GET_OUTPUT_NAMES_BUFFER')(layer.name, name))
            bufname = self._network_topology.get_output_buffer_name(name)
            ret.append(bufname)
        return ret

    def get_output_name(self, layer):
        output_name = bufname = self._network_topology.get_output_buffer_name(layer.top[0])
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_GET_OUTPUT_NAME_BUFFER')(layer.name, output_name))
        return output_name

    def get_input_dim(self, layer):
        return self.model.get_buffer_dims(str(self.get_input_name(layer)))

    def get_input_dims(self, layer):
        ret = []
        bufs = self.get_input_names(layer)
        for inp in bufs:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_GET_INPUT_DIMS')(inp))
            dim = self.model.get_buffer_dims(inp)
            # dim has to be a list
            ret.append(list(dim))
        return ret

    def get_output_dims(self, layer):
        ret = []
        bufs = self.get_output_names(layer)
        for out in bufs:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_GET_OUTPUT_DIMS')(out))
            dim = self.model.get_buffer_dims(out)
            # dim has to be a list
            ret.append(list(dim))
        return ret

    def save_axis_order(self, buffer_name):
        axis_order = self._axis_transformer.get_target_axis_order(buffer_name)
        if len(axis_order):
            self.model.set_buffer_axis_order(str(buffer_name), list(axis_order))

    def get_output_dim(self, layer):
        return self.model.get_buffer_dims(str(self.get_output_name(layer)))

    def get_conv_params(self, convParam):
        parmstype = collections.namedtuple("ConvParams",
                                           ["padx", "pady", "stridex", "stridey", "kx", "ky"])
        padx, pady = 0,0
        if convParam.pad_h or convParam.pad_w:
            padx = convParam.pad_w
            pady = convParam.pad_h
        elif isinstance(convParam.pad, int):
            # Segnet version of caffe.proto has defined  pad optional (not repeated).
            # It implies that it is scalar rather vector
            padx = convParam.pad
            pady = convParam.pad
        else:
            if len(convParam.pad) > 0:
                padx = convParam.pad[0]
                pady = convParam.pad[0]
            if len(convParam.pad) > 1:
                padx = convParam.pad[1]

        stridex, stridey = 1, 1
        if convParam.stride_h or convParam.stride_w:
            stridex = convParam.stride_w
            stridey = convParam.stride_h
        elif isinstance(convParam.stride, int):
            # Segnet version of caffe.proto has defined  stride optional (not repeated).
            # It implies that it is scalar rather vector
            stridex = convParam.stride
            stridey = convParam.stride
        else:
            if len(convParam.stride) > 0:
                stridex = convParam.stride[0]
                stridey = convParam.stride[0]
            if len(convParam.stride) > 1:
                stridex = convParam.stride[1]

        kx = 0
        ky = 0
        if convParam.kernel_h and convParam.kernel_w:
            kx = convParam.kernel_w
            ky = convParam.kernel_h
        if isinstance(convParam.kernel_size, int):
            kx = convParam.kernel_size
            ky = convParam.kernel_size
        else:
            if len(convParam.kernel_size) > 0:
                kx = convParam.kernel_size[0]
                ky = convParam.kernel_size[0]
            if len(convParam.kernel_size) > 1:
                kx = convParam.kernel_size[1]
        if kx == 0  or ky == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_SIZE'))

        return parmstype(padx, pady, stridex, stridey, kx, ky)

    def get_ssd_aspect_ratios(self, params):
        aspect_ratios = [1.]
        for val in params.aspect_ratio:
            ar = val
            already_exist = False
            for prior in aspect_ratios:
                if math.fabs(ar - prior) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratios.append(ar)
                if params.flip == True:
                    aspect_ratios.append(1. / ar)
        return aspect_ratios

    def get_ssd_num_priors(self, params):
        # OPEN_SOURCE_START
        # The following code is derived based on code from the following open source projects/packages.
        # Project name: caffe
        # Branch: ssd
        # Note: There are few minor changes to accomodate for SNPE framework

        # determine how many aspect ratios user
        # provided to calculate the number of
        # prior boxes
        aspect_ratios = self.get_ssd_aspect_ratios(params)
        num_priors = int(len(aspect_ratios) * len(params.min_size))

        if "max_size" in str(params):
            for val in params.max_size:
                num_priors += 1

        return num_priors

        # OPEN_SOURCE_END

    def process_ssd_priorbox_concat_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        input_names = self.get_input_names(layer)
        output_names = self.get_output_names(layer)

        concatenated_priorbox_data = []
        concatenated_priorbox_variance = []
        self.total_prior_box_output[output_names[0]] = []
        for input_name in input_names:
            if input_name not in self.total_prior_box_output:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_MISSING_SSD_PARAM')(str(layer.name),
                                                                                              'priorbox output for: ' + str(input_name)))
            concatenated_priorbox_data.extend(self.total_prior_box_output[input_name][0])
            concatenated_priorbox_variance.extend(self.total_prior_box_output[input_name][1])

        self.total_prior_box_output[output_names[0]] = [concatenated_priorbox_data, concatenated_priorbox_variance]

    def process_ssd_priorbox_layer(self, layer):
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        input_dims = self.get_input_dim(layer)

        output_names = self.get_output_names(layer)
        if len(output_names) != 1:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')(str(layer.name), 'num prior box outputs', len(layer.top)))

        # HWC format
        input_layer_height = input_dims[1]
        input_layer_width = input_dims[2]
        model_input_height = self.input_dim[1]
        model_input_width = self.input_dim[2]
        prior_box_params = layer.prior_box_param if layer.type == 'PriorBox' else layer.ssd_prior_box_param

        if prior_box_params.step_w == 0 or prior_box_params.step_h == 0:
            step_w = float(model_input_width) / input_layer_width
            step_h = float(model_input_height) / input_layer_height
        else:
            step_w = prior_box_params.step_w
            step_h = prior_box_params.step_h
        min_sizes = [min_size for min_size in prior_box_params.min_size]
        max_sizes = [max_size for max_size in prior_box_params.max_size]
        aspect_ratios = self.get_ssd_aspect_ratios(prior_box_params)
        variances = [variance for variance in prior_box_params.variance]

        num_priors = self.get_ssd_num_priors(prior_box_params)
        output_dim = (input_layer_height * input_layer_width * num_priors * 4)

        prior_box_output = []
        prior_box_variances = []

        for h in range(0, input_layer_height):
            for w in range(0, input_layer_width):
                center_x = (w + prior_box_params.offset) * step_w
                center_y = (h + prior_box_params.offset) * step_h

                for s in range(0, len(min_sizes)):
                    # first prior: aspect_ratio = 1, size = min_size
                    min_size = min_sizes[s]
                    box_width = box_height = min_size
                    # xmin
                    prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                    # ymin
                    prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                    # xmax
                    prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                    # ymax
                    prior_box_output.append((center_y + box_height / 2.) / model_input_height)

                    if len(max_sizes) > 0:
                        if len(min_sizes) != len(max_sizes):
                           raise ValueError(
                               code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')(str(layer.name), 'Number of min and max size for SsdPriorbox must be same', str(len(min_sizes)) + ',' + str(len(max_sizes))))

                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = math.sqrt(min_size * max_size)
                        # xmin
                        prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                        # ymin
                        prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                        # xmax
                        prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                        # ymax
                        prior_box_output.append((center_y + box_height / 2.) / model_input_height)

                    # rest of priors
                    for r in range(0, len(aspect_ratios)):
                        ar = aspect_ratios[r]
                        if math.fabs(ar - 1.) < 1e-6:
                            continue
                        box_width = min_size * math.sqrt(ar)
                        box_height = min_size / math.sqrt(ar)
                        # xmin
                        prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                        # ymin
                        prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                        # xmax
                        prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                        # ymax
                        prior_box_output.append((center_y + box_height / 2.) / model_input_height)

        # clip the prior's coordidate such that it is within [0, 1]
        if prior_box_params.clip:
            for d in range(0, output_dim):
                prior_box_output[d] = min(max(prior_box_output[d], 0.), 1.)

        # set the variances in separate array and collectively add the end of all the priorboxes.
        # This is since we are concatinating on axis 1
        # Below is the implementation for this in caffe: top_data += top[0]->offset(0, 1);
        if len(variances) == 1:
            # implementing this as follows: caffe_set < Dtype > (output_dim, Dtype(variance_[0]), top_data);
            if variances[0] == 0:
                prior_box_variances.extend(repeat(0, output_dim))  # NOLINT(caffe / alt_fn)
            else:
                for i in range(0, output_dim):
                    prior_box_variances.append(variances[0])
        else:
            for h in range(0, input_layer_height):
                for w in range(0, input_layer_width):
                    for i in range(0, num_priors):
                        for j in range(0, 4):
                            prior_box_variances.append(variances[j])

        # Save the priorbox and variance data to be processed during the "ssd concat layer"
        self.total_prior_box_output[output_names[0]] = [prior_box_output, prior_box_variances]

    def setup_preprocessing(self, data_name, transform_param):
        if self.enable_preprocessing:
            if self.preprocessing_is_setup:
                raise RuntimeError(
                    code_to_message.get_error_message('ERROR_CAFFE_PREPROCESSING_SET_TWICE_ON_MULTIPLE_INPUTS'))
            else:
                self.preprocessing_is_setup = True
                self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_SETTING_UP_PREPROCESSING'))


        if (len(self.input_dim) > 0) and self.input_size is not None and self.enable_preprocessing:
            self.network_dim = self.input_dim
            # Preserve the batch dimension for input_dim (prototxt) while allowing users to add scaling
            # layer with different input sizes. In the future, we could permit users to overwrite batch dimension
            # of prototxt via command line arg.
            self.input_dim = [self.input_dim[0], int(self.input_size[1]), int(self.input_size[0]), 3]

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_ADDING_DATA_LAYER_W_DIMS')(data_name, ",".join(map(str, self.input_dim))))
        if self.encoding == "nv21":
            # Force input_type to be image
            input_type = self.input_types_map.get(data_name, "image")
            if input_type != "image":
                raise RuntimeError(code_to_message.get_error_message('ERROR_CAFFE_CANNOT_SET_INPUT_TYPE_NV21_ENCODING'))
        else:
            input_type = self.input_types_map.get(data_name, "default")
        id_ = self.model.add_data_layer( data_name, self.input_dim, self.encoding, "bgr", input_type)
        last_layer_name = data_name

        # Update the axis order of data buffers
        self._axis_transformer.update_src_axis_order('DATA', len(self.input_dim), data_name, len(self.input_dim))
        self._axis_transformer.update_target_axis_order('DATA', len(self.input_dim), data_name, len(self.input_dim))
        self.save_axis_order(data_name)

        if not self.enable_preprocessing:
            self.layer_id_map[data_name] = id_ # layer name is blob name
            return

        if self.network_dim:
            implicit_scale_layer_name = last_layer_name + "_scale"
            implicit_scale_output_name = implicit_scale_layer_name
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_ADDING_IMPLICIT_SCALE_LAYER')(implicit_scale_layer_name, last_layer_name, str(self.network_dim)))
            # in scaling, input and output are the same
            implicit_layer = LayerAdapter(implicit_scale_layer_name, 'SCALE',
                                          [last_layer_name], [implicit_scale_output_name])
            # let network_topology add this implicit scaler later and deal with BufferProxy mapping
            self._network_topology.add_implicit_scale_layer(implicit_layer, self.model)
            id_ = self.model.add_scaling_layer(name = implicit_scale_layer_name,
                                               output_dimensions = self.network_dim,
                                               pad_value = 0,
                                               maintain_aspect_ratio = False,
                                               resize_mode = modeltools.RESIZE_BILINEAR,
                                               scale_height = 0.0,
                                               scale_width = 0.0,
                                               input_name = last_layer_name,
                                               output_name = implicit_scale_output_name,
                                               align_corners = False)
            scale_dims_string = ",".join(map(str,self.network_dim))
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE_ADDED_IMPLICIT_SCALE_LAYER_W_DIMS')(scale_dims_string))
            data_output_dim = self.model.get_buffer_dims(implicit_scale_output_name)
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_SANITY_CHECK_DIM_OF_LAYER')(last_layer_name, str(data_output_dim)))
            self._axis_transformer.update_src_axis_order('SCALE', len(self.input_dim), implicit_scale_output_name, len(self.input_dim), last_layer_name)
            self._axis_transformer.update_target_axis_order('SCALE', len(self.input_dim), implicit_scale_output_name, len(self.input_dim), last_layer_name)
            self.save_axis_order(implicit_scale_output_name)

            last_layer_name = implicit_scale_layer_name
        else:
            self.network_dim = self.input_dim

        if transform_param is None:
            self.layer_id_map[data_name] = id_ # layer name is blob name
            return


        # add cropping if present
        crop_size = transform_param.crop_size
        if len(self.network_dim) == 4:
            original_height = self.network_dim[1]
            original_weight = self.network_dim[2]
        else:
            original_height = self.network_dim[0]
            original_weight = self.network_dim[1]

        if  crop_size != 0 and (crop_size != original_height or crop_size != original_weight):
            if crop_size > original_height or crop_size > original_weight:
                errs = code_to_message.get_error_message('ERROR_CAFFE_CROP_SIZE_LARGER_THAN_INPUT_DIMS')(crop_size, self.network_dim[0:2])
                raise ValueError(errs)
            offset_y = (original_height-crop_size)//2
            offset_x = (original_weight-crop_size)//2
            if len(self.network_dim) == 4:
                offsets = [0,offset_y,offset_x,0]
                output_dim = [self.input_dim[0], crop_size, crop_size, self.input_dim[3]]
            else:
                offsets = [offset_y, offset_x, 0]
                output_dim = [crop_size, crop_size, self.input_dim[2]]
            crop_layer_name = "%s_crop" % data_name
            implicit_crop_layer = LayerAdapter(crop_layer_name, 'CROP',
                                          [last_layer_name], [crop_layer_name])
            self._network_topology.add_layer(implicit_crop_layer, 'CROP')

            # This part makes sure add_crop_layer takes the right input buffer name
            # it cannot blindly take last_layer_name as the input name since
            # if scaling (e.g.)  comes before, this input name != last_layer_name
            # It would have a different name
            self.logger.debug("get_output_buffer_name of " + last_layer_name + " : " + self._network_topology.get_output_buffer_name(last_layer_name))
            self.logger.debug("output buffers of " + last_layer_name + " : " + str(self._network_topology.get_output_buffers(last_layer_name)))

            # add_crop_layer takes a single input, so lets reduce the array to a single one and just
            # make sure as a sanity check, that we indeed have just a single one
            crop_input_buffer = self._network_topology.get_output_buffers(last_layer_name)
            if len(crop_input_buffer) > 1:
                self.logger.warn("crop_input_buffer seems to have more than one elements, expecting one, taking the fisrt one " + crop_input_buffer[0])
            crop_input_buffer = str(crop_input_buffer[0])

            id_ = self.model.add_crop_layer(crop_layer_name,
                                            offsets,
                                            output_dim,
                                            crop_input_buffer,
                                            crop_layer_name)

            self._axis_transformer.update_src_axis_order('CROP', len(self.input_dim), crop_layer_name, len(self.input_dim), last_layer_name)
            self._axis_transformer.update_target_axis_order('CROP', len(self.input_dim), crop_layer_name, len(self.input_dim), last_layer_name)
            self.save_axis_order(crop_layer_name)

            last_layer_name = crop_layer_name
            if len(self.network_dim) == 4:
                crop_dim = [self.network_dim[0], crop_size, crop_size, self.network_dim[3]]
            else:
                crop_dim = [crop_size, crop_size, self.network_dim[2]]
        else:
            crop_dim = self.network_dim[:]

        # as per caffe, either mean_file or mean_value may be specified, but not both.
        mean_file = transform_param.mean_file
        mean_data = None

        if mean_file:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE_PREPROCESSING_MEAN_DATA_FILE')(mean_file))
            if not path.isabs(mean_file):
                mean_file = path.join(path.dirname(self.caffe_prototext_path), mean_file)
            mean_data_file = open(mean_file, "rb")
            mean_data_blob = caffe_pb2.BlobProto()
            mean_data_blob.ParseFromString(mean_data_file.read())
            mean_data_file.close()
            # can't call blob2 arr because this is a BlobProto rather than a Blob.

            mean_data = numpy.array(mean_data_blob.data, dtype=numpy.float32)
            shape = tuple(mean_data_blob.shape.dim)

            if not shape: # shape is stored in old way
                shape = (mean_data_blob.num, mean_data_blob.channels, mean_data_blob.height, mean_data_blob.width)

            mean_data.shape = shape
            # roll the channels to the back
            mean_data = numpy.ascontiguousarray(numpy.rollaxis(mean_data, 1, 4))

            if mean_data.shape[1] < crop_dim[1] or mean_data.shape[2] < crop_dim[2]:
                errs = code_to_message.get_error_message('ERROR_CAFFE_MEAN_DATA_NOT_LARGE_ENOUGH')(mean_data.shape, tuple(crop_dim))
                raise ValueError(errs)
            # crop to the input shape if necessary
            if mean_data.shape[1] != crop_dim[1] or mean_data.shape[2] != crop_dim[2]:
                offset_y = (mean_data.shape[1]-crop_dim[1])//2
                offset_x = (mean_data.shape[2]-crop_dim[2])//2
                mean_data = numpy.ascontiguousarray(mean_data[offset_y:offset_y+crop_dim[1],
                                                              offset_x:offset_x+crop_dim[2],
                                                              :])

        elif len(transform_param.mean_value):
            # the length must be either 1, or the number of channels
            if len(transform_param.mean_value) == 1:
                mean_data = numpy.zeros(crop_dim, dtype=numpy.float32)
                mean_data[:] = transform_param.mean_value[0]
            elif len(transform_param.mean_value) == crop_dim[-1]:
                m = numpy.array(list(transform_param.mean_value), dtype=numpy.float32)
                tile_dim = crop_dim[:]
                tile_dim[-1] = 1
                mean_data = numpy.tile(m, tile_dim)
            else:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_INVALID_MEAN_VAL_SPECIFICATION'))

        if mean_data is not None:
            if mean_data.shape != tuple(crop_dim):
                errs = code_to_message.get_error_message('ERROR_CAFFE_MEAN_DATA_WRONG_DIMS')(data_name, str(crop_dim), str(mean_data.shape))
                raise ValueError(errs)
            subtract_mean_layer_name = "%s_subtract_mean" % data_name
            implicit_subtract_mean_layer = LayerAdapter(subtract_mean_layer_name, 'SUBTRACTMEAN',
                                          [last_layer_name], [subtract_mean_layer_name])
            self._network_topology.add_layer(implicit_subtract_mean_layer, 'SUBTRACTMEAN')

            # this is basically a copy paste from add_crop_layer part above (same function, setup_preprocessing)
            # This part makes sure add_crop_layer takes the right input buffer name
            # it cannot blindly take last_layer_name as the input name since
            # if scaling (e.g.)  comes before, this input name != last_layer_name
            # It would have a different name
            self.logger.debug("get_output_buffer_name of " + last_layer_name + " : " + self._network_topology.get_output_buffer_name(last_layer_name))
            self.logger.debug("output buffers of " + last_layer_name + " : " + str(self._network_topology.get_output_buffers(last_layer_name)))

            # add_crop_layer takes a single input, so lets reduce the array to a single one and just
            # make sure as a sanity check, that we indeed have just a single one
            subtract_mean_input_buffer = self._network_topology.get_output_buffers(last_layer_name)
            if len(subtract_mean_input_buffer) > 1:
                self.logger.warn("subtract_mean_input_buffer seems to have more than one elements, expecting one, taking the fisrt one " + subtract_mean_input_buffer[0])
            subtract_mean_input_buffer = str(subtract_mean_input_buffer[0])

            id_ = self.model.add_subtract_mean_layer(subtract_mean_layer_name,
                                                     mean_data,
                                                     subtract_mean_input_buffer,
                                                     subtract_mean_layer_name)

            self._axis_transformer.update_src_axis_order('SUBTRACTMEAN', len(self.input_dim), subtract_mean_layer_name, len(self.input_dim), last_layer_name)
            self._axis_transformer.update_target_axis_order('SUBTRACTMEAN', len(self.input_dim), subtract_mean_layer_name, len(self.input_dim), last_layer_name)
            self.save_axis_order(subtract_mean_layer_name)

            last_layer_name = subtract_mean_layer_name

        self.layer_id_map[data_name] = id_ # layer name is blob name
        # Make the last pre-processing layer's output blob a proxy for the original input layer.
        if last_layer_name != data_name:
            self._network_topology.install_buffer_proxy(data_name, last_layer_name)

    # Add fixed point information for FC or CONV layer.
    def add_fxp_layer_encoding(self, layer):
        # If quantization_param is present, and we do have layer output
        # encodings...
        if hasattr(layer, "quantization_param") and \
                    hasattr(layer.quantization_param, "min_layer_out") and \
                    len(layer.quantization_param.min_layer_out) > 0:
            if False == self.fixed_point_layers_present:
                self.fixed_point_layers_present = True
                fxp_mode_number = layer.quantization_param.fxp_mode
                fxp_mode_string = caffe_pb2.QuantizationParameter.FixedPointMode.Name(fxp_mode_number)
                self.model.set_tf_encoding_type(fxp_mode_string)
                self.model.quantize_weights(True);
            self.model.set_tf_weight_encoding(str(layer.name),
                                              layer.quantization_param.bw_weights,
                                              0,
                                              0,
                                              0)
            self.model.set_tf_output_encoding(str(layer.name),
                                              layer.quantization_param.bw_layer_out,
                                              layer.quantization_param.min_layer_out[0],
                                              layer.quantization_param.max_layer_out[0])

            # If optional accumulator encodings are present set them
            if hasattr(layer.quantization_param, "bw_acc"):
                self.model.set_tf_accumulator_encoding(str(layer.name),
                                                       layer.quantization_param.bw_acc,
                                                       layer.quantization_param.min_acc,
                                                       layer.quantization_param.max_acc)

class Utils(object):
    @staticmethod
    def check_use_global_stats(layer):
        if layer.batch_norm_param.HasField("use_global_stats"):
            return layer.batch_norm_param.use_global_stats
        else:
            return True
