#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import collections
import copy  # deep copy
import logging
import math
import numpy
import pprint  # pretty print for dicts
import sys
from functools import reduce
from caffe2.proto import caffe2_pb2

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

# Importing axis tracking types
from snpe.converters.common.utils import snpe_axis_transformer
from snpe.converters.common.utils import snpe_converter_utils
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import SNPEUtils

AxisAnnotation = snpe_axis_transformer.AxisAnnotation

snpeUtils = SNPEUtils()

#------------------------------------------------------------------------------
#   Specify caffe layers' ordered axes
#   A layer type, not listed here, will assume axis order = N, C, H, W
#------------------------------------------------------------------------------
default_caffe_axes = [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH]
caffe_layer_axes = snpe_axis_transformer.LayerOrderedAxes("Caffe",default_caffe_axes)

# Layers with NONTRIVIAL input/output axis order
caffe_layer_axes.add_axis_order('Reshape', [AxisAnnotation.NONTRIVIAL])
caffe_layer_axes.add_axis_order('Flatten', [AxisAnnotation.NONTRIVIAL])
caffe_layer_axes.add_axis_order('FlattenToVec', [AxisAnnotation.NONTRIVIAL])

# Layers with ANY input/output axis order

# Add both 4d and 3d axis order for slice layer
caffe_layer_axes.add_axis_order('Slice', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('Slice', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('Split', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('Split', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)

# Add both 4d and 3d axis order for concat layer
caffe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

# Add both 4d and 3d axis order for tile layer
caffe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
caffe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
caffe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)

caffe_layer_axes.add_axis_order('FC',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('FC',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

# FIXME: A workaround for unsupported feature: 'axis' parameter support of Softmax.
#        Softmax should be published as AxisAnnotation.ANY once 'axis' parameter of Softmax is supported.
caffe_layer_axes.add_axis_order('Softmax', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('Softmax', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])

caffe_layer_axes.add_axis_order('Relu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('Relu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('PRelu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('PRelu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('Sigmoid', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('Sigmoid', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('Tanh', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('Tanh', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('Elu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])
caffe_layer_axes.add_axis_order('Elu', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

caffe_layer_axes.add_axis_order('Add', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
caffe_layer_axes.add_axis_order('Add', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH])

#------------------------------------------------------------------------------
#   Specify SNPE layers' ordered axes
#   A layer type, not listed here, will assume axis order = H, W, C
#------------------------------------------------------------------------------
default_snpe_axes = [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL]
snpe_layer_axes = snpe_axis_transformer.LayerOrderedAxes("SNPE", default_snpe_axes)

# Layers with NONTRIVIAL input/output axis order
snpe_layer_axes.add_axis_order('Reshape', [AxisAnnotation.NONTRIVIAL])
snpe_layer_axes.add_axis_order('Flatten', [AxisAnnotation.NONTRIVIAL])
snpe_layer_axes.add_axis_order('FlattenToVec', [AxisAnnotation.NONTRIVIAL])

# Layers with ANY input/output axis order

# Add 3d axis order for slice layer
snpe_layer_axes.add_axis_order('Slice', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('Slice', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('Split', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('Split', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)


# Add both 3d and 2d axis order for concat layer
snpe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)
snpe_layer_axes.add_axis_order('Concat', [AxisAnnotation.ANY] * 1, [AxisAnnotation.ANY] * 1)

# Add both 3d and 2d axis order for tile layer
snpe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 4, [AxisAnnotation.ANY] * 4)
snpe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 3, [AxisAnnotation.ANY] * 3)
snpe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 2, [AxisAnnotation.ANY] * 2)
snpe_layer_axes.add_axis_order('Tile', [AxisAnnotation.ANY] * 1, [AxisAnnotation.ANY] * 1)

snpe_layer_axes.add_axis_order('FC',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL],
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('FC',
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL],
                                [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

# FIXME: A workaround for unsupported feature: 'axis' parameter support of Softmax.
#        Softmax should be published as AxisAnnotation.ANY once 'axis' parameter of Softmax is supported.
snpe_layer_axes.add_axis_order('Softmax', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Softmax', [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])

snpe_layer_axes.add_axis_order('Relu',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Relu',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('PRelu',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('PRelu',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Sigmoid',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Sigmoid',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Tanh',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Tanh',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Elu',
                               [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Elu',
                               [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])

snpe_layer_axes.add_axis_order('Add', [AxisAnnotation.BATCH, AxisAnnotation.CHANNEL])
snpe_layer_axes.add_axis_order('Add', [AxisAnnotation.BATCH, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL])

#------------------------------------------------------------------------------
#   Simple Layer wrapper class needed for implicit permute
#------------------------------------------------------------------------------
class LayerAdapter(object):
    def __init__(self, name, typ, input, output):
        self.name = name
        self.type = typ
        self.input = input
        self.output = output

class BufferProxy(object):
    def __init__(self):
        # proxy buffers (proxy_buffer, buf)
        self._output_buffer_proxy = {}
        # pending input buffer proxies
        self._pending_input_buffer_proxy = {}
        # output Buffer seen so far
        self._output_buffers = []
        self._logger = logging.getLogger()

    def dump(self):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_OUTPUT_BUFFER_DUMP')(str(len(self._output_buffers))))
        for v in self._output_buffers:
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_OUTPUT_BUFFER_PRINT')(str(v)))
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_INPUT_BUFFER_DUMP')(str(len(list(self._pending_input_buffer_proxy.keys())))))
        for k,v in list(self._pending_input_buffer_proxy.items()):
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_KEY_VALUE_PRINT')(str(k), str(v)))
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_OUTPUT_DUMP')(str(len(list(self._output_buffer_proxy.keys())))))
        for k,v in list(self._output_buffer_proxy.items()):
            self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_KEY_VALUE_PRINT')(str(k), str(v)))

    def _snapshot_output_buffer_proxy(self):
        self._pending_input_buffer_proxy.clear() # not mandatory
        self._pending_input_buffer_proxy = copy.deepcopy(self._output_buffer_proxy)

    def install_buffer_proxy(self, original_buffer, proxy_buffer):
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_INSTALL_BUFFER_PROXY'))
        self.dump()

        # create mapping
        self._output_buffer_proxy[original_buffer] = proxy_buffer
        self._pending_input_buffer_proxy[original_buffer] = proxy_buffer

        self.dump()

    def add_layer(self, op, model, layer_type):
        layername = op.name
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADDING_LAYER')(layername))

        # keep a current snapshot before we do anything
        self._snapshot_output_buffer_proxy()

        # Record/generate
        for t in op.output:
            bufname = str(t)
            self._logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER')(layername, bufname))
            if t in self._output_buffers:
                alias = self._gen_alias(bufname, layername)
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_ALIAS_TO')(layername, bufname, alias))
                # must update model so we can have it registered in the C++ domain
                dims = model.get_buffer_dims(bufname)
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_DESCR')(bufname, str(dims), str(list(dims))))
                model.register_buffer(str(alias), list(dims))

                # need to add proxy
                self._output_buffer_proxy[bufname] = alias
                self._logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_PROXY_TO')(layername, bufname, alias))
                # override bufname with alias, this would
                # be get into the _output_buffers below
                bufname = alias

            self._logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER')(layername, bufname))
            self._output_buffers.append(bufname)
            self.dump()

    def input_proxy(self):
        return self._pending_input_buffer_proxy

    def output_proxy(self):
        return self._output_buffer_proxy

    def get_output_proxy_buffer(self, bufname):
        return self._output_buffer_proxy[str(bufname)]

#------------------------------------------------------------------------------
#   Class for tracking network ops and generating/tracking unique names
#------------------------------------------------------------------------------
class NetworkTopology(object):
    def __init__(self, modeltools):
        self._model = modeltools
        self._layers = {}

        # Input/Output buffer Proxies
        # output Buffer seen so far
        self._output_buffers = []
        self._buffer_map = {}

        # op/layer 'names'
        self._op_names = []
        self._op_name_id_map = {}

        self._proxy = BufferProxy()
        self._logger = logging.getLogger()

    def install_buffer_proxy(self, original_buffer, proxy_buffer):
        self._proxy.install_buffer_proxy(original_buffer, proxy_buffer)

    def _get_unique_name(self, op, name):

        if op.type in self._op_name_id_map:
            op_id = self._op_name_id_map[op.type]+1
        else:
            op_id = 0

        while (name+str(op_id)) in self._op_names:
            op_id += 1

        self._op_name_id_map[op.type] = op_id
        return (name+str(op_id))

    def _gen_layer_alias(self, op):
        op_name = str(op.name) + "."
        alias = self._get_unique_name(op, op_name)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_LAYER_ALIAS_GEN')(op_name, alias))
        return str(alias)

    def _gen_layer_name(self, op):
        name = str(op.type) + "."
        name = self._get_unique_name(op, name)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_LAYER_NAME_GEN')(name))
        return str(name)

    def create_output_alias(self, opname, output):
        alias = str(opname) + "." + str(output)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_BUFFER_ALIAS_GEN')(output, alias))
        return str(alias)

    def create_layer_name(self, op):
        # Get the layer name for the op. Since op names are not strictly required, and output buffer names can be reused follow these steps:
        # 1. Check if name is empty, if so generate a new one.
        # 2. If a name is present and doesn't already exist in the list, use it.
        # 3. If the op name is not unique generate a new layer name based on the.
        # 4. If name is not set generate a new name based on index (optype_#)

        op_name = str(op.name)
        if op_name and op_name in self._op_names:
            op_name = self._gen_layer_alias(op)
        elif not op_name:
            op_name = self._gen_layer_name(op)

        # Save the op name
        self._op_names.append(op_name)
        return op_name

    def add_layer(self, op, layer_type):
        # Todo support multiple outputs... ops don't necessarily have names
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_NETWORK_TOPOLOGY_ADD_LAYER')(op.name, layer_type))
        # handle input/output proxy
        self._proxy.add_layer(op, self._model, layer_type)
        self._logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_NETWORK_TOPOLOGY_DONE_ADDING'))
        self._proxy.dump()

    def get_input_buffer_name(self, bufname):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME')(bufname))
        ret = self.__get_buffer_name(bufname, self._proxy.input_proxy())
        # the bufname is in the first element of the tuple
        return ret

    def get_output_buffer_name(self, bufname):
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_OUTPUT_BUFFER_NAME')(bufname))
        ret = self.__get_buffer_name(bufname, self._proxy.output_proxy())
        # the bufname is in the first element of the tuple
        return ret

    def __get_buffer_name(self, bufname, proxy_array):
        ret_bufname = str(bufname)
        if bufname in proxy_array:
            ret_bufname = proxy_array[bufname]
        # no proxy (thus no alias either), just return the same name
        self._logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_BUFFER_NAME')(bufname, ret_bufname))
        return (str(ret_bufname))

    def squash(self, op, input_name):
        """Squash an op onto one of it's inputs."""

#------------------------------------------------------------------------------
#   Pretrained Data Provider
#------------------------------------------------------------------------------
class PretrainedDataProvider(object):
    def __init__(self, init_net_path, ext_inputs):

        self.logger = logging.getLogger()
        self.weights_map = {}
        self.ext_inputs = ext_inputs
        net_params = caffe2_pb2.NetDef()

        try:
             with open(init_net_path, 'rb') as init_net_data:
                net_params.ParseFromString(init_net_data.read())
        except Exception as e:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_PARSE')(init_net_path, str(e)))
            sys.exit(1)

        # All weights, biases, etc are passed as inputs to the operation, not
        # stored internally. Parse the pretrained input data for the different
        # operators and build the operator -> tensor map
        for data_op in net_params.op:
            if str(data_op.name) != '' or len(data_op.input) > 1:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE2_DATA_SINGLE_INSTANCE_EXPECTED')(str(data_op.name), data_op.input))
            if len(data_op.output) != 1:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE2_DATA_SINGLE_INSTANCE_EXPECTED')(str(data_op.name)))

            buffer_name = str(data_op.output[0])
            if buffer_name in list(self.weights_map.keys()):
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_DUPLICATE_DATA_DETECTED')(buffer_name))

            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_WEIGHTMAP')(buffer_name))
            self.weights_map[buffer_name] = data_op.arg

    def get_weights(self, name):

        if name not in list(self.weights_map.keys()):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_WEIGHT_NAME_NOT_IN_MAP')(str(name)))

        map_entry = self.weights_map[name]

        dims = []
        for arg in map_entry:
            if str(arg.name) == 'shape':
                dims = arg.ints
        if len(dims) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_WEIGHT_SHAPE_NOT_IN_MAP')(str(name)))

        weights = []
        for arg in map_entry:
            if str(arg.name) == 'values':
                weights = numpy.array(arg.floats, dtype=numpy.float32).reshape(dims)
        if len(weights) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_WEIGHT_VALUES_NOT_IN_MAP')(str(name)))

        return weights

    def get_weight_inputs(self, op):

        param_list = []
        for idx, i in enumerate(op.input):
            if i not in self.ext_inputs:
                self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_WEIGHT_EXTERNAL')(str(i)))
                continue
            if idx == 0:
                self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_WEIGHT_DATA')(str(i)))
                continue

            # Weights, bias, and any other ext inputs
            param_list.append(str(i))
        return param_list

    def get_batch_norm_weights(self, op, eps):
        # Spatial BatchNorm:
        #
        # weights = 1 / sqrt(variance+epsilon)
        # bias = (-1 * mean) / sqrt(variance+epsilon)
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 4:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE2_SPATIAL_BATCH_NORM_PARAMS_ORDER_ERR')(str(op.output[0])))

        # Scale, running mean, running inv variance, and bias
        scale = numpy.ascontiguousarray(self.get_weights(param_in[0]), dtype=numpy.float32)
        bias = numpy.ascontiguousarray(self.get_weights(param_in[1]), dtype=numpy.float32)
        mean = numpy.ascontiguousarray(self.get_weights(param_in[2]), dtype=numpy.float32)
        variance = numpy.ascontiguousarray(self.get_weights(param_in[3]), dtype=numpy.float32)

        stddev = numpy.sqrt(variance+eps)
        c_weights = scale / stddev
        c_bias = ((-1 * scale * mean) / stddev) + bias
        return c_weights, c_bias

    def get_bbox_transform_params(self, op):
        im_info = numpy.ascontiguousarray(self.get_weights(op.input[2]), dtype=numpy.float32)
        return im_info

    def get_conv_weights(self, op):
        # weights are stored as [N,C,H,W], need [H,W,C,N]
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 1:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_CONV_LAYER_INPUT_ERR')(str(op.output[0])))

        c_weights = self.get_weights(param_in[0])
        num_output = c_weights.shape[0]

        c_weights = numpy.rollaxis(c_weights, 0, 4) # [C,H,W,N]
        c_weights = numpy.rollaxis(c_weights, 0, 3)
        c_weights = numpy.ascontiguousarray(c_weights, dtype=numpy.float32)

        if len(param_in) > 1:
            c_bias = self.get_weights(param_in[1])
        else:
            c_bias = numpy.require([0] * num_output, dtype=numpy.float32)
        return c_weights, c_bias

    def get_deconv_weights(self, op):
        # Deconv inputs are expected in the order: weights, bias
        # Weights are ordered as [C,N,H,W], need to convert to [H,W,C,N].
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 1:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_DECONV_LAYER_INPUT_ERR')(str(op.output[0])))

        c_weights = self.get_weights(param_in[0])
        num_output = c_weights.shape[1]
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_DECONV_WEIGHT_SHAPE')(str(c_weights.shape)))

        c_weights = numpy.rollaxis(c_weights, 0, 4)
        c_weights = numpy.rollaxis(c_weights, 0, 4)

        c_weights = numpy.ascontiguousarray(c_weights, dtype=numpy.float32)
        if len(param_in) > 1:
            c_bias = self.get_weights(param_in[1])
        else:
            c_bias = numpy.require([0] * num_output, dtype=numpy.float32)
        return c_weights, c_bias

    def get_fc_weights(self, op, input_depths):
        weights_list = []
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 2:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_OP_INPUT_WEIGHT_BIASES_ERR')(str(op.output[0])))
        weights = []

        weights.append(self.get_weights(param_in[0]))

        for input_depth, c_weights in zip(input_depths, weights):
            # Caffe2 often stores fc weights in 3 dims, in that case strip off the
            # first and keep only the last 2
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_FC_WEIGHT_SHAPE')(str(c_weights.shape)))
            if len(c_weights.shape) > 2:
               c_weights = c_weights.reshape(c_weights.shape[-2:])
               self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_STRIPPED_FC_SHAPE')(str(c_weights.shape)))
            output_size, input_size = c_weights.shape
            if input_depth == input_size:
                weights_list.append(c_weights)
            else: # need to re-order because activations go C,H,W -> H,W,C
                c_weights = numpy.reshape(c_weights, (output_size, input_depth, input_size//input_depth))
                c_weights = numpy.rollaxis(c_weights, 1, 3)
                c_weights = numpy.reshape(c_weights, (output_size,input_size))
                weights_list.append(c_weights)

        c_bias = self.get_weights(param_in[1])

        return weights_list, c_bias

    def get_generate_proposals_params(self, op):
        im_info = numpy.ascontiguousarray(self.get_weights(op.input[2]), dtype=numpy.float32)
        anchors = numpy.ascontiguousarray(self.get_weights(op.input[3]), dtype=numpy.float32)
        return im_info, anchors

    def get_instance_norm_weights(self, op):
        # weight = scale, which is stored in other_in as <operation_name>_s
        # bias = bias
        # expected order scale, bias
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 2:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE2_NORMALIZATION_PARAMS_ORDER_ERR')(str(op.output[0])))

        c_weights = numpy.ascontiguousarray(self.get_weights(param_in[0]), dtype=numpy.float32)
        c_bias = numpy.ascontiguousarray(self.get_weights(param_in[1]), dtype=numpy.float32)
        return c_weights, c_bias

    def get_prelu_weights(self, op):
        param_in = self.get_weight_inputs(op)
        if len(param_in) < 1:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE2_PRELU_EXPECTED_SLOPE_PARAM_ERR')(str(op.output[0])))

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_PRELU_WEIGHT')(param_in[0]))
        c_bias = self.get_weights(param_in[0])
        return [ float(f) for f in c_bias ]

    def get_reshape_shape(self, op):
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_PREDATA_RESHAPE_DIM'))
        # Reshape dims can either be an argument or second input. If more than one
        # input check for dims as an external input
        if len(op.input) > 1:
            for arg in self.weights_map[str(op.input[1])]:
                if str(arg.name) == 'shape':
                    return arg.ints
        return []

#------------------------------------------------------------------------------
#   Converter Class
#------------------------------------------------------------------------------
class Caffe2SnapDnnConverter(object):

    def __init__(self):
        self.model = modeltools.Model()

        self.op_registry_ = {}
        self.layer_id_map = {}
        self.pool_parms_map = {}
        self.global_op_info = {}
        self.logger = logging.getLogger()
        self.input_type_map = {}

        # Setup UDL function
        self._udl_factory_func = {}

        # Instansiate network topology to be used by the converter
        self._network_topology = NetworkTopology(self.model)

        # Register all supported operators
        self.register_ops()

        # Instantiate two axis trackers, one each for Caffe and SNPE
        self._caffe_axis_tracker = snpe_axis_transformer.AxisTracker("Caffe")
        self._snpe_axis_tracker = snpe_axis_transformer.AxisTracker("SNPE")

        # Instantiate one axis transformer with two axis trackers and two layered order axes
        self._axis_transformer = snpe_axis_transformer.AxisTransformer(caffe_layer_axes, self._caffe_axis_tracker,
                                                                                snpe_layer_axes, self._snpe_axis_tracker)

        self._axis_skip_layers = ['data', 'Dropout']

    def set_udls(self, obj):
        if not type (obj) is dict:
            self.logger.error ("set_udls needs to be dict")
            return False
        # Extract every udl object
        for layer_type, layer_obj in list(obj.items()):
            # Register udl function
            self._udl_factory_func[layer_type] = layer_obj.getLayerCallback()

            # Register all of its target axis orders
            input_axis_orders, output_axis_orders =  layer_obj.getAxisOrder()
            for i_axis_order, o_axis_order in zip(input_axis_orders, output_axis_orders):
                snpe_layer_axes.add_axis_order(layer_type, i_axis_order, o_axis_order)

            # Register all of its source axis orders
            src_input_axis_orders, src_output_axis_orders =  layer_obj.getSrcAxisOrder()
            for i_axis_order, o_axis_order in zip(src_input_axis_orders, src_output_axis_orders):
                caffe_layer_axes.add_axis_order(layer_type, i_axis_order, o_axis_order)

            # Print a warning if the type overrides an existing caffe2 op and then
            # add it to the layer registry
            if layer_type in self.op_registry_:
                self.logger.warn("UDL: " + layer_type + " overriding default Caffe2 op")
            self.op_registry_[layer_type] = self.add_udl_layer

        ppstr = pprint.pformat(self._udl_factory_func, indent=4)
        self.logger.debug("UDL factory funcs: " + ppstr)

        return True

    def register_ops(self):
        def skip_op(op):
            pass
        self.op_registry_["Add"] = self.add_elementwise_sum_layer
        self.op_registry_["AveragePool"] = self.add_pooling_layer
        self.op_registry_["BBoxTransform"] = self.add_bbox_transform_layer
        self.op_registry_["BoxWithNMSLimit"] = self.add_box_with_nms_limit_layer
        self.op_registry_["Concat"] = self.add_concat_layer
        self.op_registry_["ChannelShuffle"] = self.add_channel_shuffle_layer
        self.op_registry_["Conv"] = self.add_conv_layer
        self.op_registry_["ConvTranspose"] = self.add_deconv_layer
        self.op_registry_["Dropout"] = self.add_dropout_layer
        self.op_registry_["FC"] = self.add_fc_layer
        self.op_registry_["GenerateProposals"] = self.add_generate_proposals_layer
        self.op_registry_["GenerateProposalsCPP"] = self.add_generate_proposals_layer
        self.op_registry_["Flatten"] = self.add_reshape_layer
        self.op_registry_["FlattenToVec"] = self.add_reshape_layer
        self.op_registry_["ImageInputOp"] = self.setup_preprocessing
        self.op_registry_["ImplodeBatch"] = skip_op
        self.op_registry_["InstanceNorm"] = self.add_batch_norm_layer
        self.op_registry_["LRN"] = self.add_lrn_layer
        self.op_registry_["Max"] = self.add_elementwise_max_layer
        self.op_registry_["MaxPool"] = self.add_pooling_layer
        self.op_registry_["Mul"] = self.add_elementwise_product_layer
        self.op_registry_["ResizeNearest"] = self.add_scaling_layer
        self.op_registry_["PadImage"] = self.process_image_padding
        self.op_registry_["PRelu"] = self.add_prelu_layer
        self.op_registry_["Relu"] = self.add_relu_layer
        self.op_registry_["Elu"] = self.add_elu_layer
        self.op_registry_["Reshape"] = self.add_reshape_layer
        self.op_registry_["RoIAlign"] = self.add_roi_align_layer
        self.op_registry_["RoIWarp"] = self.add_roi_align_layer
        self.op_registry_["Sigmoid"] = self.add_logistic_layer
        self.op_registry_["Slice"] = self.add_slice_op
        self.op_registry_["Split"] = self.add_split_op
        self.op_registry_["Softmax"] = self.add_softmax_layer
        self.op_registry_["SpatialBN"] = self.add_batch_norm_layer
        self.op_registry_["Sum"] = self.add_elementwise_sum_layer
        self.op_registry_["Tanh"] = self.add_tanh_layer
        self.op_registry_["Tile"] = self.add_tile_layer

    def convert_op(self, op):
        if not str(op.type) in list(self.op_registry_.keys()):
            self.logger.error(code_to_message.get_error_message('ERROR_CAFFE2_SNPE_OP_SUPPORT_ERR')(str(op.type)))
            sys.exit(1)

        try:
            id_ = self.op_registry_[str(op.type)](op)
        except Exception as e:
            self.logger.error(code_to_message.get_error_message('ERROR_CAFFE2_PROCESSING_OP_ERR')(str(op.type), str(e)))
            sys.exit(1)

        return id_

    def convert(self,
                net_def_path,
                init_net_path,
                dnn_output_path,
                copyright_file,
                encoding,
                input_dims,
                enable_preprocessing=False,
                model_version=None,
                converter_command='N/A',
                reorder_list=[],
                opaque_inputs=[]):

        self.copyright_str = snpe_converter_utils.get_string_from_txtfile(copyright_file)
        self.enable_preprocessing = enable_preprocessing
        self.encoding = encoding
        self.input_dims = {}
        self.reorder_list = reorder_list
        self.net = caffe2_pb2.NetDef()
        for input_name in opaque_inputs:
            self.input_type_map[input_name] = 'opaque'

        try:
            with open(net_def_path, 'rb') as net_data:
                self.net.ParseFromString(net_data.read())
        except Exception as e:
            self.logger.error(code_to_message.get_error_message('ERROR_CAFFE2_PARSING_NETWORK_DEF')(net_def_path, str(e)))
            sys.exit(1)

        self.external_inputs  = self.net.external_input
        self.external_outputs = self.net.external_output

        # Validate the reorder
        for data in self.reorder_list:
            if data not in self.external_inputs and data not in self.external_outputs:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_NOT_INPUT_OR_OUTPUT_FOR_REORDER')(data))

        # Process the input(s) which are a list of lists with each list having 2 elements [0]=input_name and [1]=dims
        # Multi-input would look like [['data_a', '3,224,224'],['data_b', '3,50,50']]
        if len(input_dims) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_INPUT_DIMS_NOT_VALID'))

        for data_in in input_dims:
            if len(data_in) != 2:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_INPUT_DIMS_FORMAT_NOT_VALID'))
            dims = data_in[1].split(',')
            if len(dims) > 4:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_INPUT_DIMS_CHANNEL_FORMAT_NOT_VALID'))

            if data_in[0] not in list(map(str, self.net.external_input)):
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE2_DATA_NOT_AN_EXTERNAL_DATA_INPUT')(data_in[0]))

            self.input_dims[data_in[0]] = list(map(int, dims))
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_FOUND_DATA')(data_in[0], str(self.input_dims[data_in[0]])))

        # Setup the weight provider
        self.weight_provider = PretrainedDataProvider(init_net_path, self.net.external_input)

        # Process the network
        self.process_net(self.net)

        # Set the model version if present
        if model_version is not None:
            self.model.set_model_version(model_version[:64])

        # Save the dnn model
        self.model.set_converter_command(converter_command)
        self.model.set_model_copyright(self.copyright_str)
        self.model.save(dnn_output_path)

    def process_net( self, net):

        # Some sanity checks
        if len(net.op) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_NO_OPS_PRESENT_IN_CAFFE2_CONVERTER'))

        # Process the data input(s)
        if len(self.input_dims) > 0 and not self.enable_preprocessing:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESSNET_NO_PREPROC'))
            for data_name, dims in list(self.input_dims.items()):

                input_type = self.input_type_map.get(data_name,"default")
                if len(dims) == 4: # Reorder dims from NCHW to NHWC
                    data_dims = list(map(int, [dims[0], dims[2], dims[3], dims[1]]))
                else:
                    data_dims = list(map(int, dims))
                id_ = self.model.add_data_layer(data_name, data_dims, self.encoding, "bgr", input_type)

                self._axis_transformer.update_src_axis_order('data', len(data_dims), data_name, len(data_dims))
                self._axis_transformer.update_target_axis_order('data', len(data_dims), data_name, len(data_dims))
                self.save_axis_order(data_name)
                self.layer_id_map[data_name] = id_
                self.logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESSNET_ADD_LAYER')(data_name, str(data_dims)))

                # If input is listed in the reorder list set the input order.
                # All Caffe2 inputs are CHW, so if it's listed add it.
                if data_name in self.reorder_list:
                    self.logger.info(
                        code_to_message.get_progress_message('INFO_CAFFE2_SETUP_EXTERNAL_INPUT_REORDERING')(data_name))
                    # This is an external input which is really only "data" layers, no need to set an index
                    self.model.set_input_order(data_name, [0,2,3,1])

        # Preprocess the network to resolve issues like repeated identical input/output names that cause the converter to confuse
        # different inputs/outputs, connections, and dimensionality
        self.preprocess_net(self.net, self._network_topology)

        # Process the operations
        self.process_ops(net)

        # Process the network arguments
        args = self.get_args(net.arg)
        for arg in args:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESSING_ARG')(str(arg.name)))

    def process_net_names(self, net, topology):
        # Loop through all ops and fix op names and inputs and outputs such that if more than one op ouput
        # has the same name subsequent outputs and the following inputs with the same name get renamed
        output_map = {}
        for op in net.op:
            op.name = topology.create_layer_name(op)

            # Process inputs first... they need to use the current output mapping
            for index,i in enumerate(op.input):
               # If a mapping for an outuput exists and it's different remap the input to the new name
               if i in output_map and i != output_map[i]:
                   self.logger.debug(
                       code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_REMAP_INPUT')(op.name, i, output_map[i]))
                   op.input[index] = output_map[i]

            # Process outputs next, they may remap later inputs
            for index,o in enumerate(op.output):
               if o not in output_map:
                   output_map[o] = o
               else:
                   o_alias = topology.create_output_alias(op.name, o)
                   self.logger.debug(
                       code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_REMAP_OUTPUT')(op.name, o, o_alias))
                   output_map[o] = o_alias
                   op.output[index] = o_alias

        # A bonus for renaming in a preprocessing step... We can dump the caffe2 model with new naming and validate against
        # both the original and SNPE model
        #with open("./preprocessed_predict_net.pb", 'w') as f:
        #    f.write(net.SerializeToString())

    def preprocess_net(self, net, topology):
        # Process things like op/buffer naming and squashing in passes. This should keep things much cleaner
        # than trying to do all these things simultaneously throughout the code.

        # TODO cleanup/remove network topology when it's no longer truly required.
        self.process_net_names(net, topology)

        # TODO Remove/squash unneeded ops
        producer = {}
        self.skip_processing = set()
        for op in net.op:
            type_ = str(op.type)
            if type_ == 'ImplodeBatch':
                prev = producer[op.input[0]]
                self.skip_processing.add(op.name)
                if (str(prev.type) != 'RoIAlign'):
                    err = code_to_message.get_error_message('ERROR_CAFFE2_IMPLODE_BATCH_INPUT')
                    self.logger.error(err,op.name, str(prev.type))
                for arg in op.arg:
                    if str(arg.name) in ('tiled_batch_h','tiled_batch_w','pad_h','pad_w','padvalue'):
                        prev.arg.extend([arg])
                prev.output[0] = op.output[0]
                prev.output.append(op.output[1])
                producer[op.output[0]] = prev
                producer[op.output[1]] = prev

            else:
                for o in op.output:
                    producer[o] = op

    def process_ops(self, net):

        for op in net.op:
            if op.name in self.skip_processing:
                continue
            self._network_topology.add_layer(op, str(op.type))

            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESSING_OP')(str(op.type)))
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESS_INP')(self.get_layer_name(op), str(op.input)))
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESS_OUT')(self.get_layer_name(op), str(op.output)))
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESS_NUM_ARGS')(str(len(op.arg))))

            # Currently on NCHW ordering is supported
            args = self.get_args(op.arg)
            if "order" in list(args.keys()) and args["order"].decode() != "NCHW":
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_ONLY_NCHW_ORDER_SUPPORTED')(args["order"]))

            # Pre layer work - if implicit permute layer is required, it adds it
            self.do_pre_layer_work(op)

            # Convert the op to a layer and add it to the model
            id_ = self.convert_op(op)
            if id_ != -1:
                self.add_layer(op, id_)
                self.do_post_layer_work(op)

            # Redorder op output data if needed
            for index,output in enumerate(op.output):
                if output in self.reorder_list:
                    self.logger.info(
                        code_to_message.get_progress_message('INFO_CAFFE2_SETUP_EXTERNAL_OUTPUT_REORDERING')(output))
                    # Need to ensure we're setting the order for the correct output index
                    self.model.set_output_order_by_index(str(op.name), index, [0,3,1,2])

# Can't print dims for everything all the time as not all input buffers are used/registered.
# TODO cleanup
#            self.logger.debug(
#                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESS_INP_DIMS')(self.get_layer_name(op), op.type, str(self.get_input_dims(op))))
#            self.logger.debug(
#                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PROCESS_OUT_DIMS')(self.get_layer_name(op), op.type, str(self.get_output_dims(op))))

    def get_args(self, args):

        parsed_args = {}
        for arg in args:
            arg_name = str(arg.name)
            arg_data = self.get_arg(arg)
            if arg_data is None:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_CANT_PROCESS_ARGS')(arg_name))
            if arg_name in list(parsed_args.keys()):
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_DUPLICATE_ARG_FOUND')(arg_name))
            parsed_args[arg_name] = arg_data

        return parsed_args

    def get_arg(self, arg):

       self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_ARGS')(str(arg.name)))
       if arg.HasField('f'):
           return arg.f
       elif arg.HasField('i'):
           return arg.i
       elif len(arg.s) > 0:
           return arg.s
       elif len(arg.floats) > 0:
           return arg.floats
       elif len(arg.ints) > 0:
           return arg.ints
       elif len(arg.strings) > 0:
           return arg.strings

    def check_args(self, args, supported_args):
        bad_args = []
        for arg in list(args.keys()):
            if arg not in supported_args:
                bad_args.append(arg)
        if bad_args:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_UNSUPPORTED_ARGS_ERR')(str(bad_args)))

    def add_layer(self, op, id_):
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_ADDING_OP')(str(op.type)))
        self.layer_id_map[self.get_layer_name(op)] = id_
        # TODO Clean this up... looks redudant and should be removed as part of buffer cleanup
        if len(op.output) == 1:
            topbufname = str(op.output[0])
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_ADD_OP_SAME_BUFF')(str(op.output[0]), topbufname))
        else:
            for name in op.output:
                self.logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_ADD_OP_DIFF_BUFF')(str(name)))
                self.layer_id_map[str(name)] = id_

    def add_udl_layer(self, op):
        typ = str(op.type)
        name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_UDL')(typ, str(name)))

        func = self._udl_factory_func[typ]

        # FIXME should be list of lists
        inputDims = self.get_input_dims(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_UDL_INPUT_DIMS')(str(inputDims)))
        blob_output = func(op, inputDims)
        blob = blob_output.getBlob()
        # we need a list of lists.
        # i.e. a list of dimensions. each dimensions is a list
        output_dims = []
        for idx in range(len(op.output)):
            # FIXME do we need list() here?
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_UDL_OUTPUT_DIM_IDX')(str(idx)))
            dim = blob_output.getOutputDims(idx)
            assert(isinstance(dim, list))
            output_dims.append(dim)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_UDL_OUTPUT_DIMS')(str(output_dims)))
        if blob.getSize() == 0:
            raise ValueError(code_to_message.get_error_message(ERROR_CAFFE2_UNEXPECTED_ZERO_BLOB_SIZE))
        # need list(output_dims) since it is tuple, and the function expect list of int
        inputsList = self.get_input_names(op)
        outputList = self.get_output_names(op)
        # we cache typ again since  blob_output = func(layer, inputDims)
        # might change the layer type (we allow this)
        return self.model.add_user_defined_layer(name,
                                                 str(op.type),
                                                 inputsList,
                                                 outputList,
                                                 output_dims,
                                                 blob.getBlob())

    # Function to conduct preparatory work before layer is added
    # It includes adding implicit permute layer.
    def do_pre_layer_work(self, op):

        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PRE_WORK_OP')(self.get_layer_name(op)))

        if str(op.type) in self._axis_skip_layers:
            return

        # Get both caffe and snpe buffer names and its dims.
        target_input_names = self.get_input_names(op)

        src_input_names = self.get_input_names(op)
        input_dims = self.get_input_dims(op)
        assert( len(src_input_names) == len(input_dims) )
        assert len(src_input_names)

        # For each of its bottom buffer, get its permute order
        for idx in range(len(src_input_names)):

            # No permute order = [0,1,2] for 3d input or [0,1] for 2d or [0,1,2,3] for 4d
            no_permute_order = numpy.arange(len(input_dims[idx])).tolist()
            permute_order = self._axis_transformer.get_permute_order( str(op.type), len(input_dims[idx]),
                                                                      src_input_names[idx], target_input_names[idx] )
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_NO_PERM_ORDER')(str(no_permute_order), str(permute_order)))

            # If permute is requited
            if len(permute_order) and permute_order != list(range(len(permute_order))):
                # Note: input and output name are the same as to mimic in-place buffer
                layer_name = self.get_layer_name(op)
                self.logger.debug(
                    code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_IMPLIC_PERM_LAYER')(str(permute_order), layer_name))
                self.add_implicit_permute_layer( layer_name, permute_order, src_input_names[idx] )

    # Function to conduct finishing work after layer is added
    def do_post_layer_work(self, op):

        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_POST_WORK_LAYER')(layer_name))

        if str(op.type) in self._axis_skip_layers:
            return

        target_output_names = self.get_output_names(op)

        # Currently assumes single output only. Caffe2 often has multiple outputs which
        # aren't actively used by subsequent layers/ops, which can be ignored. When an
        # op which supports multiple "real" outputs is added we'll need to update this
        # with the required exceptions. TODO FIXME
        output_dims = []
        src_output_names = []
        src_output_names.append(self.get_output_name(op))
        output_dims.append(self.get_output_dim(op))

        # With the strict assumption that in MIMO or MISO usecases, the axis order of all inputs
        # are the same, the input_rank of only one input is passed.
        input_rank = len(self.get_input_dim(op))
        target_singleout_input_name = self.get_input_name(op)
        src_singleout_input_name = str(op.input[0])

        # For each of its top buffer, update its Caffe and SNPE axis order
        for idx in range(len(src_output_names)):
            # TBD: +1 might change for reshape/lstm where batch dimension is not disregard.
            # TBD: Figure out Caffe py api that returns per-layer input shape . That way we could partially
            #      get away with +1
            self._axis_transformer.update_src_axis_order(str(op.type), len(output_dims[idx]), src_output_names[idx],
                                                         input_rank, src_singleout_input_name)
            self._axis_transformer.update_target_axis_order(str(op.type), len(output_dims[idx]), target_output_names[idx],
                                                            input_rank, target_singleout_input_name)
            self.save_axis_order(target_output_names[idx])

    def add_batch_norm_layer(self, op):
        name = self.get_layer_name(op)
        op_type = str(op.type)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_OP_TO_BATCH_LAYER')(op_type, name))

        weights = []
        bias = []
        compute_stats = False
        if op_type == 'InstanceNorm':
            weights, bias = self.weight_provider.get_instance_norm_weights(op)
            compute_stats = True
        else:
            eps = 1e-5
            args = self.get_args(op.arg)
            if 'epsilon' in list(args.keys()):
                eps = args['epsilon']
            weights, bias = self.weight_provider.get_batch_norm_weights(op, eps)

        # from the batch_norm layer we get weights W1 and bias B1:
        # y  = W1.x + B1
        # from the scaling layer (if present), we get weights W2 and bias B2:
        # y' = W2.y + B2 = W2(W1.x + B1) + B2 =
        #                = (W2.W1)x + (W2.B1 + B2)
        #
        # For instance norm we get 1D scale and 1D bias tensor of size C
        return self.model.add_batchnorm_layer(name, weights, bias,
                                              compute_statistics = compute_stats,
                                              use_mu_sigma = True,
                                              across_spatial = True,
                                              input_name = str(self.get_input_name(op)),
                                              output_name = str(self.get_output_name(op)))
        self.add_layer(layer_batch_norm, id_)

    def add_bbox_transform_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        input_names = self.get_input_names(op)
        if len(input_names) != 2:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_INPUTS')
            raise ValueError(err("Generate Proposals",layer_name,2,len(input_names)))

        output_names = self.get_output_names(op)
        if len(output_names) != 1:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_OUTPUTS')
            raise ValueError(err("BboxTransform",layer_name,1,len(output_names)))

        im_info = self.weight_provider.get_bbox_transform_params(op)

        args = self.get_args(op.arg)
        if not 'weights' in args:
            raise ValueError("Missing required argument 'weights'")

        weights = [weight for weight in args.get('weights')] #, dtype=numpy.float32)
        apply_scale = bool(args.get('apply_scale',True))
        correct_transform_coords = bool(args.get('correct_transform_coords',False))

        return self.model.add_bbox_transform_layer(layer_name,
                                                   weights,
                                                   im_info,
                                                   apply_scale,
                                                   correct_transform_coords,
                                                   input_names[0],
                                                   input_names[1],
                                                   output_names[0])

    def add_box_with_nms_limit_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        input_names = self.get_input_names(op)
        if len(input_names) < 2:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_INPUTS')
            raise ValueError(err("BoxWithNmsLimit",layer_name,2,len(input_names)))

        output_names = self.get_output_names(op)
        if len(output_names) < 3:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_OUTPUTS')
            raise ValueError(err("BoxWithNmsLimit",layer_name,3,len(output_names)))
        args = self.get_args(op.arg)
        score_thresh = float(args.get('score_thresh', 0.05))
        nms_thresh = float(args.get('nms', 0.3))
        detections_per_im = int(args.get('detections_per_im', 100))
        soft_nms_enabled = bool(args.get('soft_nms_enabled', False))
        soft_nms_method = str(args.get('soft_nms_method', 'linear'))
        soft_nms_sigma = float(args.get('soft_nms_sigma', 0.5))
        soft_nms_min_score_thresh = float(args.get('soft_nms_min_score_thres', 0.001))

        supported_nms_methods = ['linear','gaussian']
        if not soft_nms_method in supported_nms_methods:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE2_INVALID_NMS_METHOD')(soft_nms_method, supported_nms_methods))

        return self.model.add_box_with_nms_limit_layer(layer_name,
                                                       score_thresh,
                                                       nms_thresh,
                                                       detections_per_im,
                                                       soft_nms_enabled,
                                                       soft_nms_method,
                                                       soft_nms_sigma,
                                                       soft_nms_min_score_thresh,
                                                       input_names,
                                                       output_names)

    def add_channel_shuffle_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CHANNEL_SHUFFLE_OP')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        groups = 1
        if 'group' in list(args.keys()):
            snpe_groups = args['group']
        else:
           raise ValueError(
               code_to_message.get_error_message('ERROR_CAFFE2_CHANNEL_SHUFFLE_LAYER_MISSING_GROUPS_ARG')(str(op.type)))

        return self.model.add_channel_shuffle_layer( name=layer_name,
                                                   groups = snpe_groups,
                                                   shuffle_type = modeltools.CHANNEL_SHUFFLE_GROUPED,
                                                   input_name = str(self.get_input_name(op)),
                                                   output_name = str(self.get_output_name(op)))

    def add_concat_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        caffe2_axis = args['axis'] if 'axis' in list(args.keys()) else 1
        snpe_axis = 2
        input_dim = self.get_input_dim(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CONCAT_DIM')(layer_name, str(input_dim)))
        if len(input_dim) == 1:
            snpe_axis = 0
        else:
            target_inames = self.get_input_names(op)
            snpe_axes = []
            for idx in range(len(target_inames)):
                snpe_axes.append(self._axis_transformer.get_target_axis(target_inames[idx], caffe2_axis, target_inames[idx]))

            # Sanity Check: for all buffers, the same axis is returned.
            if not all(x == snpe_axes[0] for x in snpe_axes):
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_CONCAT_LAYER_AXIS_NOT_SUPPORTED')(layer.name, caffe2_axis))
            snpe_axis = snpe_axes[0]

        return self.model.add_concatenation_layer( name=layer_name,
                                                   input_names=self.get_input_names(op),
                                                   output_name = str(self.get_output_name(op)),
                                                   axis = snpe_axis)

    def add_conv_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CONV_OP')(str(op.type), layer_name))

        args = self.get_args(op.arg)

        pad_mode = modeltools.PADDING_ZERO

        # TODO Padding should really be it's own op/layer. Set name to the current input name
        in_name = str(op.input[0])
        pad_args = {}
        if in_name in self.global_op_info and str(self.global_op_info[in_name].type) == "PadImage":
            pad_op = self.global_op_info[in_name]
            pad_args = self.get_args(pad_op.arg)
            # If spatial reflection padding is to be used set it
            if pad_args['mode'].decode() == "reflect":
                self.logger.warn(code_to_message.get_warning_message('WARNING_CAFFE2_IGNORE_LOCAL_PADDING'))
                pad_mode = modeltools.PADDING_REFLECT
            else:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE2_UNSUPPORTED_PADDING_OP_FOR_CONV_LAYER')(in_name, str(pad_op.type)))

        # Ensure the input is set to the "real" SNPE input name
        in_name = str(self.get_input_name(op))

        groups = 1
        if "group" in list(args.keys()):
            groups = args['group']

        dilation_x, dilation_y = 1, 1
        if "dilation" in list(args.keys()):
            dilation_x = args["dilation"]
            dilation_y = args["dilation"]
        elif "dilation_w" in list(args.keys()) and "dilation_h" in list(args.keys()):
            dilation_x = args["dilation_w"]
            dilation_y = args["dilation_h"]
        elif "dilations" in list(args.keys()):
            dilations = args["dilations"]
            dilation_x = dilations[0]
            dilation_y = dilations[1]

        kps_params = self.get_kps_op_params(op, args, pad_args=pad_args)
        op_input_name = self.get_input_name(op)
        input_size = self.model.get_buffer_dims(str(in_name))

        c_weights, c_bias = self.weight_provider.get_conv_weights(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CONV_WEIGHT_DIMS')(c_weights.shape))
        if len(c_bias) > 0:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CONV_BIAS_DIMS')(c_bias.shape))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_CONV_INP_DIMS')(input_size))

        # Don't support LegacyPadding::VALID(1) and  LegacyPadding::SAME(2) yet
        if 'legacy_pad' in list(args.keys()) and ( args['legacy_pad'] == 1 or args['legacy_pad'] == 2 ):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_CONV_LEGACY_AND_DEFAULT_PADDING_SUPPORTED')(layer_name))

        # Caffe2 defaults to LegacyPadding::NOTSET (0)
        # It uses floor-based padding style
        padding_style = modeltools.PADDING_SIZE_EXPLICIT_FLOOR

        if 'legacy_pad' in list(args.keys()) and args['legacy_pad'] == 3:
            padding_style = modeltools.PADDING_SIZE_EXPLICIT

        return self.model.add_conv_layer( layer_name,
                                          weights = c_weights,
                                          bias = c_bias,
                                          padx = kps_params.padx,
                                          pady = kps_params.pady,
                                          padding_mode = pad_mode,
                                          padding_size_strategy = padding_style,
                                          stridex = kps_params.stridex,
                                          stridey = kps_params.stridey,
                                          dilationx = dilation_x,
                                          dilationy = dilation_y,
                                          input_name = in_name,
                                          output_name = str(self.get_output_name(op)),
                                          groups=groups)

    def add_deconv_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        args = self.get_args(op.arg)

        kps_params = self.get_kps_op_params(op, args)

        # Is this supported?
        groups = 1
        if "group" in list(args.keys()):
            groups = args['group']

        c_weights, c_bias = self.weight_provider.get_deconv_weights(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_DECONV_WEIGHT_DIM')(c_weights.shape))

        return self.model.add_deconvolution_layer( name=layer_name,
                                                   weights = c_weights,
                                                   bias = c_bias,
                                                   stride = kps_params.stridex,
                                                   padding_size_strategy= modeltools.PADDING_SIZE_EXPLICIT,
                                                   padx=kps_params.padx,
                                                   pady=kps_params.pady,
                                                   input_name = str(self.get_input_name(op)),
                                                   output_name = str(self.get_output_name(op)),
                                                   output_width=-1,
                                                   output_height=-1,
                                                   groups=groups)

    def add_dropout_layer(self, op):
        name = self.get_layer_name(op)

        # In 'test mode' caffe2 dropout is a no-op, so it should be skipped
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_SKIP_DROPOUT_LAYER')(name))
        self._network_topology.install_buffer_proxy(str(op.output[0]), str(op.input[0]))
        return -1

    def add_elementwise_max_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        input_names = self.get_input_names(op)
        output_name = self.get_output_name(op)
        return self.model.add_elementwise_max_layer( layer_name,
                                                     input_names,
                                                     output_name)

    def add_elementwise_product_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        supported_args = []
        args = self.get_args(op.arg)
        self.check_args(args, supported_args)
        input_names = self.get_input_names(op)
        output_name = self.get_output_name(op)
        return self.model.add_elementwise_product_layer( layer_name,
                                                         input_names,
                                                         output_name)

    def add_elementwise_sum_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        supported_args = []
        args = self.get_args(op.arg)
        self.check_args(args, supported_args)
        input_names = self.get_input_names(op)
        output_name = self.get_output_name(op)

        # Currently we only support adding inputs with the same shape/size
        in_dims =  self.get_input_dims(op)
        for index, dims in enumerate(in_dims):
            if in_dims[0] != dims:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE2_ADD_ONLY_SAME_INPUT_SHAPES_SUPPORTED_ERR')(layer_name))

        coeffs = []
        coeffs.extend( [1.0 for i in range(len(input_names))] )
        return self.model.add_elementwise_sum_layer( layer_name,
                                                     coeffs,
                                                     input_names,
                                                     output_name)

    def add_fc_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        c_input_names = self.get_input_names(op)
        input_depths = [ self.model.get_buffer_dims(name)[-1] for name in c_input_names ]

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_FC_INP')(str(op.input)))

        # Get the weights and biases
        c_weights_list, c_bias = self.weight_provider.get_fc_weights(op, input_depths)

        for weights in c_weights_list:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_FC_WEIGHTS')(str(weights.shape)))

        return self.model.add_fc_layer( name = layer_name,
                                        weights_list = c_weights_list,
                                        bias = c_bias,
                                        input_names = c_input_names,
                                        output_name = str(self.get_output_name(op)))


    def add_generate_proposals_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        input_names = self.get_input_names(op)
        if len(input_names) != 2:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_INPUTS')
            raise ValueError(err("Generate Proposals",layer_name,2,len(input_names)))

        output_names = self.get_output_names(op)
        if len(output_names) != 2:
            err = code_to_message.get_error_message('ERROR_CAFFE2_WRONG_NUMBER_OF_OUTPUTS')
            raise ValueError(err("Generate Proposals",layer_name,2,len(output_names)))

        args = self.get_args(op.arg)
        im_info, anchors = self.weight_provider.get_generate_proposals_params(op)
        def require_attr(attr):
            if not attr in args:
                raise ValueError("Missing required argument %s" % attr)
        require_attr('spatial_scale')
        require_attr('pre_nms_topN')
        require_attr('post_nms_topN')
        require_attr('nms_thresh')
        require_attr('min_size')
        correct_transform_coords = bool(args.get('correct_transform_coords',False))
        return self.model.add_generate_proposals_layer(layer_name,
                                                       args['spatial_scale'],
                                                       args['pre_nms_topN'],
                                                       args['post_nms_topN'],
                                                       args['nms_thresh'],
                                                       args['min_size'],
                                                       correct_transform_coords,
                                                       anchors,
                                                       im_info,
                                                       input_names[0],
                                                       input_names[1],
                                                       output_names[0],
                                                       output_names[1])

    def add_logistic_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        return self.model.add_neuron_layer( name = (layer_name),
                                            func = modeltools.NEURON_LOGISTIC,
                                            input_name = str(self.get_input_name(op)),
                                            output_name = str(self.get_output_name(op)),
                                            a = 1.0)

    def add_pooling_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        c_pool_type = modeltools.POOL_MAX
        if str(op.type) == 'AveragePool':
            c_pool_type = modeltools.POOL_AVG

        args = self.get_args(op.arg)

        groups = 1
        if "group" in list(args.keys()):
            groups = args['group']

        kps = self.get_kps_op_params(op, args)

        # Don't support LegacyPadding::VALID(1) and  LegacyPadding::SAME(2) yet
        if 'legacy_pad' in list(args.keys()) and ( args['legacy_pad'] == 1 or args['legacy_pad'] == 2 ):
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE2_POOLING_LEGACY_AND_DEFAULT_PADDING_SUPPORTED')(layer_name))

        # Caffe2 defaults to LegacyPadding::NOTSET (0)
        # It uses floor-based padding style
        padding_style = kps.pad_type

        if 'legacy_pad' in list(args.keys()) and args['legacy_pad'] == 3:
            padding_style = modeltools.PADDING_SIZE_EXPLICIT

        kx = kps.kx
        ky = kps.ky
        stridex = kps.stridex
        stridey = kps.stridey
        padx = kps.padx
        pady = kps.pady

        include_padding = True
        input_dim = self.get_input_dim(op)
        if 'global_pooling' in list(args.keys()):
            ky = input_dim[1]
            kx = input_dim[2]
            stridex, stridey = 1, 1
            padx, pady = 0, 0
            include_padding = False

        return self.model.add_pooling_layer( layer_name,
                                             c_pool_type,
                                             kx,
                                             ky,
                                             stridex,
                                             stridey,
                                             padx,
                                             pady,
                                             padding_style,
                                             str(self.get_input_name(op)),
                                             str(self.get_output_name(op)),
                                             include_padding)
        output_dim = self.model.get_buffer_dims(str(op.output[0]))
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_POOL_OUTPUT')(str(tuple(output_dim))))

    def add_implicit_permute_layer(self, layer_name, permute_order, layer_input_name):

        # Generate unique implicit layer name by combining the layer name, "permute" and input buffer name
        # Just layer name and "permute" is not sufficient for multiple inputs.
        implicit_permute_layer_name = layer_input_name + "_permute"
        implicit_permute_output_name = layer_input_name + ".permute." + layer_name

        # The top and bottom name of the actual layer for which the implicit permute is done, is chosen as input
        # and output for the implicit permute layer. It relies on networktopology+buffer proxy to generate correct alias
        # for the actual layer as it's been doing for in-place buffer.
        # In other words, the implicit layer is added whose buffer names are chosen as in-place buffer names
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_ADD_IMPL_LAYER')(implicit_permute_layer_name, layer_name))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PERM_ORDER')(str(permute_order)))

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

    def add_implicit_permute_layer(self, layer_name, permute_order, layer_input_name):

        # Generate unique implicit layer name by combining the layer name, "permute" and input buffer name
        # Just layer name and "permute" is not sufficient for multiple inputs.
        implicit_permute_layer_name = layer_input_name + "_permute"
        implicit_permute_output_name = layer_input_name + ".permute." + layer_name

        # The top and bottom name of the actual layer for which the implicit permute is done, is chosen as input
        # and output for the implicit permute layer. It relies on networktopology+buffer proxy to generate correct alias
        # for the actual layer as it's been doing for in-place buffer.
        # In other words, the implicit layer is added whose buffer names are chosen as in-place buffer names
        self.logger.debug("Adding implicit permute layer: " + implicit_permute_layer_name + " for the layer name: " + layer_name)
        self.logger.debug("Permute order : " + str(permute_order))

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

    def add_prelu_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        bias = self.weight_provider.get_prelu_weights(op)
        if len(bias) == 1:
            # We don't explicitly support channel sharing so we must replicate the value across all channels
            input_dims = self.get_input_dims(op)
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_RESIZE_BIAS')(str(input_dims[0][3])))
            for i in range(input_dims[0][3]-1):
                bias.append(bias[0])
        return self.model.add_prelu_layer( name = (layer_name),
                                           coeff = bias,
                                           input_name = str(self.get_input_name(op)),
                                           output_name = str(self.get_output_name(op)))

    def add_lrn_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)

        if not all (k in list(args.keys()) for k in ("size","alpha","beta","bias")):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_LRN_ARG_MISSING'))

        return self.model.add_cmrn_layer( name = layer_name,
                                          window_size = args['size'],
                                          alpha = float(args['alpha']/args['size']),
                                          beta = args['beta'],
                                          k = float(args['bias']),
                                          input_name = str(self.get_input_name(op)),
                                          output_name = str(self.get_output_name(op)))

    def add_relu_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        return self.model.add_neuron_layer( name = (layer_name),
                                            func = modeltools.NEURON_RELU,
                                            input_name = str(self.get_input_name(op)),
                                            output_name = str(self.get_output_name(op)))

    def add_elu_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))
        return self.model.add_neuron_layer(name=layer_name,
                                           func=modeltools.NEURON_ELU,
                                           input_name=str(self.get_input_name(op)),
                                           output_name=str(self.get_output_name(op)),
                                           a=1.0)

    def add_reshape_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)

        # There are 2 different Caffe2 operations that are mapped to the SNPE Reshape layer.
        #  - For "Reshape", the "shape" input, or argument if the input is unspecified, indicates the
        #    new shape. 0 indicates an unchanged dimension (to be copied from the corresponding input dimension,
        #    and -1 indicates all remaining dimensionality to be folded into this dimension.
        #  - For "Flatten", all dimensions after axis 0, starting at axis 1, are folded into axis 1.
        #  - For "FlattenToVec", all dimensions are folded into axis 0.
        bufname = self._network_topology.get_output_buffer_name(self.get_input_name(op))

        input_dims = list(map(int, self.model.get_buffer_dims(bufname)))

        typ = str(op.type)
        output_dims = []

        if typ == "Reshape":
           input_size = reduce(int.__mul__, input_dims)
           # Get the reshape shape from the input params, if it's not present it must be
           # in the op arguments
           output_dims = self.weight_provider.get_reshape_shape(op)
           if len(output_dims) == 0:
               if not 'shape' in args:
                   raise ValueError(
                       code_to_message.get_error_message('ERROR_CAFFE2_RESHAPE_OP_NO_INPUT_OR_ARG_SHAPE')(layer_name))
               self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_RESHAPE_DIMS'))
               for i in args['shape']:
                   output_dims.append(int(i))

           # Caffe2 doesn't have configurable axes, only dimensions
           axis = 0
           num_axes = len(input_dims)

           # replace any 0 in the output_dims with the corresponding dimension in the input_dims.
           output_dims = list(map(lambda x: input_dims[x+axis] if output_dims[x]==0 else output_dims[x], range(len(output_dims))))
           # prefix/postfix
           output_dims = input_dims[:axis] + output_dims + input_dims[axis+num_axes:]
           # replace -1 in the output by the remainder of the inputs
           remainder_index = [i for i, j in enumerate(output_dims) if j==-1]
           if len(remainder_index)==1:
               output_size = -1*reduce(int.__mul__, output_dims) # multiply by -1 to make this positive
               output_dims[remainder_index[0]] = input_size / output_size

        elif typ == "Flatten":
           axis = 1
           end_axis = len(input_dims) - 1
           output_dims = [ reduce(int.__mul__, input_dims[axis:end_axis+1]) ]
           output_dims = input_dims[:axis] + output_dims + input_dims[end_axis+1:]

        elif typ == "FlattenToVec":
           axis = 0
           end_axis = len(input_dims) - 1
           output_dims = [ reduce(int.__mul__, input_dims[axis:end_axis+1]) ]
           output_dims = input_dims[:axis] + output_dims + input_dims[end_axis+1:]

        else:
           raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_INVALID_OP_TYPE')(str(op.type)))

        return self.model.add_reshape_layer( name = layer_name,
                                             output_dimensions = output_dims,
                                             input_name = str(self.get_input_name(op)),
                                             output_name = str(self.get_output_name(op)))

    def add_roi_align_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        input_names = self.get_input_names(op)
        output_name = self.get_output_name(op)

        args = self.get_args(op.arg)
        def require_attr(attr):
            if not attr in args:
                raise ValueError("Missing required argument %s" % attr)

        require_attr('spatial_scale')
        require_attr('pooled_h')
        require_attr('pooled_w')
        require_attr('sampling_ratio')
        if 'tiled_batch_h' in args:
            require_attr('tiled_batch_w')
            require_attr('pad_h')
            require_attr('pad_w')
            shape_name = str(op.output[1])
        else:
            shape_name = ''
        return self.model.add_roialign_layer(layer_name,
                                             args['spatial_scale'],
                                             args['pooled_h'],
                                             args['pooled_w'],
                                             args['sampling_ratio'],
                                             input_names[0],
                                             input_names[1],
                                             output_name,
                                             shape_name,
                                             args.get('tiled_batch_h',-1),
                                             args.get('tiled_batch_w',-1),
                                             args.get('pad_h',-1),
                                             args.get('pad_w',-1),
                                             args.get('padvalue',0.0))

    def add_scaling_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        input_dim = self.get_input_dim(op)

        # scale factors are set to 1.0 if not available for dimensional computation, but
        # then reset to 0.0 so they can be passed in to the model tools in such a way that
        # the factors are not saved to the DLC.
        width_scale = 1.0
        if "width_scale" in list(args.keys()):
            width_scale = args['width_scale']

        output_width = 0
        if len(input_dim) == 4: #/w batch
            output_width  = int(math.floor(float(input_dim[2])*width_scale))
        else: #w/o batch
            output_width = int(math.floor(float(input_dim[1])*width_scale))

        if "width_scale" not in list(args.keys()):
            width_scale = 0.0
        height_scale = 1.0
        if "height_scale" in list(args.keys()):
            height_scale = args['height_scale']

        output_height = 0
        if len(output_dim) == 4: #/w batch
            output_height = int(math.floor(float(input_dim[1])*height_scale))
        else: #w/o batch
            output_height = int(math.floor(float(input_dim[0])*height_scale))

        if "height_scale" not in list(args.keys()):
            height_scale = 0.0

        if len(input_dim) == 4:  #/w batch
            output_dim = [input_dim[0], output_height, output_width, input_dim[3]]
        else:
            output_dim = [output_height, output_width, input_dim[2]]

        return self.model.add_scaling_layer( name = layer_name,
                                             output_dimensions = output_dim,
                                             pad_value = 0.0,
                                             maintain_aspect_ratio = False,
                                             resize_mode = modeltools.RESIZE_NEAREST_NEIGHBOR,
                                             scale_height = height_scale,
                                             scale_width = width_scale,
                                             input_name = str(self.get_input_name(op)),
                                             output_name = str(self.get_output_name(op)),
                                             align_corners = False)

    def add_slice_op(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        if not all (k in list(args.keys()) for k in ("start", "end")):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_SLICE_OP_INDICIES_MISSING'))

        start = args['start']
        end = args ['end']
        input_dim = self.model.get_buffer_dims(str(self.get_input_name(layer)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_SLICEOP_INP')(str(input_dim)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_SLICEOP_START')(str(start)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_SLICEOP_END')(str(end)))

        if len(start) != len(input_dim) or len(start) != len(end):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_SLICE_OP_INPUT_DIM_MISMATCH'))

        if len(start) != 3 and len(start) != 4:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_SLICE_ONLY_SUPPORTED_FOR_3AND4_DIMS_DATA'))

        # Get the output dim and offset. Note that only end indices are non-inclusive
        def get_dim_off(dim, start, end):

            offset = start
            output_dim = dim
            real_end = end
            if offset < 0:
                offset += dim
            if end < 0:
                real_end = dim + end + 1
            if offset > real_end:
                output_dim = dim - offset + real_end
            else:
                output_dim = real_end - offset

            return output_dim, offset

        # Get the output dims in HWC from CHW
        offsets = [0] * len(input_dim)
        output_dim = [0] * len(input_dim)
        if len(input_dim) == 4:
            output_dim[0],offsets[0] = get_dim_off(input_dim[0], start[0], end[0])
            output_dim[1],offsets[1] = get_dim_off(input_dim[2], start[2], end[2])
            output_dim[2],offsets[2] = get_dim_off(input_dim[3], start[3], end[3])
            output_dim[3],offsets[3] = get_dim_off(input_dim[1], start[1], end[1])
        else:
            output_dim[0], offsets[0] = get_dim_off(input_dim[1], start[1], end[1])
            output_dim[1], offsets[1] = get_dim_off(input_dim[2], start[2], end[2])
            output_dim[2], offsets[2] = get_dim_off(input_dim[0], start[0], end[0])

        return self.model.add_crop_layer(layer_name,
                                         offsets,
                                         output_dim,
                                         str(self.get_input_name(op)),
                                         str(self.get_output_name(op)))

    def add_split_op(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        input_dim = self.model.get_buffer_dims(str(self.get_input_name(op)))

        if len(op.input) > 1:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_AXIS_ORDER_SPLIT_ARGS_ONLY_SUPPORTED'))

        args = self.get_args(op.arg)

        # By default, axis is (channel) 1 from NCHW
        axis = 1
        if 'axis' in args:
            if args['axis'] < 0:
                axis = len(input_dim) + args['axis']
            else:
                axis = args['axis']
        elif 'order' in args:
            if args['order'].decode() == 'NHWC':
                axis = 3

        # Add the slice points if they exist
        slice_points = []
        if 'split' in args:
            slice_points = list(map(int, args['split']))

        return self.model.add_slice_layer( name = layer_name,
                                           input_name = str(self.get_input_name(op)),
                                           axis = self._axis_transformer.get_target_axis(self.get_input_name(op), axis, self.get_input_name(op)),
                                           slice_points = slice_points,
                                           output_names = self.get_output_names(op) )

    def add_softmax_layer(self, op):
        return self.model.add_softmax_layer( name = self.get_layer_name(op),
                                             input_name = str(self.get_input_name(op)),
                                             output_name = str(self.get_output_name(op)))

    def add_tanh_layer(self, op):
        return self.model.add_neuron_layer( name= (self.get_layer_name(op)),
                                            func = modeltools.NEURON_TANH,
                                            input_name = str(self.get_input_name(op)),
                                            output_name = str(self.get_output_name(op)),
                                            a=1.0,
                                            b=1.0 )
    def add_tile_layer(self, op):
        layer_name = self.get_layer_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_LAYER')(str(op.type), layer_name))

        args = self.get_args(op.arg)
        inputs = self.get_input_names(op)

        # Args can be specified as inputs or args, inputs take precedence. Support this?
        if len(inputs) == 1:
            caffe2_axis = args['axis'] if 'axis' in list(args.keys()) else 0
            caffe2_tiles = args['tiles'] if 'tiles' in list(args.keys()) else 1
        else:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_TILE_INPUTS_NOT_SUPPORTED'))

        input_dim = self.get_input_dim(op)
        if len(input_dim) == 1:
            snpe_axis = 0
        else:
            snpe_axis = self._axis_transformer.get_target_axis(inputs[0], caffe2_axis, inputs[0])

        # The number of tiles is equal to the number of inputs we must have
        tile_inputs = []
        for i in range(caffe2_tiles):
            tile_inputs.append(self.get_input_name(op))
        return self.model.add_concatenation_layer( name=layer_name,
                                                   input_names=tile_inputs,
                                                   output_name = str(self.get_output_name(op)),
                                                   axis = snpe_axis)

    def strip_input_buffers(self, op, ext_list):
        bufs = []
        for b in op.input:
            for ext in ext_list:
                if b.endswith(ext):
                    bufs.append(b)

        for b in bufs:
            op.input.remove(b)

    def get_input_id(self, op):
        if len(op.input) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_GET_INPUT_ID_INVALID_INPUT'))

        input_name = self.get_input_name(op)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_INPUT_ID')(self.get_layer_name(op), input_name))
        return self.layer_id_map[input_name]

    def get_input_name(self, op):
        input_names = self.get_input_names(op)
        input_name = input_names[0]
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_INP_NAME')(self.get_layer_name(op), str(op.input[0]), input_name))
        return input_name

    def get_input_names(self, op):
        # Retrieve all the 'real' inputs to the SNPE layer
        names = []
        for name in op.input:

            # Ignore weight and bias and other 'external inputs'. In SNPE these aren't actually inputs
            if name not in self.input_dims and name in self.external_inputs:
                continue

            names.append(self._network_topology.get_input_buffer_name(str(name)))
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_INP_NAMES')(str(name)))
        return names

    def get_input_id_list(self, op):
        ret = []
        for name in op.input:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_INP_ID_LIST')(self.get_layer_name(op), str(name)))
            ret.append(self.layer_id_map[str(name)])
        return ret

    def get_layer_name(self, op):
        layername = str(op.name)
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_LAYER_NAME')(layername))
        return layername

    def get_output_names(self, op):
        ret = []
        for name in op.output:
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_OUT_NAMES')(self.get_layer_name(op), str(name)))
            bufname = self._network_topology.get_output_buffer_name(str(name))
            ret.append(bufname)
        return ret

    def get_output_name(self, op):
        if len(op.output) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_GET_INPUT_ID_INVALID_OUTPUT'))

        output_name = bufname = self._network_topology.get_output_buffer_name(str(op.output[0]))
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_OUT_NAME')(str(op.output[0]), output_name))
        return output_name

    def get_input_dim(self, op):
        return self.model.get_buffer_dims(str(self.get_input_name(op)))

    def get_input_dims(self, op):
        ret = []
        bufs = self.get_input_names(op)
        for inp in bufs:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_INP_DIMS')(inp))
            dim = self.model.get_buffer_dims(inp)
            # dim has to be a list
            ret.append(list(dim))
        return ret

    def get_output_dim(self, op):
        return self.model.get_buffer_dims(str(self.get_output_name(op)))

    def get_output_dims(self, op):
        ret = []
        bufs = self.get_output_names(op)
        for out in bufs:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_OUT_DIMS')(out))
            dim = self.model.get_buffer_dims(out)
            # dim has to be a list
            ret.append(list(dim))
        return ret

    def process_image_padding(self, op):
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_PADDING_OP'))
        out_name = self.get_output_name(op)

        args = self.get_args(op.arg)
        if 'mode' not in args or args['mode'].decode() != "reflect":
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_REFLECT_PAD_MODE_ONLY_SUPPORTED')(args['mode']))

        # SNPE doesnt have a padding 'op' but does so as part of other layers (conv/pooling/etc).
        # Cache the input name and args for this output so we can refer to the info later since we
        # are essentially skipping over this layer
        self._network_topology.install_buffer_proxy(str(op.output[0]), str(op.input[0]))
        self.global_op_info[out_name] = op
        return -1

    def get_kps_op_params(self, op, args, pad_args = {}):
        # Common helper to get kernel, padding, and stide parameters

        parmstype = collections.namedtuple("KPSParams",
                                           ["padx", "pady", "stridex", "stridey", "kx", "ky", "pad_type"])

        pad = args['pad'] if 'pad' in list(args.keys()) else 0
        stride = args['stride'] if 'stride' in list(args.keys()) else 1
        kernel = args['kernel'] if 'kernel' in list(args.keys()) else 0

        def get_pad(args):
            padx, pady = 0,0
            pad_type = modeltools.PADDING_SIZE_EXPLICIT_FLOOR
            if "pad_t" in list(args.keys()) and "pad_b" in list(args.keys()):
                if(args['pad_t'] != args['pad_b']):
                    pad_type = modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC
                pady = args['pad_b']
            else:
                if "pad_t" in list(args.keys()):
                    pady = args['pad_t']
                if "pad_b" in list(args.keys()):
                    pady = args['pad_b']
            if "pad_l" in list(args.keys()) and "pad_r" in list(args.keys()):
                if(args['pad_l'] != args['pad_r']):
                    pad_type = modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC
                padx = args['pad_r']
            else:
                if "pad_l" in list(args.keys()):
                    padx = args['pad_l']
                if "pad_r" in list(args.keys()):
                    padx = args['pad_r']
            if "pad" in list(args.keys()):
                padx, pady = pad,pad
            if "pads" in list(args.keys()):
                pads = args["pads"]
                padx, pady = pads[0], pads[1]
            return padx, pady, pad_type

        # Pad args are processed later during conv or other layers
        pad_args = pad_args if len(pad_args) > 0 else args
        padx, pady, pad_type = get_pad(pad_args)

        stridex, stridey = 1, 1
        if 'stride_h' in list(args.keys()) or 'stride_w' in list(args.keys()):
            stridex = args['stride_w']
            stridey = args['stride_h']
        elif 'strides' in list(args.keys()):
            strides = args['strides']
            stridex = strides[0]
            stridey = strides[1]
        else:
            if isinstance(stride, list):
                if len(stride) > 0:
                    stridex = stride[0]
                    stridey = stride[0]
                if len(stride) > 1:
                    stridex = stride[1]
            else:
                stridex = stride
                stridey = stride

        kx = 0
        ky = 0
        if 'kernels' in list(args.keys()):
            kernels = args['kernels']
            if len(kernels) != 2:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE2_ARG_KERNEL_LEN_NOT_EXPECTED')(len(kernels)))
            # Height comes first, then width
            ky = int(kernels[0])
            kx = int(kernels[1])
        elif 'kernel_h' in list(args.keys()) or 'kernel_w' in list(args.keys()):
            kx = args['kernel_w']
            ky = args['kernel_h']
        else:
            if isinstance(kernel, list):
                if len(kernel) == 1:
                    kx = kernel[0]
                    ky = kernel[0]
                if len(kernel) > 1:
                    ky = kernel[0]
                    kx = kernel[1]
            else:
              kx = kernel
              ky = kernel

        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GOT_PADX')(str(padx), str(pady)))
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GOT_STRIDEX')(str(stridex), str(stridey)))
        self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GOT_KX')(str(kx), str(ky)))
        return parmstype(padx, pady, stridex, stridey, kx, ky, pad_type)

    def save_axis_order(self, buffer_name):
        axis_order = self._axis_transformer.get_target_axis_order(buffer_name)
        if len(axis_order):
            self.model.set_buffer_axis_order(str(buffer_name), list(axis_order))

    def setup_preprocessing(self, op):
        if not self.enable_preprocessing:
            self.logger.debug(code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_NO_PREPROCESS'))
            return

        data_name = str(self.get_input_name(op))
        output_name = str(self.get_output_name(op))
        last_layer_name = data_name

        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_SETUP_PREPROCESS_OP')(output_name, data_name))

        args = self.get_args(op.arg)

        # Get the data dims and reorder from CHW to HWC
        net_dims = self.input_dims[data_name]
        if len(net_dims) == 4:  # Reorder dims from NCHW to NHWC
            data_dims = list(map(int, [net_dims[0], net_dims[2], net_dims[3], net_dims[1]]))
        elif len(net_dims) == 3:  # Reorder dims from CHW to HWC
            data_dims = list(map(int, [1, net_dims[1], net_dims[2], net_dims[0]]))
        else:
            data_dims = list(map(int, dims))

        id_ = self.model.add_data_layer(data_name, data_dims, self.encoding, "bgr", "default")
        self._axis_transformer.update_src_axis_order('data', len(net_dims), data_name, len(net_dims))
        self._axis_transformer.update_target_axis_order('data', len(data_dims), data_name, len(data_dims))
        self.save_axis_order(data_name)
        self.layer_id_map[data_name] = id_
        self.logger.debug(
            code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_SETUP_ADD_LAYER')(data_name, str(data_dims)))

        # add cropping if present
        crop_size = 0
        if 'crop' in list(args.keys()):
            crop_size = args['crop']
            if len(net_dims) == 4:
                original_height = net_dims[1]
                original_weight = net_dims[2]
            else:
                original_height = net_dims[0]
                original_weight = net_dims[1]

        if  crop_size != 0 and (crop_size != original_height or crop_size != original_weight):
            if crop_size > original_height or crop_size > original_weight:
                errs = code_to_message.get_error_message('ERROR_CAFFE2_CROP_SIZE_LARGER_THAN_INPUT_DIMS')(crop_size, net_dims[0:2])
                raise ValueError(errs)
            offset_y = (original_height-crop_size)//2
            offset_x = (original_weight-crop_size)//2
            if len(net_dims) == 4:
                offsets = [0, offset_y, offset_x, 0]
                output_dim = [data_dims[0],crop_size,crop_size,data_dims[3]]
            else:
                offsets = [0, offset_y, offset_x, 0]
                output_dim = [1,crop_size,crop_size,data_dims[2]]
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
                self.logger.warn(
                    code_to_message.get_warning_message('WARNING_CAFFE2_CROP_INPUT_BUFFER_MORE_THAN_ONE_EL')(str(crop_input_buffer[0])))
            crop_input_buffer = str(crop_input_buffer[0])

            id_ = self.model.add_crop_layer(crop_layer_name,
                                            offsets,
                                            output_dim,
                                            crop_input_buffer,
                                            crop_layer_name)

            self._axis_transformer.update_src_axis_order('CROP', len(net_dims), crop_layer_name, len(net_dims), last_layer_name)
            self._axis_transformer.update_target_axis_order('CROP', len(data_dims), crop_layer_name, len(data_dims), last_layer_name)
            self.save_axis_order(crop_layer_name)

            last_layer_name = crop_layer_name

            if len(net_dims) == 4:
                crop_dim = [net_dims[0], crop_size, crop_size, net_dims[3]]
            else:
                crop_dim = [1,crop_size, crop_size, net_dims[2]]
        else:
            crop_dim = net_dims[:]

        # as per caffe, either mean_file or mean_value may be specified, but not both.
        mean_data = None

        if 'mean' in list(args.keys()):
            mean_data = numpy.zeros(crop_dim, dtype=numpy.float32)
            mean_data[:] = float(args['mean'])

        if mean_data is not None:
            if mean_data.shape != tuple(crop_dim):
                errs = code_to_message.get_error_message('ERROR_CAFFE2_MEAN_DATA_WRONG_DIMS')(data_name, str(crop_dim), str(mean_data.shape))
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
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_GET_OUTPUT_BUFFER_NAME')(last_layer_name, str(self._network_topology.get_output_buffer_name(last_layer_name))))
            self.logger.debug(
                code_to_message.get_debugging_message('DEBUG_CAFFE2_CONVERT_OUTPUT_BUFFER')(last_layer_name, str(self._network_topology.get_output_buffers(last_layer_name))))

            # add_crop_layer takes a single input, so lets reduce the array to a single one and just
            # make sure as a sanity check, that we indeed have just a single one
            subtract_mean_input_buffer = self._network_topology.get_output_buffers(last_layer_name)
            if len(subtract_mean_input_buffer) > 1:
                self.logger.warn(
                    code_to_message.get_debugging_message('WARNING_CAFFE2_SUB_MEAN_BUFFER_MORE_THAN_ONE_EL')(str(subtract_mean_input_buffer[0])))
            subtract_mean_input_buffer = str(subtract_mean_input_buffer[0])

            id_ = self.model.add_subtract_mean_layer(subtract_mean_layer_name,
                                                     mean_data,
                                                     subtract_mean_input_buffer,
                                                     subtract_mean_layer_name)

            self._axis_transformer.update_src_axis_order('SUBTRACTMEAN', len(net_dims), subtract_mean_layer_name, len(net_dims), last_layer_name)
            self._axis_transformer.update_target_axis_order('SUBTRACTMEAN', len(data_dims), subtract_mean_layer_name, len(data_dims), last_layer_name)
            self.save_axis_order(subtract_mean_layer_name)

            last_layer_name = subtract_mean_layer_name

        self.layer_id_map[data_name] = id_

        # Make the last pre-processing layer's output blob a proxy for the original input layer.
        if last_layer_name != data_name:
            self._network_topology.install_buffer_proxy(data_name, last_layer_name)
