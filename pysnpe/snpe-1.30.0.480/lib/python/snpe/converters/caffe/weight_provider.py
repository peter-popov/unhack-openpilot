# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy
import random

from snpe.converters.common.utils.snpe_converter_utils import SNPEUtils
from snpe.converters.common.utils import code_to_message
from functools import reduce
snpeUtils = SNPEUtils()


# ------------------------------------------------------------------------------
#   Weight Providers
# ------------------------------------------------------------------------------
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
        scale_factor = scale_factor if scale_factor != 0 else 1
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
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])

        if bias_term:
            c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])
        else:
            c_bias = numpy.require([0] * layer.convolution_param.num_output, dtype=numpy.float32)
        return c_weights, c_bias

    def get_deconv_weights(self, layer, bias_term):
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])

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
            # get (input_size, output_size) shape which is what IR optimize will expect
            c_weights = numpy.ascontiguousarray(numpy.transpose(c_weights, (1, 0)))
            weights_list.append(c_weights)
        return weights_list, c_bias

    def get_lstm_weights(self, layer):
        c_x_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        c_bias = snpeUtils.blob2arr(self.weights_map[layer.name][1])
        c_h_weights = snpeUtils.blob2arr(self.weights_map[layer.name][2])

        return c_x_weights, c_bias, c_h_weights

    def get_prelu_weights(self, layer):
        c_weights = snpeUtils.blob2arr(self.weights_map[layer.name][0])
        return [float(f) for f in c_weights]

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
    def __init__(self, model, graph):
        self.model = model
        self._graph = graph

    @staticmethod
    def make_weights(shape, macs_per_output=1):
        """
        Create random weights with a given shape, which will
        keep the power of a layer's activations the same as that of
        it's input. Keep in mind that by the time this is called axis tracking is not done hence
        we will use Caffe's shape ordering to calculate dims.

        parameters:
        shape           -- tuple of int, the shape the weights should have.
        macs_per_output -- The number of macs which will contribute to a single
                           activation. Used for normalization.
        """
        weights = numpy.random.random(shape)*2-1
        weights /= macs_per_output
        return numpy.require(weights, dtype=numpy.float32)

    def get_bn_weights(self, layer):
        input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
        buf = self._graph.get_buffer(input_layer_name)
        input_depth = buf.get_buf_dims()[1]
        c_weights = self.make_weights((input_depth,))
        return c_weights, c_weights  # use weights for bias

    def get_batch_norm_weights(self, layer, input_depth_prev):
        # For RandomWeightProvider previous layer might not be registered
        # by the time the batchnorm layer weights are requested since
        # batchnorm weights might be needed in the conv layer itself
        # if it is being folded in. In this case the prev input depth
        # is provided as a parameter
        if input_depth_prev is None:
            input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
            buf = self._graph.get_buffer(input_layer_name)
            input_depth = buf.get_buf_dims()[1]
        else:
            input_depth = input_depth_prev
        c_weights = self.make_weights((input_depth,))
        return c_weights, c_weights  # use weights for bias

    def get_normalize_weights(self, layer):
        if layer.norm_param.channel_shared:
            input_depth = 1
        else:
            input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
            buf = self._graph.get_buffer(input_layer_name)
            input_depth = buf.get_buf_dims()[1]
        c_weights = self.make_weights((input_depth,))
        return c_weights

    def get_conv_weights(self, layer, bias_term):
        conv_param = layer.convolution_param
        kx = 0
        ky = 0

        # determine kernel dims to create/generate synthetic weights

        if conv_param.kernel_h and conv_param.kernel_w:
            kx = conv_param.kernel_w
            ky = conv_param.kernel_h
        if isinstance(conv_param.kernel_size, int):
            kx = conv_param.kernel_size
            ky = conv_param.kernel_size
        else:
            if len(conv_param.kernel_size) > 0:
                kx = conv_param.kernel_size[0]
                ky = conv_param.kernel_size[0]
            if len(conv_param.kernel_size) > 1:
                kx = conv_param.kernel_size[1]
        if kx == 0 or ky == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_FIELDS')
                             (str(layer.name)))

        output_depth = conv_param.num_output
        groups = conv_param.group
        input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
        buf = self._graph.get_buffer(input_layer_name)
        input_depth = buf.get_buf_dims()[1]
        weights_shape = (output_depth, input_depth/groups, ky, kx)

        macs_per_output = ky*ky*input_depth/groups
        c_weights = self.make_weights(weights_shape, macs_per_output)
        c_bias = self.make_weights((output_depth,))
        return c_weights, c_bias

    def get_deconv_weights(self, layer, bias_term):
        conv_param = layer.convolution_param

        kx = 0
        ky = 0
        if conv_param.kernel_h and conv_param.kernel_w:
            kx = conv_param.kernel_w
            ky = conv_param.kernel_h
        if isinstance(conv_param.kernel_size, int):
            kx = conv_param.kernel_size
            ky = conv_param.kernel_size
        else:
            if len(conv_param.kernel_size) > 0:
                kx = conv_param.kernel_size[0]
                ky = conv_param.kernel_size[0]
            if len(conv_param.kernel_size) > 1:
                kx = conv_param.kernel_size[1]
        if kx == 0 or ky == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_FIELDS')
                             (str(layer.name)))

        output_depth = conv_param.num_output
        groups = conv_param.group
        input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
        buf = self._graph.get_buffer(input_layer_name)
        input_depth = buf.get_buf_dims()[1]
        weights_shape = (input_depth, output_depth/groups, ky, kx)

        macs_per_output = ky*kx*input_depth/groups
        c_weights = self.make_weights(weights_shape, macs_per_output)
        c_bias = self.make_weights((output_depth,))
        return c_weights, c_bias

    def get_fc_weights(self, layer, input_depths, bias_term):
        # input_depths unused
        fc_parm = layer.inner_product_param
        output_depth = fc_parm.num_output
        c_weights_list = []
        for name in layer.bottom:
            input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(name)
            buf = self._graph.get_buffer(input_layer_name)
            # Strip batch dimension for input size calculation
            input_dims = buf.get_buf_dims()[1:]
            input_size = reduce(lambda a, b: a*b, input_dims)
            w = self.make_weights((input_size, output_depth), input_size)

            c_weights_list.append(w)
        if bias_term:
            c_bias = self.make_weights((output_depth,))
        else:
            c_bias = numpy.zeros((output_depth,), dtype=numpy.float32)
        return c_weights_list, c_bias

    def get_lstm_weights(self, layer):
        input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
        buf = self._graph.get_buffer(input_layer_name)
        input_depth = buf.get_buf_dims()[-1]
        output_depth = layer.recurrent_param.num_output
        c_x_weights = self.make_weights((output_depth * 4, input_depth), input_depth)
        c_bias = self.make_weights((output_depth * 4,))
        c_h_weights = self.make_weights((output_depth * 4, output_depth), input_depth)
        return c_x_weights, c_bias, c_h_weights

    def get_prelu_weights(self, layer):
        input_layer_name = self._graph.naming_policy.get_caffe_name_mapping(layer.bottom[0])
        buf = self._graph.get_buffer(input_layer_name)
        input_depth = buf.get_buf_dims()[1]
        return [(2*random.random())-1 for _ in range(input_depth)]

    def get_scale_weights(self, layer, bias_term, input_depth):
        # Bias should always be created. Weights are only created when scale takes a
        # single input
        c_bias = self.make_weights((input_depth,))

        # Only create weights when there isn't a second input (which ARE the weights)
        c_weights = None
        if len(layer.bottom) == 1:
            c_weights = c_bias
        return c_weights, c_bias
