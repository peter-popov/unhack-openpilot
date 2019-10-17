#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.layers.convolution import ConvolutionLayerBuilder
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.util import OperationNotFoundError


class DeConvolutionOptimizedLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, deconv_op, bias_op, weights, strides, padding, biases, input_tensor,
                     output_names=None):
            super(DeConvolutionOptimizedLayerResolver.Descriptor, self).__init__('Deconvolution', name, nodes,
                                                                                 output_names=output_names)
            self.deconv_op = deconv_op
            self.bias_op = bias_op
            self.weights = weights
            self.strides = strides
            self.padding = padding
            self.biases = biases
            self.input_tensor = input_tensor

        def is_input_tensor(self, op, tensor):
            if op == self.deconv_op and tensor != self.deconv_op.inputs[2]:
                return False
            return True

        @property
        def output_names(self):
            if self.bias_op:
                output_name = str(self.bias_op.outputs[0].name)
            else:
                output_name = str(self.deconv_op.outputs[0].name)
            return [output_name]

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Conv2DBackpropInput'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_trans_op = match['root']
            _, weights_tensor, input_tensor = GraphHelper.get_op_input_tensors(conv_trans_op, ('?', '?', '?'))
            if weights_tensor.op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_DECONV_CANT_FIND_WEIGHTS_NODE'))
            strides = conv_trans_op.get_attr('strides')
            padding = conv_trans_op.get_attr('padding')
            weights = graph_helper.evaluate_tensor_output(weights_tensor)
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            bias_op = None
            try:
                output_ops = graph_helper.get_op_outputs(conv_trans_op)
                bias_op = GraphHelper.filter_single_op_by_type(output_ops, 'BiasAdd')

                _, biases = GraphHelper.get_op_input_tensors(bias_op, ('?', '?'))
                if biases.op.type not in ['Const', 'Identity']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_DECONV_CANT_FIND_BIAS_NODE'))
                biases = graph_helper.evaluate_tensor_output(biases)
                consumed_nodes.append(bias_op)
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
            except OperationNotFoundError:
                biases = np.zeros(np.shape(weights)[-2], dtype=np.float32)

            descriptors.append(
                DeConvolutionOptimizedLayerResolver.Descriptor(str(conv_trans_op.name),
                                                               consumed_nodes,
                                                               conv_trans_op,
                                                               bias_op,
                                                               weights,
                                                               strides,
                                                               padding,
                                                               biases,
                                                               input_tensor,
                                                               output_names=output_op_nodes_names))
        return descriptors


class DeConvolutionLayerBuilder(LayerBuilder, object):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: DeConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.input_tensor.op)
        if descriptor.bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op)

        pad_y, pad_x, padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=output_dims[-3:-1],
                                                                                        output_size=input_dims[-3:-1],
                                                                                        strides=descriptor.strides[1:3],
                                                                                        padding=descriptor.padding,
                                                                                        filter_dims=descriptor.weights.shape,
                                                                                        dilation=[1, 1])
        if pad_y != pad_x:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_DECONV_NO_SUPPORT_RECT_PADDING'))

        weights = np.transpose(descriptor.weights, (0, 1, 3, 2)).copy()

        input_names = self.get_input_name(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_deconvolution_layer(name=descriptor.layer_name,
                                                               weights=weights,
                                                               bias=descriptor.biases,
                                                               stride=descriptor.strides[1],
                                                               padding_size_strategy=padding_strategy,
                                                               padx=pad_x,
                                                               pady=pad_y,
                                                               input_name=input_names,
                                                               output_name=descriptor.output_names[0],
                                                               output_width=output_dims[-2],
                                                               output_height=output_dims[-3],
                                                               groups=1)

    @classmethod
    def calculate_output_size(cls, input_size, strides, padding, filter_dims, dilation_x, dilation_y):
        if padding.decode() == 'SAME':
            height = input_size[0] * strides[0]
            width = input_size[1] * strides[1]
        else:
            height = (input_size[0] - 1) * strides[0] + dilation_y * filter_dims[0]
            width = (input_size[1] - 1) * strides[1] + dilation_x * filter_dims[1]
        return height, width
