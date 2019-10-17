#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    ConverterRepeatableSequenceTreeNode,
    NonConsumableConverterSequenceNode
)
from snpe.converters.tensorflow.layers.batchnorm import BatchNormLayerResolver
from snpe.converters.tensorflow.layers.crop import CropLayerResolver
from snpe.converters.tensorflow.layers.pad import PadLayerResolver
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.util import OperationNotFoundError


class ConvolutionLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_STRIDES = 'strides'
    TF_ATTRIBUTE_PADDING = 'padding'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, conv_op, bias_op, strides, padding, weights, biases, output_names=None):
            super(ConvolutionLayerResolver.Descriptor, self).__init__('Convolution', name, nodes,
                                                                      output_names=output_names)
            self.conv_op = conv_op
            self.bias_op = bias_op
            self.strides = strides
            self.padding = padding
            self.weights = weights
            self.biases = biases
            self.dilationX = 1
            self.dilationY = 1
            self.groups = len([op for op in nodes if op.type == 'Conv2D'])
            self.output_op = self.child_ops[-1]
            self.input_ops = [conv_op]

        def is_input_op(self, op):
            return op in self.input_ops

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Conv2D'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['root']
            bias_op = None
            biases = None
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = list(match.consumed_nodes)
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            conv_output_ops = graph_helper.get_op_outputs(conv_op)
            bias_op, biases = self.get_conv_bias(graph_helper, conv_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)
            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes, conv_op, bias_op,
                                                             strides, padding, weights, biases,
                                                             output_names=output_op_nodes_names)
            descriptors.append(descriptor)
        return descriptors

    def get_biases(self, graph_helper, conv_op, bias_op):
        _, biases_tensor = GraphHelper.get_op_input_tensors(bias_op, ('?', '?'))
        if biases_tensor.op.type not in ['Identity', 'Const'] and \
                not graph_helper.check_tensor_const_origin(biases_tensor):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_BIAS')(conv_op.name))
        biases = graph_helper.evaluate_tensor_output(biases_tensor)
        return biases

    def get_conv_bias(self, graph_helper, input_op, conv_output_ops):
        bias_op = None
        biases = None
        try:
            bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'BiasAdd')
            biases = self.get_biases(graph_helper, input_op, bias_op)

        except OperationNotFoundError:
            pass

        if bias_op is None:
            try:
                bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'Add')
                biases = self.get_biases(graph_helper, input_op, bias_op)
            except (OperationNotFoundError, ConverterError):
                bias_op = None
                biases = None

        if biases is not None and len(biases.shape) != 1:
            bias_op = None
            biases = None

        return bias_op, biases

    def get_weights(self, graph_helper, conv_op):
        _, weights_tensor = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        if weights_tensor.op.type not in ['Identity', 'Const', 'Split', 'FakeQuantWithMinMaxVars']:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_WEIGHTS')(conv_op.name))
        weights = graph_helper.evaluate_tensor_output(weights_tensor)
        return weights


class DilatedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            ConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('dilation_sizes', ['?']),
            ConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['Conv2D']),
            ConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),
            ConverterSequenceNode('block_shape_out', ['?']),
            ConverterSequenceNode('crops', ['?'])]
        )
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]
            batch_to_space_op = match['batch_to_space']
            conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)
            bias_op, biases = self.get_conv_bias(graph_helper, batch_to_space_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)
            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, strides, padding, weights, biases,
                                                    output_names=output_op_nodes_names)
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [match['space_to_batch']]
            descriptors.append(d)
        return descriptors


class DepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):

    def __init__(self):
        super(DepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence_with_bias = GraphSequence([
            ConverterSequenceNode('conv', ['DepthwiseConv2dNative']),
            ConverterSequenceNode('bias', ['BiasAdd', 'Add']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.graph_sequence_with_bias.set_inputs('bias', ['conv', 'other'])
        self.graph_sequence_with_bias.set_outputs(['bias'])

        self.graph_sequence = GraphSequence([ConverterSequenceNode('conv', ['DepthwiseConv2dNative'])])
        self.graph_sequence.set_outputs(['conv'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        matches += graph_matcher.match_sequence(self.graph_sequence_with_bias)
        descriptors = []
        for match in matches:
            self._resolve_from_match(descriptors, graph_helper, match)
        return descriptors

    def _resolve_from_match(self, descriptors, graph_helper, match):
        conv_op = match['conv']
        strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
        padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
        weights = self.get_weights(graph_helper, conv_op)
        weights = np.transpose(weights, [0, 1, 3, 2])

        if 'bias' in match:
            biases = self.get_biases(graph_helper, conv_op, match['bias'])
        else:
            biases = np.zeros(np.shape(weights)[-1], dtype=np.float32)
        consumed_nodes = match.consumed_nodes
        d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                conv_op, None, strides, padding, weights, biases)
        input_tensor, _ = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        d.groups = graph_helper.get_op_output_shape(input_tensor)[-1]
        descriptors.append(d)


class DilatedDepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedDepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('dilation_sizes', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['DepthwiseConv2dNative']),
            ConverterSequenceNode('kernel', ['?']),
            NonConsumableConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),  # output
            NonConsumableConverterSequenceNode('block_shape_out', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?'])
        ])
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            biases = None
            bias_op = None
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            if weights.shape[3] != 1:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_WEIGHTS_SHAPE')(conv_op.name))
            weights = np.transpose(weights, [0, 1, 3, 2])
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]
            batch_to_space_op = match['batch_to_space']
            conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)
            bias_op, biases = self.get_conv_bias(graph_helper, batch_to_space_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)
            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            space_to_batch_op = match['space_to_batch']
            paddings_op = match['paddings']
            paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])
            input_op = conv_op

            batch_to_space_op = match['batch_to_space']
            crop_op = match['crops']
            crops_tensor = graph_helper.evaluate_tensor_output(crop_op.outputs[0])
            output_names = [str(conv_op.outputs[0].name)]

            if paddings_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor):
                paddings_tensor = np.pad(paddings_tensor, ((1, 1), (0, 0)), 'constant')
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(space_to_batch_op.name),
                    [match['space_to_batch'], match['dilation_sizes'], match['paddings']],
                    paddings_tensor,
                    modeltools.PADDING_CONSTANT,
                    0.0,
                    output_names=[str(space_to_batch_op.outputs[0].name)])
                descriptors.append(pad_descriptor)
            else:
                consumed_nodes.extend([space_to_batch_op, paddings_op, match['dilation_sizes']])
                input_op = space_to_batch_op

            if crops_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor):
                crops_tensor = np.pad(crops_tensor, ((1, 1), (0, 0)), 'constant')
                offsets = crops_tensor[:, 0]
                size = np.array(graph_helper.get_op_output_shape(match['batch_to_space']), dtype=np.int32)
                crop_descriptor = CropLayerResolver.Descriptor(
                    str(match['batch_to_space'].name),
                    [match['batch_to_space'], match['block_shape_out'], match['crops']],
                    offsets,
                    size,
                    output_names=[str(match['batch_to_space'].outputs[0].name)])
                descriptors.append(crop_descriptor)
            else:
                consumed_nodes.extend([batch_to_space_op, crop_op, match['block_shape_out']])
                output_names = output_op_nodes_names

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, strides, padding, weights,
                                                    biases,
                                                    output_names=output_names)

            d.groups = graph_helper.get_op_output_shape(space_to_batch_op)[-1]
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [input_op]
            descriptors.append(d)

        return descriptors


class ConvolutionLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.get_input_layer_output_shape_for(descriptor.input_ops[0])

        if descriptor.bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.output_op)

        pad_y, pad_x, padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=input_dims[-3:-1],
                                                                                        output_size=output_dims[-3:-1],
                                                                                        strides=descriptor.strides[1:3],
                                                                                        padding=descriptor.padding,
                                                                                        filter_dims=descriptor.weights.shape,
                                                                                        dilation=[descriptor.dilationY,
                                                                                                  descriptor.dilationX])

        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        # using layer name for output buffer name
        return converter_context.model.add_conv_layer(name=descriptor.layer_name,
                                                      weights=descriptor.weights,
                                                      bias=descriptor.biases,
                                                      padx=pad_x,
                                                      pady=pad_y,
                                                      padding_mode=modeltools.PADDING_ZERO,
                                                      padding_size_strategy=padding_strategy,
                                                      stridex=int(descriptor.strides[2]),
                                                      stridey=int(descriptor.strides[1]),
                                                      dilationx=descriptor.dilationX,
                                                      dilationy=descriptor.dilationY,
                                                      input_name=input_name,
                                                      output_name=descriptor.output_names[0],
                                                      groups=descriptor.groups)

    @classmethod
    def calculate_padding_size(cls, input_size, output_size, strides, padding, filter_dims, dilation):
        pad_y, pad_x = 0, 0
        padding_size_strategy = modeltools.PADDING_SIZE_IMPLICIT_VALID
        if padding.decode() == "SAME":
            filter_h = filter_dims[0] + (filter_dims[0] - 1) * (dilation[0] - 1)
            filter_w = filter_dims[1] + (filter_dims[1] - 1) * (dilation[1] - 1)
            pad_y = max(((output_size[0] - 1) * strides[0] + filter_h - input_size[0]), 0)
            pad_x = max(((output_size[1] - 1) * strides[1] + filter_w - input_size[1]), 0)
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Asymmetry
            pad_y //= 2
            pad_x //= 2
            padding_size_strategy = modeltools.PADDING_SIZE_IMPLICIT_SAME
        return int(pad_y), int(pad_x), padding_size_strategy

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):

        filtered_descriptors = [d for d in output_descriptors if isinstance(d, BatchNormLayerResolver.Descriptor)]
        if filtered_descriptors == output_descriptors and len(output_descriptors) == 1:
            descriptor.weights = descriptor.weights * filtered_descriptors[0].weights
            descriptor.biases = (descriptor.biases * filtered_descriptors[0].weights) + filtered_descriptors[0].biases
            # TODO: remove adding the quantization params here when tf_to_ir is complete
            if not filtered_descriptors[0].pre_calculated:
                q_params = {descriptor.layer_name:
                            {
                                "bn_params": {"gamma": filtered_descriptors[0].scale,
                                              "beta": filtered_descriptors[0].beta},
                                "output_encodings": {},
                                "param_encodings": {}
                            }
                            }
                converter_context.model.add_quantization_params(q_params)
            descriptor.output_names = output_descriptors[0].output_names
            converter_context.merge_descriptors(output_descriptors[0], descriptor)


class GroupedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(GroupedConvolutionLayerResolver, self).__init__()

        # grouped convolution with split
        tree_output_node = ConverterSequenceNode('conv_op', ['Conv2D'])
        self.sequence = GraphSequence([
            ConverterSequenceNode('a', ['Split']),
            ConverterSequenceNode('b', ['Split']),
            ConverterRepeatableSequenceTreeNode('repeatable_graph', tree_output_node, tree_output_node),
            ConverterSequenceNode('concat_op', ['Concat']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('concat_dim', ['Const']),
            NonConsumableConverterSequenceNode('split_dim1', ['Const']),
            ConverterSequenceNode('split_dim2', ['Const'])
        ])
        self.sequence.set_inputs('a', ['inputs', 'split_dim1'])
        self.sequence.set_inputs('b', ['weights', 'split_dim2'])
        self.sequence.set_inputs('repeatable_graph', ['a', 'b'])
        self.sequence.set_inputs('concat_op', ['repeatable_graph', 'concat_dim'])
        self.sequence.set_outputs(['concat_op'])

        # grouped convolution with strided slice
        repeatable_sequence = GraphSequence([
                ConverterSequenceNode('ss', ['StridedSlice']),
                ConverterSequenceNode('ss_begin', ['Const']),
                ConverterSequenceNode('ss_end', ['Const']),
                ConverterSequenceNode('ss_strides', ['Const']),
                ConverterSequenceNode('conv', ['Conv2D']),
                ConverterSequenceNode('bias', ['BiasAdd']),
                ConverterSequenceNode('weights', ['Identity', 'Const']),
                ConverterSequenceNode('biases', ['Identity', 'Const'])
        ])
        repeatable_sequence.set_inputs('ss', ['ss_begin', 'ss_end', 'ss_strides'])
        repeatable_sequence.set_inputs('conv', ['ss', 'weights'])
        repeatable_sequence.set_inputs('bias', ['biases', 'conv'])
        repeatable_sequence.set_outputs(['bias'])

        self.sequence_with_strided_slice = GraphSequence([
            ConverterRepeatableSequenceTreeNode('repeatable_graph',
                                                tree_output_node=repeatable_sequence['bias'],
                                                tree_input_node=repeatable_sequence['ss']),
            ConverterSequenceNode('concat', ['Concat', 'ConcatV2']),
            ConverterSequenceNode('axis', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        self.sequence_with_strided_slice.set_inputs('repeatable_graph', ['input'])
        self.sequence_with_strided_slice.set_inputs('concat', ['repeatable_graph', 'axis'])
        self.sequence_with_strided_slice.set_outputs(['concat'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            conv_op = match['conv_op_1']
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = match['weights']
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.sequence.output_nodes]
            concat_op = match['concat_op']
            concat_op_output_ops = graph_helper.get_op_outputs(concat_op)
            bias_op, biases = self.get_conv_bias(graph_helper, concat_op, concat_op_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.outputs[0].get_shape()[-1], dtype=np.float32)

            weights = graph_helper.evaluate_tensor_output(weights.outputs[0])
            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes, conv_op, bias_op,
                                                             strides, padding, weights, biases,
                                                             output_names=output_op_nodes_names)
            descriptor.input_ops = [match['a'], match['b']]
            descriptors.append(descriptor)

        for match in graph_matcher.match_sequence(self.sequence_with_strided_slice):
            if not match.consumed_nodes:
                continue
            input_op = match['input']
            concat_op = match['concat']
            axis_op = match['axis']
            conv_ops = self._get_repeatable_op_by_id(match, 'conv')
            weight_ops = self._get_repeatable_op_by_id(match, 'weights')
            bias_ops = self._get_repeatable_op_by_id(match, 'biases')
            bias_add_ops = self._get_repeatable_op_by_id(match, 'bias')
            ss_ops = self._get_repeatable_op_by_id(match, 'ss')

            input_shape = graph_helper.get_op_output_shape(input_op)
            weight_shapes = [graph_helper.get_op_output_shape(weight_op) for weight_op in weight_ops]

            ss_strides = [graph_helper.evaluate_tensor_output(ss_strides_op.outputs[0]).tolist()
                          for ss_strides_op in self._get_repeatable_op_by_id(match, 'ss_strides')]
            ss_begins = [graph_helper.evaluate_tensor_output(ss_begin_op.outputs[0]).tolist()
                          for ss_begin_op in self._get_repeatable_op_by_id(match, 'ss_begin')]
            ss_ends = [graph_helper.evaluate_tensor_output(ss_end_op.outputs[0]).tolist()
                          for ss_end_op in self._get_repeatable_op_by_id(match, 'ss_end')]

            bias_add_shapes = [graph_helper.get_op_output_shape(bias_add_op) for bias_add_op in bias_add_ops]

            strides = [conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES) for conv_op in conv_ops]
            paddings = [conv_op.get_attr(self.TF_ATTRIBUTE_PADDING) for conv_op in conv_ops]

            ss_shapes = [graph_helper.get_op_output_shape(ss_op.outputs[0])
                         for ss_op in ss_ops]

            num_groups = len(conv_ops)

            axis = graph_helper.evaluate_tensor_output(axis_op.outputs[0])

            is_grouped_convolution = True
            is_grouped_convolution &= self._elements_are_same(bias_add_shapes)
            is_grouped_convolution &= self._elements_are_same(weight_shapes)
            is_grouped_convolution &= self._elements_are_same(strides)
            is_grouped_convolution &= self._elements_are_same(paddings)
            is_grouped_convolution &= self._elements_are_same(ss_shapes)
            is_grouped_convolution &= self._elements_are_same(ss_strides)
            is_grouped_convolution &= not self._elements_are_same(ss_begins)
            is_grouped_convolution &= not self._elements_are_same(ss_ends)
            # stride slices must evenly divide the last dimension of input to number of groups
            is_grouped_convolution &= ss_shapes[0][-1] * num_groups == input_shape[-1]
            # strides must be all ones at all dimensions
            is_grouped_convolution &= ss_strides[0] == [1] * len(ss_strides[0])
            # concat must be on the last axis in grouped convolution
            is_grouped_convolution &= axis == -1 or axis == (len(bias_add_shapes[0]) - 1)

            if not is_grouped_convolution:
                logging.getLogger().warning(code_to_message.get_error_message('WARNING_TF_GROUP_CONV_RESOLVE'))
                continue

            weight_tensors = [graph_helper.evaluate_tensor_output(weight_op.outputs[0])
                              for weight_op in weight_ops]
            weights = np.concatenate(weight_tensors, axis=-1)

            bias_tensors = [graph_helper.evaluate_tensor_output(bias_op.outputs[0])
                              for bias_op in bias_ops]
            biases = np.concatenate(bias_tensors, axis=-1)

            descriptor = ConvolutionLayerResolver.Descriptor(
                str(concat_op.name), match.consumed_nodes, conv_ops[0], None,
                strides[0], paddings[0], weights, biases,
                output_names=[str(concat_op.outputs[0].name)])
            descriptor.input_ops = ss_ops
            descriptor.output_op = concat_op
            descriptors.append(descriptor)

        return descriptors

    @classmethod
    def _get_repeatable_op_by_id(cls, match, name):
        ops = []
        indexed_id = name + '_{}'
        i = 1
        while indexed_id.format(i) in match:
            ops.append(match[indexed_id.format(i)])
            i += 1
        return ops

    @classmethod
    def _elements_are_same(cls, array):
        return all([element == array[0] for element in array])
