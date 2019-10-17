#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.layers.lstm import LstmLayerResolver
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.util import TensorNotFoundError


class SliceLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, split_sizes, split_count):
            super(SliceLayerResolver.Descriptor, self).__init__('Slice', name, nodes)
            self.axis = axis
            self.split_sizes = split_sizes
            self.split_count = split_count

        @property
        def output_names(self):
            return [str(t.name) for t in self.child_ops[-1].outputs]

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Split', 'SplitV'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            split_op = match['root']
            split_axis, split_sizes = self.get_split_axis_and_sizes(graph_helper, split_op)
            split_count = int(split_op.get_attr('num_split'))
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                SliceLayerResolver.Descriptor(str(split_op.name), consumed_nodes,
                                              split_axis,
                                              split_sizes,
                                              split_count))
        return potential_descriptors

    @classmethod
    def get_split_axis_and_sizes(cls, graph_helper, split_op):
        try:
            _, split_sizes, split_axis = GraphHelper.get_op_input_tensors(split_op, ('?', 'Const', 'Const'))
            split_sizes = list(graph_helper.evaluate_tensor_output(split_sizes))
        except TensorNotFoundError:
            split_axis, _ = GraphHelper.get_op_input_tensors(split_op, ('Const', '?'))
            split_sizes = []

        split_axis = int(graph_helper.evaluate_tensor_output(split_axis))
        return split_axis, split_sizes


class SliceLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SliceLayerResolver.Descriptor
        :rtype: int
        """
        input_shape = converter_context.get_input_layer_output_shape_for(descriptor.child_ops[0])
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_names = descriptor.output_names

        split_points = self.get_split_positions(input_shape, descriptor.split_count, descriptor.split_sizes,
                                                descriptor.axis)

        return converter_context.model.add_slice_layer(name=descriptor.layer_name,
                                                       input_name=input_name,
                                                       axis=descriptor.axis,
                                                       slice_points=split_points,
                                                       output_names=output_names)

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        lstm_outputs = [d for d in output_descriptors if
                        isinstance(d, LstmLayerResolver.UnrolledTimeStepDescriptor) or
                        isinstance(d, LstmLayerResolver.StateDescriptor)]
        ignore = len(output_descriptors) > 0 and lstm_outputs == output_descriptors
        if ignore:
            descriptor.set_ignored(True)

    @classmethod
    def get_split_positions(cls, input_shape, split_count, split_sizes, split_axis):
        split_points = []
        if len(split_sizes) > 0:
            if sum(split_sizes) != input_shape[split_axis]:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_SLICE_SIZE_MISMATCH'))
            split_index = split_sizes[0]
            for size in split_sizes[1:]:
                split_points.append(int(split_index))
                split_index += size
        else:
            split_axis_dim = input_shape[split_axis]
            split_size = split_axis_dim // split_count
            if split_axis_dim % split_count:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_SLICE_UNEVEN_SPLIT'))
            for index in range(1, split_count):
                split_points.append(int(index * split_size))
        return split_points
