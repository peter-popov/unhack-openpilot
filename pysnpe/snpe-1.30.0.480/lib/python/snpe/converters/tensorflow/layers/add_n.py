#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
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
from snpe.converters.tensorflow.util import ConverterError


class AddNLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(AddNLayerResolver.Descriptor, self).__init__('ElementWiseSumN', name, nodes)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['AddN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []

        descriptors = []
        for match in matches:
            add_op = match['root']
            descriptors.append(AddNLayerResolver.Descriptor(str(add_op.name), match.consumed_nodes))
        return descriptors


class AddNLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        if len(input_names) < 2:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_ADD_N_NUM_OF_INPUTS')(descriptor.layer_name))
        output_name = descriptor.output_names[0]
        current_input_names = [input_names[0], input_names[1]]
        current_output_name = descriptor.layer_name + '_unroll_1'
        converter_context.model.add_elementwise_sum_layer(descriptor.layer_name + '_unroll_1',
                                                          [1.0 for _ in current_input_names],
                                                          current_input_names,
                                                          current_output_name)

        for input_index in range(2, len(input_names) - 1):
            current_input_names = [current_output_name, input_names[input_index]]
            current_output_name = descriptor.layer_name + '_unroll_' + str(input_index)
            converter_context.model.add_elementwise_sum_layer(descriptor.layer_name + '_unroll_' + str(input_index),
                                                              [1.0 for _ in current_input_names],
                                                              current_input_names,
                                                              current_output_name)
        current_input_names = [current_output_name, input_names[-1]]
        return converter_context.model.add_elementwise_sum_layer(descriptor.layer_name,
                                                                 [1.0 for _ in current_input_names],
                                                                 current_input_names,
                                                                 output_name)
