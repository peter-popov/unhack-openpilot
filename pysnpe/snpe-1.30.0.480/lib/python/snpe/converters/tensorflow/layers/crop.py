#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CropLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, offset, size, output_names=None):
            super(CropLayerResolver.Descriptor, self).__init__('Crop', name, nodes, output_names=output_names)
            self.offset = offset
            self.size = size

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Slice']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('offsets', ['?']),
            NonConsumableConverterSequenceNode('size', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'offsets', 'size'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        descriptors = []
        for match in matches:
            slice_op = match['root']
            input_shape = graph_helper.get_op_output_shape(match['input'])
            offset = graph_helper.evaluate_tensor_output(match['offsets'].outputs[0])
            size = graph_helper.evaluate_tensor_output(match['size'].outputs[0])
            for index in range(0, len(size)):
                if size[index] == -1:
                    size[index] = input_shape[index] - offset[index]

            consumed_nodes = match.consumed_nodes
            descriptors.append(
                CropLayerResolver.Descriptor(str(slice_op.name), consumed_nodes, offset, size))
        return descriptors


class CropLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_crop_layer(descriptor.output_names[0],
                                                      descriptor.offset.tolist(),
                                                      descriptor.size.tolist(),
                                                      input_name,
                                                      descriptor.output_names[0])
