#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class PixelShuffleLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_BLOCK_SIZE = 'block_size'

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, upscale_factor):
            super(PixelShuffleLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)

            self.upscale_factor = upscale_factor

        @property
        def output_names(self):
            return [str(self.child_ops[0].outputs[0].name)]

        def is_output_op(self, op):
            return op in self.child_ops

        def get_output_names_for(self, input_tensors):
            return self.output_names

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['DepthToSpace'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)

        # Nothing matched
        if len(matches) == 0:
            return []

        potential_descriptors = []
        for match in matches:
            depth_to_space_op = match['root']
            upscale_factor = depth_to_space_op.get_attr(self.TF_ATTRIBUTE_BLOCK_SIZE)
            consumed_nodes = match.consumed_nodes

            potential_descriptors.append(
                PixelShuffleLayerResolver.Descriptor('PixelShuffle', str(depth_to_space_op.name), consumed_nodes, upscale_factor))

        return potential_descriptors


class PixelShuffleLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PixelShuffleLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_pixel_shuffle_layer(name = descriptor.layer_name,
                                                               input_name = input_name,
                                                               output_name = output_name,
                                                               upscale_factor = descriptor.upscale_factor)
