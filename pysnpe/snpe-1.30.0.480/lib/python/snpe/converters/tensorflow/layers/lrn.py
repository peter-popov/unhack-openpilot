#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class LrnLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, window_size, alpha, beta, bias):
            super(LrnLayerResolver.Descriptor, self).__init__('LRN', name, operations)
            self.window_size = window_size
            self.alpha = alpha
            self.beta = beta
            self.bias = bias

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['LRN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            lrn_op = match['root']
            window_size = 1 + lrn_op.get_attr('depth_radius') * 2
            alpha = lrn_op.get_attr('alpha')
            beta = lrn_op.get_attr('beta')
            bias = lrn_op.get_attr('bias')
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                LrnLayerResolver.Descriptor(str(lrn_op.name), consumed_nodes, window_size, alpha, beta, bias))
        return potential_descriptors


class LrnLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LrnLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_cmrn_layer(name=descriptor.layer_name,
                                                      window_size=descriptor.window_size,
                                                      alpha=float(descriptor.alpha),
                                                      beta=descriptor.beta,
                                                      k=descriptor.bias,
                                                      input_name=input_name,
                                                      output_name=output_name)
