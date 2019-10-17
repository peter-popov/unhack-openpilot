#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


from snpe.converters.tensorflow.common import LayerDescriptor
from snpe.converters.tensorflow.layers.relu_min_max import ReluMinMaxLayerResolver
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class Relu6LayerResolver(ReluMinMaxLayerResolver, object):

    class Descriptor(ReluMinMaxLayerResolver.Descriptor):
        def __init__(self, name, nodes):
            super(Relu6LayerResolver.Descriptor, self).__init__('Relu6', name, nodes, min_clamp=0, max_clamp=6)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Relu6'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            relu6_op = match['root']
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                Relu6LayerResolver.Descriptor(str(relu6_op.name), consumed_nodes))
        return potential_descriptors
