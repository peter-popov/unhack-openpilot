#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class SigmoidLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Sigmoid'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            sigmoid_op = match['root']
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                SigmoidLayerResolver.Descriptor('Sigmoid', str(sigmoid_op.name), consumed_nodes))
        return potential_descriptors


class SigmoidLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SigmoidLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_neuron_layer(name=descriptor.layer_name,
                                                        func=modeltools.NEURON_LOGISTIC,
                                                        input_name=input_name,
                                                        output_name=output_name,
                                                        a=1.0)
