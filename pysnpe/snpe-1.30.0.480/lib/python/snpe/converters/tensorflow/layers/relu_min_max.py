#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
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
    GraphSequence,
    NonConsumableConverterSequenceNode
)

class ReluMinMaxLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, min_clamp=0, max_clamp=0, output_names=None):
            super(ReluMinMaxLayerResolver.Descriptor, self).__init__(layer_type, name, nodes,
                                                                     output_names=output_names)
            self.min_clamp = min_clamp
            self.max_clamp = max_clamp

    def __init__(self):
        sequence_keras = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('root', ['Relu']),
            ConverterSequenceNode('min', ['Minimum']),
            ConverterSequenceNode('min_cast', ['Cast']),
            ConverterSequenceNode('min_const', ['Const']),
            ConverterSequenceNode('max', ['Maximum']),
            ConverterSequenceNode('max_const', ['Const'])
        ])
        sequence_keras.set_inputs('root', ['input'])
        sequence_keras.set_inputs('min_cast', ['min_const'])
        sequence_keras.set_inputs('min', ['root', 'min_cast'])
        sequence_keras.set_inputs('max', ['min', 'max_const'])
        sequence_keras.set_outputs(['max'])

        self.sequences = [sequence_keras]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                relu_op = match['root']

                min_const_op = match['min_const']
                max_const_op = match['max_const']
                min_value = graph_helper.evaluate_tensor_output(min_const_op.outputs[0])
                max_value = graph_helper.evaluate_tensor_output(max_const_op.outputs[0])

                # Only scalar values of min and max resolve to this pattern, else continue
                # and let the elementwise layers handle these ops
                if not np.isscalar(min_value) or not np.isscalar(max_value):
                    continue

                consumed_nodes = match.consumed_nodes
                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]

                potential_descriptors.append(
                    ReluMinMaxLayerResolver.Descriptor('ReluMinMax', str(relu_op.name),
                                                       consumed_nodes, min_value, max_value,
                                                       output_names=output_op_nodes_names))
        return potential_descriptors


class ReluMinMaxLayerBuilder(LayerBuilder):

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReluLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_neuron_layer(name=descriptor.layer_name,
                                                        func=modeltools.NEURON_RELU_MIN_MAX,
                                                        input_name=input_name,
                                                        output_name=output_name,
                                                        min_clamp=descriptor.min_clamp,
                                                        max_clamp=descriptor.max_clamp)
